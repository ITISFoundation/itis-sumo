  How the wheel is built

  - Backend: py-build-cmake (custom build_backend/build.py wraps it to also run setuptools_scm for versioning) — driven by pyproject.toml + CMakeLists.txt.
  - Dakota source is not vendored in this git repo. make get-dakota-src (Makefile:86) clones snl-dakota/dakota at a pinned tag (currently v6.24.0), its submodules (packages/external, packages/pecos), a separate surfpack
  clone, then applies src_patches_v624/*.patch. I just ran this for you — it's now checked out at dakota/ (gitignored, 639MB).
  - CMake builds Dakota as a pybind11 extension module literally named environment (dakota/src/dakota_python.cpp), installed by itis_dakota's CMakeLists.txt into dakota/environment/ alongside dakota/src/__init__.py
  (from .environment import *). So the real runtime API users import is dakota.environment — a separate top-level package from itis_dakota (which is just a thin version-shim: itis_dakota/__init__.py only re-exports
  __version__).
  - cibuildwheel does the manylinux/macOS packaging; scripts/repair_wheel*.sh handle the auditwheel/delocate repair step.

  Typing stubs

  The recent py.typed commit only covers itis_dakota (which has near-zero public API). The actual functional surface — dakota.environment.study, Variables, Response, CommandLine, version(), etc. — is a compiled pybind11
  .so with no type information at all, and no py.typed marker in the dakota package. Type checkers treat all of it as Any.

  It's feasible to fix, and the API is small (dakota_python.cpp is only ~330 lines): either hand-write a .pyi for it, or (better, stays in sync automatically) run pybind11-stubgen against the built environment module as
  a post-build CI step, then have CMakeLists.txt's install(...) rules also ship the generated .pyi + a py.typed file into dakota/environment/.

  JSON schema → Pydantic

  Yes — dakota/src/dakota.json (82K lines, 1474 $defs) is a genuine JSON Schema (title: "DakotaStudy", $defs, $ref, anyOf, additionalProperties: false, x-materialization metadata mapping each field to its C++
  ProblemDescDB key/type). It's the real, currently-accepted schema for Dakota's new JSON study-input format — confirmed via LibraryEnvironment(const nlohmann::json&) constructors in
  dakota_python.cpp/LibraryEnvironment.hpp, and tests/simple/simple.json + tests/simple/test_simple.py in this repo, which already exercises dakenv.study(callbacks=..., input_json=...).

  I ran a proof of concept:
  uv run --with datamodel-code-generator datamodel-codegen \
    --input dakota/src/dakota.json --input-file-type jsonschema \
    --output-model-type pydantic_v2.BaseModel
  It generated a valid, importable ~30K-line pydantic_v2 module out of the box, with a root DakotaStudy model. Validating it against the repo's own tests/simple/simple.json fixture mostly works but isn't 100% faithful
  yet — e.g. Dakota's JSON materializer accepts presence-only keywords as {} (as in "lhs": {}), while the schema declares them as Literal[True]. That's a small, fixable gap (a BeforeValidator coercing {} → True for
  presence flags), not a fundamental blocker.

  So the practical path: schema-generate Pydantic models from dakota.json as a build step (like the stubs), patch the handful of presence-flag quirks, and ship it as an optional itis_dakota.models (or separate)
  subpackage — giving downstream users IDE autocomplete and validation for Dakota JSON study configs.

Engine regression: Interface::interface_cache (6.23+) — pinned out with ==1.5.9

  - 6.20 (wheel 1.5.9, current parity pin): the study() ctor used problem_description_db().interface_list() (proven by src_patches_v621/dakenv_restart.patch) — no static cache, surrogate-only confs work.
  - 6.23 (fork commit f62e240, 2026-04-27) introduced Interface::interface_cache(const ProblemDescDB*): a static std::map<const ProblemDescDB*, std::list<std::shared_ptr<Interface>>> (DakotaInterface.hpp:71), populated ONLY by Interface::get_interface
    from the SimulationModel/NestedModel ctors (SimulationModel.cpp:25, NestedModel.cpp:86). Pure data-fit surrogate confs (surrogate global + import_build_points_file, no interface block) never construct a model/interface → the
    map entry is missing → DakotaInterface.cpp:77 .at(study_ptr) throws std::out_of_range (Cerr: "Interface::interface_cache() called with nonexistent study!") → surfaces as IndexError: map::at inside the study() constructor,
    before execute(). Confirmed empirically on wheel 6.24.3: callback+sampling confs (SimulationModel path) run fine; interface-less confs crash.
  - Upstream fix candidate (3-line): make interface_cache() get-or-create (operator[]) instead of .at() in DakotaInterface.cpp:74-82. Applied locally then reverted (git checkout in the nested dakota repo) to keep the fork pristine.
  - Do NOT route around via CommandLine(...).execute(): it bypasses the buggy ctor BUT abort_mode defaults to ABORT_EXITS (dakota_global_defs.cpp:50) and CommandLine never sets exit_mode("throw") → a failed fold std::exit()s the
    whole process (kills the B22/B23 warn-and-continue resilience). Only study()/create_libEnv sets exit_mode("throw").
  - Upgrade ladder (SPEC T16mo): 1.5.9 (6.20) → 1.5.11 (packaging-only rebuild: drop py3.8, add cp313 wheels; same 6.20 engine → behavior-identical) → 6.24.x co-shipped with the JSON input seam, once the regression is fixed
    upstream. 1.5.9 ships no cp313 wheel (max cp312) — that is why the parity pin forces py 3.11.
  - NKM bounds stderr noise: surrogate builds via "gaussian_process surfpack" print "You didn't enter the right number of lower/upper bounds" (packages/surfpack/src/surfaces/nkm/NKM_KrigingModel.cpp:121,132). Benign: the
    assert(false) is compiled out in the release wheel (NDEBUG), results are correct, and flaskapi emits the identical message (byte-identical environment.so, md5 30f09e1e2fbb492483d997f6e319a2e6). Not a shim target.