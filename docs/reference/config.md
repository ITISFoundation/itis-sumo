# Config — NIDR string composers

`itis_sumo.config.funs_create_dakota_conf` builds Dakota's NIDR
(Named Input, Reduced) input-file syntax as plain Python f-strings — no
templating engine, no intermediate object model. This is deliberate (`V5ui`
in [`SPEC.md`](../about/spec.md)): it's the proven, near-verbatim port of the
mmux/vite flaskapi composers, with a seam left for a JSON/pydantic input
layer later (see the Dakota 6.24 experimental JSON-input work referenced in
[`DAKOTA-STUBS.md`](https://github.com/ITISFoundation/itis-sumo/blob/main/DAKOTA-STUBS.md)).

## Building blocks

Small composers, each returning a NIDR fragment string, assembled by the
higher-level `create_*_conffile` functions:

| Function | NIDR block |
|---|---|
| `start_dakota_file` | `environment` block + `tabular_data_file` |
| `add_continuous_variables` | `variables` block (continuous design vars) |
| `add_responses` | `responses` block from output descriptors |
| `add_surrogate_model` | `model` block wrapping a Gaussian-process surrogate over training data — see [How Gaussian Processes work](../explanation/gaussian-processes.md) for the theory behind this block |
| `add_sampling_method` | `method` block for LHS/sampling studies |
| `add_evaluation_method` | `method` block for a surrogate-evaluation-only study |
| `add_moga_method` | `method` block for MOGA multi-objective optimization |
| `add_evaluator_model`, `add_adaptive_sampling` | supporting model/sampling fragments |

## Study-level composers

Each combines the fragments above into a full input string for one Dakota
study type:

- `create_sumo_evaluation_conffile` — fit a surrogate, evaluate it at
  supplied points (`evaluate_sumo`'s NIDR).
- `create_sumo_crossvalidation_conffile` — Dakota's built-in cross-validation
  study.
- `create_sumo_manual_crossvalidation_conffile` — K-fold done in Python
  (`sklearn.model_selection.KFold`) around repeated evaluation calls, used
  because Dakota's built-in CV has a parsing bug (see
  [Verification & Validation § Known limitations](../verification-validation.md#known-limitations)).
- `create_uq_propagation_conffile` — Dakota-native uncertainty propagation
  (normal-uncertain variables only).
- `create_moga_optimization_conffile` — multi-objective genetic algorithm
  study.

## `add_surrogate_model` and the `has_eval_id_column` question

The training file handed to `add_surrogate_model` may or may not carry a
leading `eval_id` tabular column, depending on how it was staged. The
currently shipped heuristic (`infer_has_eval_id_column_from_filename`)
guesses from whether the substring `"processed"` appears in the training
file's *name* — a workaround, not a first-class parameter. This caused a
real bug (`SPEC.md` `B2`): staging the re-import training file under a name
that didn't contain `"processed"` desynced Dakota's `use_variable_labels`
expectation from the actual CSV columns. The current fix keeps the
substring convention alive rather than removing the heuristic
(`sumo_model_store.py` names its staged files `{id}.processed_training.dat`).

The tracked root-cause fix — an explicit `has_eval_id_column: bool` param
supplied by whichever caller staged the file, replacing filename sniffing
entirely — is proposed in `SPEC.md` as `V15zx`/`T17bq` but not yet the
committed behavior; see `SPEC.md` §V/§T for current status before relying
on either code path.
