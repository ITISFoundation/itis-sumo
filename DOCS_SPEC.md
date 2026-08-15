# DOCS_SPEC — itis-sumo documentation

Caveman-encoded (drop articles/filler; `→` becomes, `!` must, `?` may/uncertain, `⊥` never). Consolidation pass over the MkDocs site (Diátaxis-structured: tutorials/how-to/reference/explanation/about), triggered by a UX review of `docs/index.md` as the site's front door. Companion to the package `SPEC.md` — this one owns docs, not code.

## LINKS
- framework ! Diátaxis (diataxis.fr): tutorials=learning-oriented, how-to=task-oriented, reference=information-oriented, explanation=understanding-oriented — each section answers ONE of those, never mixes
- site config → `mkdocs.yml` nav tree = source of truth for section membership
- source review → Opus UX pass on `develop`'s `docs/index.md`, `docs/about/index.md`, `docs/reference/index.md` (2026-08-14): verdict = page has the right routing idiom but fails as a front door — no value prop, wrong lede, dead-weight install block, broken section links, missing visual
- scope: this SPEC covers docs shipped on `develop` (tutorials/{getting-started,examples}, how-to/{cross-validate,sensitivity-analysis,moga,preprocess}, reference/*, explanation/{surrogate-modeling,gaussian-processes}, about/*). NIH-in-silico/Merck-specific docs (`nih-in-silico.md`, `convergence-diagnostics.md`, `select-distribution-scale.md`) live only on the confidential `feat/nih-in-silico-example` incubator branch — ⊥ in scope here until individually ported (see root SPEC's promotion rule)

## §G
`docs/index.md` ! work as a front door: state what a visitor can DO here + WHY it matters + route them onward, skimmable in under one screen. ⊥ technical detail (that's reference's job), ⊥ lengthy explanation/rationale (that's explanation's or tutorials'). Every Diátaxis section stays in its lane end to end.

## §C
- landing page ! no API names, function signatures, config syntax, install/CLI transcripts — those belong in reference/tutorials
- landing page ! no multi-paragraph rationale or theory — that belongs in explanation
- routing bullets ! point at a section's actual full scope (an index page, or all its members inline) — ⊥ one arbitrary member page standing in for the whole section
- existing visual assets (`docs/assets/examples/*.png`) ! surfaced where they answer "what do I get", not left orphaned in a folder
- the "is this for me?" scope statement (headless core, no Flask/oSPARC/UI dependency; web UI lives in `mmux_documentation`) ! visible on first screen of the landing page, not buried in `about/`

## §I
- `docs/index.md` — the landing page, primary entry point
- `docs/about/index.md` — secondary page currently carrying content (scope statement) that belongs on the landing page instead

## §V
V1ab: `docs/index.md` ! opens with a 1-sentence problem/payoff statement before any acronym (UQ/Sobol/MOGA/LHS), routes to all 5 Diátaxis sections + the V&V report, ⊥ owns API/config detail or extended rationale itself
V2cd: every landing-page section bullet ! links either that section's index page or all of its member pages — ⊥ a single arbitrary member page standing in for the section

## §T
id|status|task|cites
TD1ef|x|`docs/index.md`: rewrite opening paragraph — lead with one plain-language sentence on the problem this solves and the payoff (simulation is slow → train a fast stand-in you can sweep/optimize/analyze, with honest error bars), THEN the existing feature list (UQ/Sobol/sampling/MOGA) as elaboration, not replacement|§G,V1ab
TD2gh|x|`docs/index.md`: delete the "Install & verify" section (`uv sync` / `itis-sumo validate` block) — it duplicates `tutorials/getting-started.md` step 1 and is reference/tutorial material, not front-door material; replace with a single one-line "Getting started" call-to-action sentence linking there|§C
TD3ij|x|`docs/index.md`: fix the "How-to guides" bullet — currently links only `how-to/cross-validate.md`; change to link all four recipes inline (cross-validate, sensitivity-analysis, moga, preprocess) so it represents the whole section|§C,V2cd
TD4kl|x|`docs/index.md`: fix the "Explanation" bullet — currently links only `explanation/surrogate-modeling.md`; change to link both pages inline (surrogate-modeling, gaussian-processes)|§C,V2cd
TD5mn|x|`docs/index.md`: add a new "Worked examples" bullet linking `tutorials/examples.md` (real Dakota fits against closed-form analytical answers, with figures) — currently unlinked from the landing page entirely|§C
TD6op|x|`docs/index.md`: embed `docs/assets/examples/gp_fit_uncertainty.png` near the top of the page with a one-line caption, so the payoff is visible, not just described|§C
TD7qr|x|`docs/index.md`: move the scope statement from `about/index.md`'s opening ("headless core, no Flask/oSPARC/UI dependency; for the web UI see `mmux_documentation`") onto the landing page, near the top, right after the opening paragraph|§C,I
TD8st|.|`docs/index.md`: promote the V&V trust-signal line (currently the closing sentence) further up the page and reframe it positively — drop the "not just 'didn't crash'" defensive phrasing, state directly what's been verified against analytical solutions|§C
TD9uv|.|`docs/about/index.md`: after TD7qr moves the scope statement out, trim the opening so `about/index.md` stops functioning as a second landing page — keep a short one-line pointer back instead of restating scope, keep provenance/engine/spec pointers as-is|§G,TD7qr

## §B
id|date|cause|fix
(none yet)
