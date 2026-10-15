# DOCS_SPEC — itis-sumo documentation

Caveman-encoded (drop articles/filler; `→` becomes, `!` must, `?` may/uncertain, `⊥` never). Consolidation pass over the MkDocs site (Diátaxis-structured: tutorials/how-to/reference/explanation/about), triggered by a UX review of `docs/index.md` as the site's front door. Companion to the package `SPEC.md` — this one owns docs, not code.

## LINKS
- framework ! Diátaxis (diataxis.fr): tutorials=learning-oriented, how-to=task-oriented, reference=information-oriented, explanation=understanding-oriented — each section answers ONE of those, never mixes
- site config → `mkdocs.yml` nav tree = source of truth for section membership
- source review → Opus UX pass on `develop`'s `docs/index.md`, `docs/about/index.md`, `docs/reference/index.md` (2026-08-14): verdict = page has the right routing idiom but fails as a front door — no value prop, wrong lede, dead-weight install block, broken section links, missing visual
- scope: this SPEC covers docs shipped on `develop` (tutorials/{getting-started,examples}, how-to/{cross-validate,sensitivity-analysis,moga,preprocess}, reference/*, explanation/{surrogate-modeling,gaussian-processes}, about/*). NIH-in-silico/Merck-specific docs (`nih-in-silico.md`, `convergence-diagnostics.md`, `select-distribution-scale.md`) live only on the confidential `feat/nih-in-silico-example` incubator branch — ⊥ in scope here until individually ported (see root SPEC's promotion rule)
- V&V report distribution model (product decision, 2026-08-15): target format = standalone downloadable PDF, ⊥ in Nav either way (request-gated, not browsable). Current format = in-progress HTML page — landing-page line uses "available on request" framing already, but link target stays the HTML report until a real PDF deliverable exists; see TD11zb for the deferred swap to a `support@sim4life.io` contact flow

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
V3hd: every page shipped under `docs/` ! reachable from `docs/index.md`'s nav, directly or via a linked section index — ⊥ orphan pages with no inbound nav link (caught B4kp: `tutorials/examples.md` existed, unlinked)
V4lc: subagent-authored doc edits ! ⊥ invent external URLs absent from the source file being edited — `mkdocs --strict` validates internal links only, never external URL existence, so a fabricated external link passes the build oracle silently; parent thread ! diff-check any newly introduced external link against source before commit close-out (caught B8ry)

## §T
id|status|task|cites
TD1ef|x|`docs/index.md`: rewrite opening paragraph — lead with one plain-language sentence on the problem this solves and the payoff (simulation is slow → train a fast stand-in you can sweep/optimize/analyze, with honest error bars), THEN the existing feature list (UQ/Sobol/sampling/MOGA) as elaboration, not replacement|§G,V1ab
TD2gh|x|`docs/index.md`: delete the "Install & verify" section (`uv sync` / `itis-sumo validate` block) — it duplicates `tutorials/getting-started.md` step 1 and is reference/tutorial material, not front-door material; replace with a single one-line "Getting started" call-to-action sentence linking there|§C
TD3ij|x|`docs/index.md`: fix the "How-to guides" bullet — currently links only `how-to/cross-validate.md`; change to link all four recipes inline (cross-validate, sensitivity-analysis, moga, preprocess) so it represents the whole section|§C,V2cd
TD4kl|x|`docs/index.md`: fix the "Explanation" bullet — currently links only `explanation/surrogate-modeling.md`; change to link both pages inline (surrogate-modeling, gaussian-processes)|§C,V2cd
TD5mn|x|`docs/index.md`: add a new "Worked examples" bullet linking `tutorials/examples.md` (real Dakota fits against closed-form analytical answers, with figures) — currently unlinked from the landing page entirely|§C
TD6op|x|`docs/index.md`: embed `docs/assets/examples/gp_fit_uncertainty.png` near the top of the page with a one-line caption, so the payoff is visible, not just described|§C
TD7qr|x|`docs/index.md`: move the scope statement from `about/index.md`'s opening ("headless core, no Flask/oSPARC/UI dependency; for the web UI see `mmux_documentation`") onto the landing page, near the top, right after the opening paragraph|§C,I
TD8st|x|`docs/index.md`: promote the V&V trust-signal line (currently the closing sentence) further up the page and reframe it positively — drop the "not just 'didn't crash'" defensive phrasing, state directly what's been verified against analytical solutions|§C
TD9uv|x|`docs/about/index.md`: after TD7qr moves the scope statement out, trim the opening so `about/index.md` stops functioning as a second landing page — keep a short one-line pointer back instead of restating scope, keep provenance/engine/spec pointers as-is|§G,TD7qr
TD10wx|x|`mkdocs.yml`: remove the "Verification & Validation" section from Nav (report stays a built, linkable page — just not a first-class nav item); `docs/index.md`: reframe the V&V line from "see the report" to "available on request" — reads as a deliberate, serious offering rather than a defensive plea for trust|§C
TD11zb|.|`docs/index.md`: once the V&V report ships as a standalone downloadable PDF (not before — see LINKS decision), swap the "available on request" link target from `verification-validation.md` to a `mailto:support@sim4life.io` contact flow (or a direct download link, if self-serve becomes the model instead of request-gated) — ⊥ make this swap while the report is still the in-progress HTML page|§C

## §B
id|date|cause|fix
B1nq|2026-08-14|`docs/index.md` opened with a jargon feature list (UQ, Sobol', sampling, MOGA) before ever stating the problem solved or the payoff — no reason for a first-time visitor to keep reading|TD1ef (03ccdfb): rewrote opening to lead with a plain-language problem/payoff sentence, feature list kept as elaboration after it; V1ab
B2wr|2026-08-14|"Install & verify" section on the landing page duplicated `tutorials/getting-started.md` step 1 — reference/tutorial material occupying front-door space|TD2gh (89612f6): removed the install block, replaced with a single "Getting started" CTA sentence linking there; §C
B3fx|2026-08-14|"How-to guides" and "Explanation" nav bullets each hyperlinked one arbitrary member page (`cross-validate.md`, `surrogate-modeling.md`) standing in for the whole section, silently hiding the other 3 how-to recipes and 1 explanation page from a visitor scanning the landing page|TD3ij (128decf) + TD4kl (0fa93cc): linked all four how-to recipes and both explanation pages inline; V2cd
B4kp|2026-08-14|`tutorials/examples.md` (worked Dakota fits vs. closed-form analytical answers, with figures) shipped on the site but was never linked from the landing page — an orphan page with no discovery path from the front door|TD5mn (5db144f): added a "Worked examples" bullet; V3hd (new invariant drafted from this — nothing in §V/§C previously required landing-page reachability for shipped pages)
B5tz|2026-08-14|no visuals anywhere on the landing page despite `docs/assets/examples/gp_fit_uncertainty.png` already existing — the surrogate-with-error-bars payoff was described in prose but never shown|TD6op (576a6c9): embedded the figure near the top with a one-line caption; §C
B6vb|2026-08-14|the "is this for me?" scope statement (headless core, no Flask/oSPARC/UI dependency) sat only on `docs/about/index.md`, several clicks from the front door, instead of on first screen where a visitor decides whether to keep reading|TD7qr (165a9e9) + TD9uv (913bdde): copied the scope statement to the top of `docs/index.md`; trimmed `about/index.md`'s now-redundant restatement to a 1-line pointer back; §C
B7sm|2026-08-14|V&V trust signal (pipeline verified against known analytical solutions) was the closing sentence of the page, phrased defensively as a negation ("not just 'didn't crash'") instead of a direct positive claim — a proof point buried where fewest visitors scroll to it|TD8st (d08c25b): promoted the line to just below the opening/scope block, reframed as a direct positive statement; §C
B8ry|2026-08-15|Haiku subagent executing TD7qr fabricated an external GitHub URL for `mmux_documentation` (`https://github.com/ITISFoundation/mmux_documentation`) that appears nowhere in any source file — `about/index.md`'s own text only ever named it as unlinked plain text. `mkdocs --strict` built clean anyway since strict mode checks internal links/images only, never external URL existence, so the fabricated link would have shipped undetected without a manual diff review|caught in parent-thread review before task close-out, corrected in 0910343 (link removed, plain text kept, matching source); V4lc (new invariant: forbid subagent URL invention, require parent diff-check on new external links)

## §W (visual identity)
W-G: docs ! read as IT'IS family w/ TIP + S4L — shared workflow-diagram lang, card-grid landing, IT'IS footer, Inter font.
W-C: logo in nav (diamond mark + "SuMo" wordmark, `assets/logo.svg`); favicon `assets/favicon.svg`; primary = TIP blue `#0190d0`, accent = TIP indigo `#7280f5` (custom palette in `assets/css/fonts.css`); Inter font; footer via `overrides/partials/copyright.html` override (⊥ YAML `footer.links` — key does not exist); workflow component reused via `assets/css/workflow.css` + documented snippet (⊥ raw CSS per page); admonition ≤3 sentences + fixed vocab.
W-V:
W1ab: footer URLs ! invented/changed w/o verifying they resolve — reuse TIP's `itis.swiss` set (`/who-we-are/contact/`, ``, `/who-we-are/`), re-check on change (extends V4lc to the footer)
W2cd: every shipped page ! reachable from `docs/index.md` nav, directly or via section index (V3hd)
W3ef: workflow diagram / card grid ! hand-rolled per page — use the shared `.tip-workflow` component + Material `.grid.cards` (⊥ paste raw HTML/CSS)
W4gh: admonition ! exceed 3 sentences; type ! outside the 6-vocab set (info/warning/note/tip/example/danger)
W-T:
TW1ab|x|create `docs/overrides/` + wire `custom_dir`; add `overrides/partials/copyright.html` (TIP-style 3-link footer)|W-C,W1ab
TW2cd|x|create `assets/css/workflow.css` (.tip-workflow adapted from TIP VitePress CSS to Material vars); create `assets/css/fonts.css` (Inter + custom blue/indigo palette)|W-C
TW3ef|x|create `assets/logo.svg` (diamond + "SuMo") + `assets/favicon.svg` (S monogram)|W-C
TW4gh|x|`mkdocs.yml`: logo/favicon, custom primary/accent, features (navigation.tabs/footer, search.suggest/highlight, content.tabs.link, content.code.annotate), plugins: search, extra_css, rename nav → Quick Start/Workflows/Background/Reference/About|W-C
TW5ij|x|`docs/index.md` rewrite as card-grid hero (capability cards, 2 CTAs, embedded gp figure, scope, V&V line)|W-C,W3ef
TW6kl|x|`docs/reference/index.md`: replace ASCII pipeline w/ `.tip-workflow` component (Sampling→Config→Core→Evaluate + side branches)|W3ef
TW7mn|x|`docs/about/style-guide.md` (dev-facing, unlinked): documents workflow snippet, card grid, admonition vocab, Diátaxis rename|W-C
TW8op|x|apply admonition vocab to top pages; add §W to DOCS_SPEC|W4gh
TW9qr|x|build + `mkdocs build --strict` to verify (no broken links/assets)|W2cd,W1ab
