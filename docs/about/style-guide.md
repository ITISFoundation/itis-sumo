# Docs style guide (dev-facing)

Internal guide for authoring itis-sumo documentation. Mirrors the visual
language of the TIP and S4L family manuals so the three sites read as one
IT'IS Foundation family. **This page is not linked from the public nav** — it
is for contributors, not end users.

## Brand

- Primary color: TIP blue `#0190d0`. Accent: TIP indigo `#7280f5`. Font: Inter.
- Logo: `assets/logo.svg` (diamond mark + "SuMo" wordmark). Favicon: `assets/favicon.svg`.
- Footer (overridden in `overrides/partials/copyright.html`): "Contact IT'IS ·
  IT'IS Website · About IT'IS" + "© IT'IS Foundation". Footer URLs are
  external IT'IS Foundation links — **never invent or change them without
  verifying they resolve** (see `DOCS_SPEC.md` V4lc).

## Workflow diagrams

Use the `.tip-workflow` component (`assets/css/workflow.css`) instead of ASCII
art or Mermaid for pipeline/step flows. It adapts to light/dark automatically.

```html
<div class="tip-workflow">
  <a class="tip-workflow__step twf-blue" href="sampling.md">Sampling</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-indigo" href="config.md">Config</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-orange" href="core.md">Core</a>
  <span class="tip-workflow__arrow">→</span>
  <a class="tip-workflow__step twf-teal" href="evaluate.md">Evaluate</a>
</div>
```

Color map: `twf-blue` `#0190d0`, `twf-indigo` `#7280f5`, `twf-orange`
`#f08a35`, `twf-teal` `#14c4b0`, `twf-rose` `#f06292`, `twf-amber` `#f5a623`,
`twf-purple` `#a77bf0`. Non-clickable steps: add `twf-disabled` and drop the
`href`. **Do not paste raw CSS** — the component is global once `workflow.css`
is loaded.

## Card grids (landing / overview pages)

Use Material's built-in card grid instead of tables for capability lists:

```markdown
<div class="grid cards" markdown>

- :material-flask: __Train surrogates__
  Train on simulation data, in-process via the Dakota wheel.

- :material-chart-line: __Quantify sensitivity__
  Sobol' indices and UQ propagation with `scipy.stats`.

</div>
```

## Admonitions

Keep each admonition to **≤ 3 sentences**. Vocabulary:

| Type | Use for |
|------|---------|
| `!!! info "Key concept"` | One-line definition |
| `!!! warning "Invariant"` | Hard constraint (cite the §V id) |
| `!!! note "Research"` | §R finding with source |
| `!!! tip "Quick start"` | Actionable step / shortcut |
| `!!! example "Worked example"` | Runnable code + expected output |
| `!!! danger "Known issue"` | §B bug / regression |

## Diátaxis discipline

- **Quick Start** (was Tutorials): learning-oriented.
- **Workflows** (was How-to): task-oriented recipes.
- **Reference**: information-oriented, module-by-module.
- **Background** (was Explanation): understanding-oriented.
- **About**: provenance, porting, the living spec.

Every shipped page must be reachable from `index.md` (V3hd).
