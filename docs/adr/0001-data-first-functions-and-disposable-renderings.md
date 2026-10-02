# Data-first surface: functions return data, renderings are disposable

Consumers of automated reports rejected "magic functions that do everything in an
unspecified way": the package's value is trustworthy *data* plus honest estimators,
so every public function returns the data (typed dataclass fields — the fields ARE
the data), and presentation lives in separate, opinionated **renderings** that take
one result, pass documented kwargs through to the backend (plotly / pandas), and may
be tweaked or deleted without leaving the notebook. Chosen over deep all-inclusive
report verbs (`render_report(model, config)` — opaque) and over pasted full-chunk
notebook code (transparent but impossible to upgrade).

**Consequences:** any function that consumes one query's output to serve another's
summary while hiding the middle is retired (V52rc — `evaluate_uncertainty`,
`evaluate_correlations` die into open compositions; draw-returning results must keep
the raw draws, V53md); docstrings of functions are the contract and get auto-rendered
into the docs; the report layer regenerates notebooks as data-cell + rendering-cell
pairs whose heavy logic stays in the package (upgradeable) while every cell stays
editable (transparent).
