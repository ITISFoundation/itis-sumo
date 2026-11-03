# Report renderings

The presentation door (V54hz): each rendering consumes exactly **one**
function result, never runs the engine, never recomputes a statistic, and
returns a plotly `Figure` or a pandas `DataFrame` — a deletable "serving
suggestion". kwargs pass through to the backend; the escape hatch is always
the returned object. `sections()` is the catalog of what exists.

Requires the optional extra: `pip install itis-sumo[report]`.

::: itis_sumo.report

## The section catalog

::: itis_sumo.report.sections

A notebook pattern — data cell, rendering cell, version stamp — is worked
through in [Getting started](../tutorials/getting-started.md) and in
[`examples/report_notebook_template.py`](https://github.com/ITISFoundation/itis-sumo/blob/develop/examples/report_notebook_template.py).
