# Types & errors

Every result is a typed dataclass in your units under your names (V22rs);
every input spec you pass is a small dataclass too. Failures are typed
(`SumoInputError` for what you can fix in your table or config,
`SumoEngineError` carrying the surviving run directory for what Dakota did).

::: itis_sumo.types

::: itis_sumo.errors
