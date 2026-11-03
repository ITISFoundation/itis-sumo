# SuMo handle

`fit()` trains a surrogate and hands you a handle; the handle's methods are
the **catalog** — everything queryable from the fitted model. The run
directory lives exactly as long as the handle does (V24af): close it (or exit
the `with` block) and the working files are discarded on success, preserved on
failure for diagnosis. `load()` reopens a saved model for prediction only.

::: itis_sumo.fit

::: itis_sumo.load

The handle class is exported as `SuMo`; its methods are documented below.

::: itis_sumo.surrogate._session.SumoSession
    options:
        heading: SuMo
