"""The fitted SuMo: the catalog lives on the handle (V50kv, V52rc).

`fit` trains and hands back a live SuMo whose methods are the catalog
(`cross_validate`/`along_axes`/`grid`/`sobol`/`uncertainty`/`predict`/`save`);
`load` reopens a stored one (E1).
"""

from itis_sumo.surrogate._session import SuMo, SumoSession, fit, load

__all__ = ["SuMo", "SumoSession", "fit", "load"]
