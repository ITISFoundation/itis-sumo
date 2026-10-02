# Abolish `itis_sumo.api`; the root `__all__` is the public manifest

With the data-first restructure (ADR-0001) the `api` package's fused one-shots
largely die, and a blessed consumer namespace would be the *third* public surface
beside domain packages and the report façade. Decision: `itis_sumo.api` is abolished
(T56jw); the public, versioned, cross-repo surface is exactly root
`itis_sumo.__all__` ∪ `SuMo` methods ∪ `report.__all__` (V50kv), with ≤2 public
functions per domain module as the partition forcing-function.

**Considered options:** api shrinks to handle+types+errors (rejected — keeps a
namespace whose distinguishing job the root manifest now does); underscore-private
internals (rejected — still allows two-namespaces-deep cross-repo imports);
docs-as-manifest (rejected — the public line would live in prose, untestable).

**Consequences:** V16qf's and V49rp's consumer rules survive re-pointed at the
manifest (CI import-surface test); flaskapi migrates via one coordinated clean break
(T63zw), no deprecation shim — the shim would re-materialize the abolished module and
give flaskapi's 522 tests a reason to never move.
