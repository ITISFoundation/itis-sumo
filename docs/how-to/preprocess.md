# How to preprocess data for the Dakota engine

**Goal:** turn a dataframe with your own column names and scales into the
standardized `x1, x2, …` / `y1, y2, …` files the Dakota engine expects — and
turn results back the other way, into your original names and scales.

## Map and fit

```python
from itis_sumo.preprocess.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
preprocessor.setup_variables(
    input_vars=["length", "width"],
    output_vars=["stress"],
)  # length -> x1, width -> x2, stress -> y1

preprocessor.fit(train_raw)  # computes normalization stats, if any
train_processed = preprocessor.transform(train_raw)
train_processed.to_csv(training_file, sep=" ", index=False)
```

`setup_variables` alone is enough to run — normalization and sign-switching
are both opt-in.

## Add normalization or sign-switching (optional)

```python
preprocessor.setup_normalization(
    input_normalizations={"length": "z_score"},
    output_normalizations={"stress": "min_max"},
)
preprocessor.setup_sign_switching(output_sign_switches=["stress"])
```

Call these **before** `fit()` — `fit()` is what computes the actual
mean/std or min/max from your data; calling it after `fit()` leaves those
stats uncomputed and normalization silently does nothing (a warning is
logged, not an exception).

- `normalization_method`: `"z_score"` or `"min_max"` per variable.
- Sign-switching negates the variable both directions — useful when Dakota's
  variable-bound assumptions (e.g. MOGA's uniform lower/upper bounds) fit a
  positive-going convention better than your raw data's sign.

## Bring predictions back to your own scale

```python
predictions = {"y1": preds["y1_hat"]}  # mapped-name keys, as returned by evaluate_sumo
original_scale = preprocessor.inverse_transform(predictions)
# {"stress": [...]}  -- original name, denormalized/sign-restored
```

`inverse_transform` accepts a dict, a DataFrame, or an ndarray (ndarray
form needs column order to match `input_variables` then `output_variables`,
since there are no column names to key off of).

## Persist and reload (across processes / requests)

```python
preprocessor.save_config(run_dir / "preprocessor_config.json")

# ...later, possibly a different process...
reloaded = DataPreprocessor().load_config(run_dir / "preprocessor_config.json")
reloaded.inverse_transform(predictions)
```

`load_config` restores a preprocessor already marked fitted — you can call
`inverse_transform` on it directly without re-fitting, as long as the saved
config came from a preprocessor that was actually fitted.

## Filter variables without starting over

If you set up a preprocessor with more variables than you end up training
on:

```python
preprocessor.filter_by_names(input_names=["length"], exclude=True)  # drop "length"
# or: preprocessor.filter_by_patterns(input_patterns=[r"^geom_"])
```

This re-derives the `x1, x2, …` mapping over the remaining variables
(filtering renumbers the mapped names), so do it once, up front — not
between a `fit()`/`transform()` pair on the same run.

Full class reference: [Reference → Data preprocessing](../reference/preprocess.md).
