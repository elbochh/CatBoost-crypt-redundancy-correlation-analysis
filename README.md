# CatBoost Crypto: Clustered MAE Model with Expanding-Time Folds

This repository implements a **per-cluster CatBoost regression model** for crypto assets,  
with **leak-free expanding-time folds**, **correlation + redundancy-based feature selection**,  
and **strong baselines** for next-day log return prediction.

All evaluation is strictly out-of-sample in time, with metrics reported per fold and per cluster.

---

## Problem Setting

- **Universe:** Crypto symbols, each assigned to a **cluster** (e.g., by volatility/market-cap style).
- **Target:** Next-day log return  
  \[
  y_t = \log(\text{close}_{t+1}) - \log(\text{close}_t)
  \]
  taken from `y_ret_next` if present, or derived from `close`.

- **Goal:** For each cluster, train a CatBoost model to predict next-day log returns using engineered OHLCV features.

---

## Key Techniques in This Code

### 1. Per-Cluster Modeling

Data is split by `cluster` (from `ohlcv_clusters.csv`), and the full pipeline is run **independently** for each cluster:

- Separate folds and splits per cluster.
- Separate feature selection per cluster, per fold.
- Separate models and configs saved per cluster.

This allows each style bucket (e.g., “large-cap majors”, “small-cap alts”) to have its own tailored model.

---

### 2. Expanding-Time Folds (Months-Based)

The script builds **end-anchored, expanding folds** defined in calendar months:

- For each fold:
  - **Test window:** last `test_months` worth of months for that fold.
  - **Validation window:** `val_months` immediately before the test window.
  - **Train window:** **all history strictly before validation start** (expanding from the beginning).

By default:

- `n_folds = 6`
- `val_months = 3`
- `test_months = 2`

The folds are constructed so the **last fold’s test window is at the very end of the dataset**, and earlier folds move backwards in time in blocks of `test_months`. This gives you:

- Properly **forward-only**, time-respecting splits.
- Multiple OOS evaluation periods across the sample.

---

### 3. Feature Selection: Correlation + Redundancy (Train-Only)

For each fold (train data only):

1. **Start from a global feature pool**  
   All numeric columns (excluding IDs, target, and raw unstable columns like `price`, `volume_24h`, etc.).

2. **Correlation pruning (`remove_correlated_features`)**
   - Compute the correlation matrix on training data.
   - If `|ρ(i,j)| ≥ corr_threshold` (default `0.90`), drop the feature with **lower variance**.
   - This keeps the more informative one and avoids multicollinearity.

3. **Redundancy pruning (`remove_redundant_features`)**
   - For each remaining feature, try to predict it from all other features using **Ridge regression**.
   - If out-of-sample \(R^2 \geq\) `redundancy_r2` (default `0.70`), it is considered redundant and dropped.
   - Train/valid split for this step is internal to TRAIN (via `train_test_split`).

The script logs:

- How many features you start with.
- How many are dropped due to zero variance, correlation, and redundancy.
- The final list of selected features per fold, saved to:
  - `features_{cluster}__fold_{k}.txt`
  - `features_{cluster}__fold_{k}.json` (with debug info)

All selection is **train-only** to keep it leak-free.

---

### 4. Model: CatBoostRegressor (MAE)

The model is:

- **CatBoostRegressor** trained with **MAE loss**:
  ```python
  "cat_params": {
      "loss_function": "MAE",
      "eval_metric": "MAE",
      "iterations": 1200,
      "learning_rate": 0.035,
      "depth": 6,
      "l2_leaf_reg": 6.0,
      "bootstrap_type": "Bernoulli",
      "subsample": 0.8,
      "rsm": 0.8,
      "od_type": "Iter",
      "od_wait": 100,
      "random_seed": 1338,
      "verbose": False,
      "allow_writing_files": False,
  }
