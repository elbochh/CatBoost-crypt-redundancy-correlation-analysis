
"""
Per-cluster CatBoost (MAE) with time-series CV (but run only the most recent fold),
SHAP-based diverse feature selection, plus GA hyperparameter search (per cluster).
Logs GA live, prints winning features+params, evaluates on last 60d test,
and saves the final model (.cbm) + a pickle copy with .pt suffix (not PyTorch).
"""

import json
import re
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error as rmse_func
except Exception:
    rmse_func = None

warnings.filterwarnings("ignore", message=".*feature_perturbation='interventional'.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*'squared' is deprecated.*", category=FutureWarning)

# --- Model: CatBoost ---
from catboost import CatBoostRegressor, Pool
import joblib  

# ==============================
#            CONFIG
# ==============================
CONFIG = {
    # File paths
    "features_file": "features_crypto_ohlcv.csv",
    "latest_file":   "latest_day_features.csv",
    "clusters_file": "ohlcv_clusters.csv",
    "out_dir":       "out_catboost_clusters",

    # CV / windowing (we will ONLY use the most recent fold)
    "n_folds": 5,
    "train_len": 999999,   
    "val_len":   90,      
    "test_len":  60,      

    # Feature selection via SHAP-like importance
    "top_features_per_fold": 30,   
    "max_lags_per_base": 4,        # limit how many lags of same base feature
    "lag_suffix_regex": r"_lag(\d+)$",

    # CatBoost base params (we’ll GA-tune main knobs around these)
    "cat_base_params": {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 2000,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 6.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "rsm": 0.8,
        "od_type": "Iter",
        "od_wait": 200,
        "random_seed": 1338,
        "verbose": False
    },

    "early_stopping_rounds": 200,
    "use_best_model": True,

    # GA hyperparams (per-cluster, only on the most-recent fold)
    "ga": {
        "pop_size": 25,
        "n_gen": 6,
        "elite_k": 6,
        "tourn_size": 3,       
        "mut_prob": 0.35,     
        "seed": 1338
    },

    # How many recent in-sample days to log with actuals next to pred
    "recent_days_log": 14,
}

# ==============================
#        HELPERS
# ==============================

ID_COLS = ["symbol", "date"]
DROP_ALWAYS = set([
    "time_open","time_close","time_high","time_low","timestamp","price",
    "percent_change_1h","percent_change_24h","percent_change_7d","percent_change_30d",
    "volume_24h","total_supply","circulating_supply","fear_greed_value",
    "btc_dominance","eth_dominance","active_cryptocurrencies","active_exchanges","active_market_pairs",
    "total_market_cap_usd","total_volume_24h_usd","total_volume_24h_reported_usd"
])

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df

def detect_target(df: pd.DataFrame) -> str:
    if "y_ret_next" in df.columns:
        return "y_ret_next"
    if "close" in df.columns:
        return "__tmp_y"
    raise ValueError("No target found and 'close' missing to derive y_ret_next.")

def build_feature_list(df: pd.DataFrame, target_col: str) -> List[str]:
    drop_cols = set(ID_COLS + [target_col]) | DROP_ALWAYS
    feats = [c for c in df.columns if c not in drop_cols]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    return feats

def compute_baselines(df_fold: pd.DataFrame, target_col: str) -> Tuple[pd.Series, pd.Series]:
    if "ret_1d" in df_fold.columns:
        r = df_fold["ret_1d"]
    elif "close" in df_fold.columns:
        r = df_fold.groupby("symbol")["close"].pct_change(1)
    else:
        r = pd.Series(index=df_fold.index, dtype=float)

    r_lag1 = r.groupby(df_fold["symbol"]).shift(1)
    bl1 = r_lag1
    bl5 = (r_lag1.groupby(df_fold["symbol"])
                  .apply(lambda s: s.rolling(5, min_periods=1).mean())
                  .reset_index(level=0, drop=True))
    return bl1, bl5

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = rmse_func(y_true, y_pred) if rmse_func else mean_squared_error(y_true, y_pred, squared=False)
    wr = float(np.mean(np.sign(y_true) == np.sign(y_pred))) if len(y_true) else np.nan
    return {"MAE": mae, "RMSE": rmse, "WR": wr}

def metrics_safe(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "WR": np.nan, "_n": 0, "_dropped": int(len(y_true))}
    base = metrics(y_true[mask], y_pred[mask])
    base["_n"] = int(mask.sum())
    base["_dropped"] = int((~mask).sum())
    return base

def cat_split_params(cfg_params: Dict) -> Tuple[Dict, Dict]:
    fit_keys = {"early_stopping_rounds", "use_best_model", "verbose"}
    fit_params = {k: cfg_params[k] for k in list(cfg_params.keys()) if k in fit_keys}
    est_params = {k: v for k, v in cfg_params.items() if k not in fit_keys}
    est_params.setdefault("allow_writing_files", False)
    return est_params, fit_params

def cat_fit_with_es(X_tr, y_tr, X_val, y_val, cat_params: Dict) -> CatBoostRegressor:
    est_params, fit_params = cat_split_params(cat_params)
    fit_params.setdefault("use_best_model", CONFIG["use_best_model"])
    fit_params.setdefault("early_stopping_rounds", CONFIG["early_stopping_rounds"])
    fit_params.setdefault("verbose", False)
    model = CatBoostRegressor(**est_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), **fit_params)
    return model

def cat_make(cfg_params: Dict) -> CatBoostRegressor:
    est_params, _ = cat_split_params(cfg_params)
    return CatBoostRegressor(**est_params)

def cat_shap_rank(model: CatBoostRegressor,
                  X_val: pd.DataFrame,
                  y_val: Optional[np.ndarray],
                  X_bg: Optional[pd.DataFrame],
                  max_rows_bg: int = 2000,
                  max_rows_sv: int = 5000) -> pd.Series:
    try:
        if X_bg is None or len(X_bg) == 0:
            X_bg = X_val
        if X_bg is not None and len(X_bg) > max_rows_bg:
            X_bg = X_bg.sample(max_rows_bg, random_state=CONFIG["ga"]["seed"])
        XS = X_val
        if XS is not None and len(XS) > max_rows_sv:
            XS = XS.sample(max_rows_sv, random_state=CONFIG["ga"]["seed"])

        if XS is not None and y_val is not None and len(XS) == len(y_val):
            pool = Pool(XS, y_val)
            shap_vals = model.get_feature_importance(data=pool, type="ShapValues")
            shap_core = np.abs(shap_vals[:, :-1]).mean(axis=0)  # drop expected_value col
            return pd.Series(shap_core, index=XS.columns).sort_values(ascending=False)
    except Exception:
        pass

    try:
        pvc = model.get_feature_importance(type="PredictionValuesChange")
        return pd.Series(pvc, index=X_val.columns).sort_values(ascending=False)
    except Exception:
        return pd.Series(0.0, index=X_val.columns)

def enforce_diverse_lags(ranked: pd.Series, max_features: int, max_per_base: int, lag_regex: str) -> List[str]:
    pat = re.compile(lag_regex)
    chosen: List[str] = []
    per_base: Dict[str, int] = {}

    def base_name(name: str) -> str:
        m = pat.search(name)
        return name[:m.start()] if m else name

    for f in ranked.index:
        b = base_name(f)
        cnt = per_base.get(b, 0)
        if cnt >= max_per_base:
            continue
        chosen.append(f)
        per_base[b] = cnt + 1
        if len(chosen) >= max_features:
            break
    return chosen

def make_windows(unique_dates: List[pd.Timestamp],
                 n_folds: int,
                 train_len: int,
                 val_len: int,
                 test_len: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    unique_dates = sorted(pd.to_datetime(pd.Series(unique_dates)).unique())
    total_len = len(unique_dates)
    folds = []

    if total_len < (val_len + test_len + 1):
        return folds

    # Build most-recent folds first
    for i in range(n_folds, 0, -1):
        test_end_idx = total_len - (n_folds - i) * test_len - 1
        test_start_idx = test_end_idx - test_len + 1
        if test_start_idx < 0:
            continue

        val_end_idx = test_start_idx - 1
        val_start_idx = val_end_idx - val_len + 1
        if val_start_idx < 0:
            continue

        train_end_idx = val_start_idx - 1
        if train_end_idx < 0:
            continue

        if train_len >= total_len:
            train_start_idx = 0
        else:
            train_start_idx = max(0, train_end_idx - train_len + 1)

        if not (train_start_idx <= train_end_idx < val_start_idx <= val_end_idx < test_start_idx <= test_end_idx):
            continue

        folds.append((
            unique_dates[train_start_idx],
            unique_dates[train_end_idx],
            unique_dates[val_end_idx],
            unique_dates[test_end_idx],
        ))
    return folds

def slice_by_dates(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[m].copy()

def print_fold_header(cluster_id, dates):
    tr_s, tr_e, val_e, te_e = dates
    print(f"\n[Cluster {cluster_id}] (Most recent fold only)"
          f" | Train: {tr_s.date()}→{tr_e.date()}  "
          f"| Val end: {val_e.date()}  | Test end: {te_e.date()}")

# -------------------- GA: CatBoost param search --------------------

# Search space ranges 
                               
CAT_SPACE = {
    "depth":          (4, 10),        # int
    "learning_rate":  (0.01, 0.15),   # float
    "l2_leaf_reg":    (1.0, 12.0),    # float
    "subsample":      (0.5, 1.0),     # float
    "rsm":            (0.5, 1.0),     # float
    "iterations":     (800, 3000),    # int
}

def _rand_int(lo, hi):     return int(np.random.randint(lo, hi + 1))
def _rand_float(lo, hi):   return float(np.random.uniform(lo, hi))

def ga_random_params(base: Dict) -> Dict:
    p = dict(base)
    p["depth"]         = _rand_int(*CAT_SPACE["depth"])
    p["learning_rate"] = _rand_float(*CAT_SPACE["learning_rate"])
    p["l2_leaf_reg"]   = _rand_float(*CAT_SPACE["l2_leaf_reg"])
    p["subsample"]     = _rand_float(*CAT_SPACE["subsample"])
    p["rsm"]           = _rand_float(*CAT_SPACE["rsm"])
    p["iterations"]    = _rand_int(*CAT_SPACE["iterations"])
    return p

def ga_mutate(p: Dict, mut_prob: float) -> Dict:
    q = dict(p)
    if np.random.rand() < mut_prob:
        q["depth"] = int(np.clip(q["depth"] + np.random.randint(-2, 3), CAT_SPACE["depth"][0], CAT_SPACE["depth"][1]))
    if np.random.rand() < mut_prob:
        q["learning_rate"] = float(np.clip(q["learning_rate"] * np.exp(np.random.uniform(-0.3, 0.3)),
                                           CAT_SPACE["learning_rate"][0], CAT_SPACE["learning_rate"][1]))
    if np.random.rand() < mut_prob:
        q["l2_leaf_reg"] = float(np.clip(q["l2_leaf_reg"] + np.random.uniform(-1.5, 1.5),
                                         CAT_SPACE["l2_leaf_reg"][0], CAT_SPACE["l2_leaf_reg"][1]))
    if np.random.rand() < mut_prob:
        q["subsample"] = float(np.clip(q["subsample"] + np.random.uniform(-0.15, 0.15),
                                       CAT_SPACE["subsample"][0], CAT_SPACE["subsample"][1]))
    if np.random.rand() < mut_prob:
        q["rsm"] = float(np.clip(q["rsm"] + np.random.uniform(-0.15, 0.15),
                                 CAT_SPACE["rsm"][0], CAT_SPACE["rsm"][1]))
    if np.random.rand() < mut_prob:
        q["iterations"] = int(np.clip(q["iterations"] + np.random.randint(-300, 301),
                                      CAT_SPACE["iterations"][0], CAT_SPACE["iterations"][1]))
    return q

def ga_crossover(p1: Dict, p2: Dict) -> Dict:
    child = {}
    for k in ["depth","learning_rate","l2_leaf_reg","subsample","rsm","iterations"]:
        child[k] = p1[k] if np.random.rand() < 0.5 else p2[k]
    # carry through the rest (fixed keys)
    fixed = {k: v for k, v in p1.items() if k not in child}
    fixed.update({k: v for k, v in p2.items() if k not in child and k not in fixed})
    child.update(fixed)
    return child

def ga_eval_params(params: Dict,
                   X_tr: pd.DataFrame, y_tr: np.ndarray,
                   X_val: pd.DataFrame, y_val: np.ndarray) -> Tuple[float, float, CatBoostRegressor]:
    """Returns (val_MAE, val_WR, model) with early stopping."""
    model = cat_fit_with_es(X_tr, y_tr, X_val, y_val, params)
    pred_val = model.predict(X_val)
    m = metrics_safe(y_val, pred_val)
    return float(m["MAE"]), float(m["WR"]), model

def tournament_select(pop, fitness, k):
    """Return index of winner from a tournament of size k."""
    idxs = np.random.choice(len(pop), size=k, replace=False)
    best = min(idxs, key=lambda i: fitness[i][0])  # lower MAE better
    return best

# ==============================
#            MAIN
# ==============================
def main():
    cfg = CONFIG
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load data ----------
    feats = pd.read_csv(cfg["features_file"])
    feats = ensure_datetime(feats).sort_values(["symbol", "date"]).reset_index(drop=True)

    latest = pd.read_csv(cfg["latest_file"])
    latest = ensure_datetime(latest).sort_values(["symbol", "date"]).reset_index(drop=True)

    clusters = pd.read_csv(cfg["clusters_file"])
    if "cluster" not in clusters.columns or "symbol" not in clusters.columns:
        raise ValueError("clusters file must have columns: symbol, cluster")
    clusters["symbol"] = clusters["symbol"].astype(str)
    feats["symbol"] = feats["symbol"].astype(str)
    latest["symbol"] = latest["symbol"].astype(str)

    # Merge cluster labels
    feats = feats.merge(clusters, on="symbol", how="inner")
    latest = latest.merge(clusters, on="symbol", how="inner")

    # target
    target_col = detect_target(feats)
    if target_col == "__tmp_y":
        # derive next-day return
        feats = feats.sort_values(["symbol","date"]).copy()
        feats[target_col] = feats.groupby("symbol")["close"].pct_change(-1)

    all_features = build_feature_list(feats, target_col)
    feats = feats.dropna(subset=[target_col])

    # for baselines, ensure ret_1d exists or compute
    if "ret_1d" not in feats.columns and "close" in feats.columns:
        feats["ret_1d"] = feats.groupby("symbol")["close"].pct_change(1)

    cluster_ids = sorted(feats["cluster"].unique())

    # Save a copy of config
    (out_dir / "global_config.json").write_text(json.dumps(cfg, indent=2))

    for clus in cluster_ids:
        dfc = feats[feats["cluster"] == clus].copy()
        if dfc.empty:
            print(f"[Cluster {clus}] No rows, skipping.")
            continue

        uniq_dates = sorted(dfc["date"].unique())
        folds = make_windows(
            uniq_dates,
            n_folds=cfg["n_folds"],
            train_len=cfg["train_len"],
            val_len=cfg["val_len"],
            test_len=cfg["test_len"]
        )
        if not folds:
            print(f"[Cluster {clus}] Not enough dates to build {cfg['n_folds']} folds. Skipping.")
            continue

        # -------------------------------------------------------------
        # Only use the MOST RECENT fold (first in `folds`)
        # -------------------------------------------------------------
        (tr_s, tr_e, val_e, te_e) = folds[0]
        print_fold_header(clus, (tr_s, tr_e, val_e, te_e))

        train_df = slice_by_dates(dfc, tr_s, tr_e)
        val_df   = slice_by_dates(dfc, tr_e + pd.Timedelta(days=1), val_e)
        test_df  = slice_by_dates(dfc, val_e + pd.Timedelta(days=1), te_e)

        train_df = train_df.dropna(subset=[target_col])
        val_df   = val_df.dropna(subset=[target_col])
        test_df  = test_df.dropna(subset=[target_col])

        feats_cols = [c for c in all_features if c in train_df.columns]
        if len(feats_cols) == 0:
            print(f"[Cluster {clus}] No usable features. Skipping.")
            continue

        X_tr_full = train_df[feats_cols].astype(float)
        y_tr_full = train_df[target_col].astype(float).values
        X_val_full = val_df[feats_cols].astype(float)
        y_val_full = val_df[target_col].astype(float).values
        X_te_full = test_df[feats_cols].astype(float)
        y_te_full = test_df[target_col].astype(float).values

        # ------- Stage 1: initial fit (base params) + SHAP ranking -------
        base_params = dict(cfg["cat_base_params"])
        model_init = cat_fit_with_es(X_tr_full, y_tr_full, X_val_full, y_val_full, base_params)

        shap_rank = cat_shap_rank(
            model_init,
            X_val=X_val_full if len(X_val_full) else X_tr_full,
            y_val=y_val_full if len(X_val_full) else y_tr_full,
            X_bg=X_tr_full
        )
        sel_feats = enforce_diverse_lags(
            ranked=shap_rank,
            max_features=cfg["top_features_per_fold"],
            max_per_base=cfg["max_lags_per_base"],
            lag_regex=cfg["lag_suffix_regex"]
        )
        if len(sel_feats) == 0:
            print(f"[Cluster {clus}] SHAP selection returned 0 features; using all for safety.")
            sel_feats = feats_cols

        # Narrow matrices to the selected features (fixed during GA)
        X_tr  = train_df[sel_feats].astype(float)
        X_val = val_df[sel_feats].astype(float)
        X_te  = test_df[sel_feats].astype(float)
        y_tr  = y_tr_full
        y_val = y_val_full
        y_te  = y_te_full

        # ----------------- Stage 2: GA over CatBoost params -----------------
        print(f"[Cluster {clus}] GA search over params (features fixed: {len(sel_feats)})")
        rng = np.random.RandomState(cfg["ga"]["seed"])

        def init_individual():
            return ga_random_params(base_params)

        pop_size   = cfg["ga"]["pop_size"]
        n_gen      = cfg["ga"]["n_gen"]
        elite_k    = cfg["ga"]["elite_k"]
        tourn_size = cfg["ga"]["tourn_size"]
        mut_prob   = cfg["ga"]["mut_prob"]

        # Initialize population
        population = [init_individual() for _ in range(pop_size)]
        fitness = [None] * pop_size   # (MAE, WR)
        models_cache: List[Optional[CatBoostRegressor]] = [None] * pop_size

        # Evaluate gen 0
        for i, params in enumerate(population):
            mae, wr, model = ga_eval_params(params, X_tr, y_tr, X_val, y_val)
            fitness[i] = (mae, wr)
            models_cache[i] = model
            print(f"[Cluster {clus}] Gen 0 | Ind {i:03d} | MAE={mae:.6f} | WR={wr:.3f} "
                  f"| depth={params['depth']} lr={params['learning_rate']:.4f} "
                  f"l2={params['l2_leaf_reg']:.2f} sub={params['subsample']:.2f} "
                  f"rsm={params['rsm']:.2f} iters={params['iterations']}")

        # GA loop
        for g in range(1, n_gen + 1):
            # Rank by MAE (ascending)
            order = np.argsort([f[0] for f in fitness])
            population = [population[i] for i in order]
            fitness    = [fitness[i] for i in order]
            models_cache = [models_cache[i] for i in order]

            best_mae, best_wr = fitness[0]
            best_params = population[0]
            print(f"[Cluster {clus}] Gen {g-1} COMPLETE | Best MAE={best_mae:.6f} | WR={best_wr:.3f} "
                  f"| depth={best_params['depth']} lr={best_params['learning_rate']:.4f} "
                  f"l2={best_params['l2_leaf_reg']:.2f} sub={best_params['subsample']:.2f} "
                  f"rsm={best_params['rsm']:.2f} iters={best_params['iterations']}")

            # Elitism
            next_pop = population[:elite_k]
            next_fit = fitness[:elite_k]
            next_models = models_cache[:elite_k]

            # Reproduce
            while len(next_pop) < pop_size:
                i1 = tournament_select(population, fitness, tourn_size)
                i2 = tournament_select(population, fitness, tourn_size)
                p1, p2 = population[i1], population[i2]
                child = ga_crossover(p1, p2)
                child = ga_mutate(child, mut_prob)
                next_pop.append(child)
                next_fit.append(None)
                next_models.append(None)

            # Evaluate new individuals
            for i in range(elite_k, pop_size):
                params = next_pop[i]
                mae, wr, model = ga_eval_params(params, X_tr, y_tr, X_val, y_val)
                next_fit[i] = (mae, wr)
                next_models[i] = model
                print(f"[Cluster {clus}] Gen {g} | Ind {i:03d} | MAE={mae:.6f} | WR={wr:.3f} "
                      f"| depth={params['depth']} lr={params['learning_rate']:.4f} "
                      f"l2={params['l2_leaf_reg']:.2f} sub={params['subsample']:.2f} "
                      f"rsm={params['rsm']:.2f} iters={params['iterations']}")

            population, fitness, models_cache = next_pop, next_fit, next_models

        # Final best after GA
        order = np.argsort([f[0] for f in fitness])
        best_idx = order[0]
        best_params = population[best_idx]
        best_val_mae, best_val_wr = fitness[best_idx]
        print(f"[Cluster {clus}] GA DONE | Best on VAL → MAE={best_val_mae:.6f} | WR={best_val_wr:.3f}")
        print(f"[Cluster {clus}] Winner params: {json.dumps(best_params, indent=2)}")
        print(f"[Cluster {clus}] Winner features ({len(sel_feats)}): {sel_feats}")

        # ----------------- Stage 3: Retrain on train+val, test on last 60d -----------------
        X_trval = pd.concat([X_tr, X_val], axis=0)
        y_trval = np.concatenate([y_tr, y_val], axis=0)

        # Refit with ES on a small tail of trval for stability, else plain fit
        # Here we keep it simple: fit on trval without extra ES slice (already GA-optimized)
        final_model = cat_make(best_params)
        final_model.fit(X_trval, y_trval)

        pred_te = final_model.predict(X_te)
        m_te = metrics_safe(y_te, pred_te)
        bl1_te, bl5_te = compute_baselines(test_df, target_col)
        m_bl1_te = metrics_safe(y_te, bl1_te.values)
        m_bl5_te = metrics_safe(y_te, bl5_te.values)

        def fmt(m): return f"MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | WR={m['WR']:.3f} | n={m.get('_n',np.nan)}"
        print(f"[Cluster {clus}] TEST Model → {fmt(m_te)}")
        print(f"[Cluster {clus}] TEST BL-1d → {fmt(m_bl1_te)}")
        print(f"[Cluster {clus}] TEST BL-5d → {fmt(m_bl5_te)}")

        # -------- Save test predictions and config --------
        out_pred_te = test_df[ID_COLS + [target_col]].copy()
        out_pred_te["set"] = "TEST"
        out_pred_te["pred"] = pred_te
        out_pred_te["cluster"] = clus
        test_out_path = out_dir / f"preds_{clus}__lastfold_test.csv"
        out_pred_te.to_csv(test_out_path, index=False)
        print(f"[Cluster {clus}] Saved last-fold TEST preds → {test_out_path}")

        model_cfg = {
            "cluster": int(clus),
            "best_params": best_params,
            "selected_features_last_fold": sel_feats,
            "val_metrics_best": {"MAE": best_val_mae, "WR": best_val_wr},
            "test_metrics": m_te
        }
        (out_dir / f"model_{clus}__lastfold_config.json").write_text(json.dumps(model_cfg, indent=2))
        print(f"[Cluster {clus}] Saved model config → {out_dir / f'model_{clus}__lastfold_config.json'}")

        # -------- Save model: .cbm (native) + pickle copy with .pt suffix --------
        cbm_path = out_dir / f"model_{clus}__lastfold_best.cbm"
        final_model.save_model(str(cbm_path))
        print(f"[Cluster {clus}] Saved CatBoost model (.cbm) → {cbm_path}")

        # Optional pickle with .pt suffix (NOT PyTorch weights; just a pickle for portability)
        pt_pickle_path = out_dir / f"model_{clus}__lastfold_best.pt"
        try:
            joblib.dump(final_model, pt_pickle_path)
            print(f"[Cluster {clus}] Saved pickle copy (.pt) → {pt_pickle_path}  (NOTE: not a PyTorch state_dict)")
        except Exception as e:
            print(f"[Cluster {clus}] Could not save .pt pickle: {e}")

        # -------- Latest-day inference + Recent-days log (using GA best) --------
        latest_c = latest[latest["cluster"] == clus].copy()
        if latest_c.empty:
            print(f"[Cluster {clus}] No rows in latest_day for this cluster.")
        else:
            needed = [f for f in sel_feats if f in latest_c.columns]
            if needed:
                X_latest = latest_c[needed].astype(float)
                latest_pred = final_model.predict(X_latest)
                latest_out = latest_c[ID_COLS].copy()
                latest_out["pred"] = latest_pred
                latest_out["actual"] = np.nan
                latest_out["cluster"] = clus
                latest_path = out_dir / f"preds_{clus}__latest.csv"
                latest_out.to_csv(latest_path, index=False)
                print(f"[Cluster {clus}] Saved latest preds → {latest_path}")
                print(latest_out.head().to_string(index=False))

        ndays = CONFIG.get("recent_days_log", 0)
        if ndays and ndays > 0:
            last_in_date = dfc["date"].max()
            start_recent = last_in_date - pd.Timedelta(days=ndays-1)
            recent = dfc[(dfc["date"] >= start_recent) & (dfc["date"] <= last_in_date)].copy()
            sel_recent = [f for f in sel_feats if f in recent.columns]
            if len(sel_recent):
                X_recent = recent[sel_recent].astype(float)
                recent["pred"] = final_model.predict(X_recent)
                recent["actual"] = recent[target_col].astype(float)
                recent_log = recent[ID_COLS + ["pred", "actual"]].copy().sort_values(["date","symbol"])
                recent_path = out_dir / f"preds_{clus}__recent.csv"
                recent_log.to_csv(recent_path, index=False)
                print(f"[Cluster {clus}] Saved recent {ndays}d log (with actuals) → {recent_path}")
                print(recent_log.tail(10).to_string(index=False))

   

if __name__ == "__main__":
    main()
