#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kmeans_crypto.py

Cluster CRYPTO symbols by realized volatility (rolling), mirroring the STOCKS KMeans pipeline:
- Per-symbol ret_1d (pct_change on close if missing)
- Realized volatility = rolling std over a lookback window, annualized
- Aggregate a single vol per symbol (median/mean)
- Robust-scale vol, KMeans on 1D feature
- Output: ohlcv_clusters.csv with columns [symbol, vol, cluster_name, cluster]

Notes:
- Annualization uses sqrt(365) by default (continuous crypto markets). Set to 252 if desired.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

INPUT_FILE  = "merged_all.parquet"    
OUTPUT_FILE = "ohlcv_clusters.csv"
N_CLUSTERS  = 2

# volatility controls
RET_COL          = "ret_1d"       
VOL_LOOKBACK     = 63             
ANNUALIZE_DAYS   = 365.0 #252.0 
AGGREGATION      = "median"#"mean"
MIN_OBS_PER_SYM  = 200            

SYMBOL_COL_CANDS = ["symbol", "ticker", "asset", "coin"]
CLOSE_COL_CANDS  = ["close", "adj_close", "price_close"]

# ===================== Helpers =====================
def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)

def _detect_col(df: pd.DataFrame, cands: list[str], fuzzy_contains: list[str] | None = None) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    if fuzzy_contains:
        for c in df.columns:
            cl = c.lower()
            if all(k in cl for k in fuzzy_contains) and "time" not in cl:
                return c
    return None

def realized_vol_series(ret: pd.Series, win=63, annualize_days=365.0) -> pd.Series:
    vol = ret.rolling(win, min_periods=win).std()
    if annualize_days and annualize_days > 0:
        vol = vol * np.sqrt(annualize_days)
    return vol

def per_symbol_vol(df: pd.DataFrame, symbol_col: str, close_col: str) -> pd.DataFrame:
    d = df[[symbol_col, "date", close_col] + ([RET_COL] if RET_COL in df.columns else [])].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values([symbol_col, "date"], kind="mergesort")

    # returns
    if RET_COL in d.columns:
        d["ret_1d"] = pd.to_numeric(d[RET_COL], errors="coerce")
    else:
        d["ret_1d"] = d.groupby(symbol_col, sort=False)[close_col].pct_change()

    # rolling realized vol, per row
    d["rv"] = d.groupby(symbol_col, sort=False)["ret_1d"].transform(
        lambda s: realized_vol_series(s, win=VOL_LOOKBACK, annualize_days=ANNUALIZE_DAYS)
    )

    # aggregate to one vol per symbol
    agg_fn = {"median": np.nanmedian, "mean": np.nanmean}[AGGREGATION]
    vt = (
        d.groupby(symbol_col, as_index=False)
         .agg(n_obs=("ret_1d","count"), vol=("rv", agg_fn))
    )
    vt = vt[(vt["n_obs"] >= MIN_OBS_PER_SYM) & vt["vol"].notna()].reset_index(drop=True)
    return vt

# ===================== Main =====================
def main():
    df = _read_any(INPUT_FILE)
    print(f"Loaded {len(df):,} rows | columns: {len(df.columns)}")

    # detect columns
    symbol_col = _detect_col(df, SYMBOL_COL_CANDS)
    if symbol_col is None:
        raise ValueError("❌ Expected a symbol-like column (e.g., 'symbol'/'ticker').")

    close_col = _detect_col(df, CLOSE_COL_CANDS, fuzzy_contains=["close"])
    if close_col is None:
        raise ValueError("❌ Could not find a price column (close/adj_close).")

    # ensure numeric price/ret
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    if RET_COL in df.columns:
        df[RET_COL] = pd.to_numeric(df[RET_COL], errors="coerce")

    # drop unusable
    df = df.dropna(subset=[symbol_col, close_col])
    if df.empty:
        raise ValueError("No valid rows after basic cleaning.")

    print(f"Using symbol col: {symbol_col}, close col: {close_col}")

    # compute per-symbol realized volatility
    vt = per_symbol_vol(df, symbol_col=symbol_col, close_col=close_col)
    if vt.empty:
        raise ValueError( No valid volatility computed. Check data / MIN_OBS_PER_SYM / VOL_LOOKBACK.")

    # cluster on volatility only
    X = vt[["vol"]].values
    Xs = RobustScaler().fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    vt["cluster"] = labels

    # map clusters to names by cluster mean (low_vol / high_vol)
    means = vt.groupby("cluster")["vol"].mean().sort_values()
    order = {cluster_id: rank for rank, cluster_id in enumerate(means.index)}
    vt["cluster_rank"] = vt["cluster"].map(order)
    vt["cluster_name"] = np.where(vt["cluster_rank"] == 0, "low_vol", "high_vol")

    out = vt[[symbol_col, "vol", "cluster_name", "cluster"]].rename(columns={symbol_col: "symbol"})
    out = out.sort_values("vol")
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(out):,} symbols with labels → {OUTPUT_FILE}")
    print(out.groupby("cluster_name")["vol"].agg(["count","mean","min","max"]))

if __name__ == "__main__":
    main()
