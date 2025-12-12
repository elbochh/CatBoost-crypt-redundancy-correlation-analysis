from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH   = Path("merged_all.parquet")
OUT_TRAIN = Path("features_crypto_ohlcv.csv")
OUT_LATEST= Path("latest_day_features.csv")

# --------------------------- utilities ---------------------------

def safe_div(a, b):
    b = b.replace({0: np.nan})
    return a / b

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(price: pd.Series, window=14) -> pd.Series:
    d  = price.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    ag = up.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    al = dn.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = safe_div(ag, al)
    return 100 - (100/(1+rs))

def macd(price: pd.Series, fast=12, slow=26, signal=9):
    e_fast  = ema(price, fast)
    e_slow  = ema(price, slow)
    line    = e_fast - e_slow
    signal_ = ema(line, signal)
    hist    = line - signal_
    return line, signal_, hist

def dmi_adx(high, low, close, n=14):
    pc = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    up_move   = high.diff()
    down_move = -low.diff()
    pos_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    neg_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
    pos_dm_ema = pos_dm.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    neg_dm_ema = neg_dm.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    pos_di = 100 * safe_div(pos_dm_ema, atr)
    neg_di = 100 * safe_div(neg_dm_ema, atr)
    dx = 100 * safe_div((pos_di - neg_di).abs(), (pos_di + neg_di))
    adx = dx.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return pos_di, neg_di, adx, atr, tr

def roc(price: pd.Series, n: int) -> pd.Series:
    return safe_div(price, price.shift(n)) - 1.0

def roll_zscore(s: pd.Series, win=21) -> pd.Series:
    m  = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    return safe_div(s - m, sd)

def rolling_beta(y: pd.Series, x: pd.Series, win: int) -> pd.Series:
    """Beta_t = Cov(y,x)/Var(x) over a rolling window."""
    cov = (y.rolling(win, min_periods=win)
             .cov(x))
    var = (x.rolling(win, min_periods=win)
             .var())
    return safe_div(cov, var)

# ----------------------- lag/rolling helpers -----------------------

LAG_SET = [1,2,3,4,5,6,7,14,21,28]
RET_ROLL_WINS = [2,5,7,14,21,28]
ROLL_WINS = [5, 10, 21, 63]  # keep short/mid windows

def add_lags(df: pd.DataFrame, cols: list[str], lags=LAG_SET) -> pd.DataFrame:
    """
    Add lagged columns for each column in `cols` (if present).
    Names: <col>_lag<k>
    """
    for col in cols:
        if col in df.columns:
            for k in lags:
                df[f"{col}_lag{k}"] = df[col].shift(k)
    return df

# ----------------------- per-symbol feature builder -----------------------

def build_features_per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    # Ensure we have a market_cap column (fallback to usd_market_cap if needed)
    if "market_cap" not in g.columns and "usd_market_cap" in g.columns:
        g["market_cap"] = pd.to_numeric(g["usd_market_cap"], errors="coerce")

    # date & core types
    g["date"] = pd.to_datetime(g["date"], errors="coerce").dt.tz_localize(None)
    g = g.dropna(subset=["date"]).sort_values("date")
    for col in ["open", "high", "low", "close", "volume", "market_cap"]:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    # Often-provided USD columns
    for col in ["usd_volume", "usd_volume_24h", "usd_market_cap",
                "usd_total_supply", "usd_circulating_supply",
                "btc_dominance", "eth_dominance",
                "total_market_cap_usd", "altcoin_market_cap_usd",
                "total_volume_24h_usd", "altcoin_volume_24h_usd",
                "fear_greed_value"]:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    price = g["close"]
    c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"]

    feats = {}

    # Returns
    feats["ret_1d"]     = price.pct_change(1, fill_method=None)
    feats["ret_log_1d"] = np.log(price).diff()

    # Momentum (price-based)
    for w in ROLL_WINS:
        feats[f"mom_{w}"] = roc(price, w)

    # Volatility
    feats["vol_21"] = feats["ret_1d"].rolling(21, min_periods=21).std()
    feats["vol_63"] = feats["ret_1d"].rolling(63, min_periods=63).std()
    feats["vol_21_over_63"] = safe_div(feats["vol_21"], feats["vol_63"])

    # Realized-vol style (sqrt of sum of r^2)
    for w in [7, 14, 30]:
        feats[f"rv_{w}"] = np.sqrt((feats["ret_log_1d"]**2).rolling(w, min_periods=w).sum())
    feats["rv_change_7"] = feats["rv_7"] - feats["rv_7"].shift(1)

    # SMA / EMA (shorter only)
    for w in [10, 20, 50, 100]:
        feats[f"sma_{w}"] = price.rolling(w, min_periods=w).mean()
        feats[f"ema_{w}"] = ema(price, w)

    # RSI / Bollinger
    feats["rsi_7"]  = rsi(price, 7)
    feats["rsi_14"] = rsi(price, 14)
    ma20 = price.rolling(20, min_periods=20).mean()
    sd20 = price.rolling(20, min_periods=20).std()
    up20 = ma20 + 2 * sd20
    lo20 = ma20 - 2 * sd20
    feats["bb_ma20"]  = ma20
    feats["bb_up20"]  = up20
    feats["bb_lo20"]  = lo20
    feats["bb_pct20"] = safe_div((price - lo20), (up20 - lo20))
    feats["bb_bw20"]  = safe_div((up20 - lo20), ma20)

    # ATR & ranges (plus NATR/TR%)
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    feats["atr_14"]   = atr14
    feats["hl_range"] = safe_div(h - l, l)
    feats["co_return"]= safe_div(c - o, o)
    feats["gap_return"]= safe_div(o - pc, pc)
    feats["natr_14"]  = 100 * safe_div(atr14, c)
    feats["tr_pct"]   = safe_div(tr, c)

    # Z-scored log return
    feats["ret_log_1d_z21"] = roll_zscore(feats["ret_log_1d"], win=21)

    # Stochastic
    ll14 = l.rolling(14, min_periods=14).min()
    hh14 = h.rolling(14, min_periods=14).max()
    k    = 100 * safe_div(c - ll14, hh14 - ll14)
    feats["stoch_k14"] = k
    feats["stoch_d3"]  = k.rolling(3, min_periods=3).mean()

    # MACD / DMI-ADX
    macd_line, macd_sig, macd_hist = macd(price, 12, 26, 9)
    feats["macd_line"]   = macd_line
    feats["macd_signal"] = macd_sig
    feats["macd_hist"]   = macd_hist
    pos_di, neg_di, adx, _, _ = dmi_adx(h, l, c, n=14)
    feats["dmi_pos14"] = pos_di
    feats["dmi_neg14"] = neg_di
    feats["adx_14"]    = adx

    # Cumulative returns & price position/breakouts
    for w in [21, 63]:
        feats[f"cumret_{w}"] = roc(price, w)
    for w in [10, 20, 50]:
        hh = h.rolling(w, min_periods=w).max()
        ll = l.rolling(w, min_periods=w).min()
        feats[f"price_pos_{w}"]     = safe_div(c - ll, hh - ll)
        feats[f"breakout_up_{w}"]   = (c > hh).astype("int8")
        feats[f"breakout_down_{w}"] = (c < ll).astype("int8")

    # Volume-based additions
    lvol = np.log(v.clip(lower=1.0))
    feats["log_volume"]   = lvol
    feats["dlvol_1"]      = lvol.diff()  # log-volume change
    feats["vol_z21"]      = roll_zscore(lvol, win=21)

    # Keep ALL original columns in g (fear/greed, dominance, etc.)
    features_df = pd.DataFrame(feats)
    out = pd.concat([g.reset_index(drop=True),
                     features_df.reset_index(drop=True)], axis=1)

    # ---- Market cap features (only if present) ----
    if "market_cap" in out.columns:
        mc = pd.to_numeric(out["market_cap"], errors="coerce")
        log_mc = np.log(mc.clip(lower=1.0))
        dlog_mc = log_mc.diff()
        mc_feats = pd.DataFrame({
            "log_market_cap": log_mc,
            "dlog_market_cap_1": dlog_mc,
            "mc_volatility_63": dlog_mc.rolling(63, min_periods=63).std(),
            "dollar_volume": out["close"] * out["volume"],
        })
        out = pd.concat([out.reset_index(drop=True),
                         mc_feats.reset_index(drop=True)], axis=1)

    # -------------------- Lags & rolling means (existing) --------------------
    lag_targets = []
    if "ret_1d" in out.columns:       lag_targets.append("ret_1d")
    if "market_cap" in out.columns:   lag_targets.append("market_cap")
    if "volume" in out.columns:       lag_targets.append("volume")
    if "close" in out.columns:        lag_targets.append("close")
    out = add_lags(out, lag_targets, lags=LAG_SET)

    if "ret_1d" in out.columns:
        ret_lag1 = out["ret_1d"].shift(1)
        for w in RET_ROLL_WINS:
            out[f"ret_ma_{w}"] = ret_lag1.rolling(w, min_periods=w).mean()

    # -------------------- NEW: Market-wide & sentiment features --------------------
    # Use already-merged, date-aligned columns (added before groupby in main):
    # - mkt_ret_btc_lag1, mkt_ret_eth_lag1
    # - dlog_total_mcap_lag1, dlog_alt_mcap_lag1, dlog_total_vol_lag1, dlog_alt_vol_lag1
    # - dlog_btc_dom_lag1, dlog_eth_dom_lag1
    # - fear_greed_value_lag1, d_fear_greed_lag1
    # - dlog_circ_supply_lag1, dlog_total_supply_lag1
    # - turnover_lag1

    # Rolling betas to BTC/ETH based on lagged market returns (no lookahead)
    if "mkt_ret_btc_lag1" in out.columns and "ret_1d" in out.columns:
        out["beta_to_btc_60"] = rolling_beta(out["ret_1d"], out["mkt_ret_btc_lag1"], 60)
        out["resid_to_btc"]   = out["ret_1d"] - out["beta_to_btc_60"] * out["mkt_ret_btc_lag1"]

    if "mkt_ret_eth_lag1" in out.columns and "ret_1d" in out.columns:
        out["beta_to_eth_60"] = rolling_beta(out["ret_1d"], out["mkt_ret_eth_lag1"], 60)

    # Illiquidity (Amihud-style): |ret| / (USD volume)
    usd_vol_for_illiq = None
    if "usd_volume" in out.columns:
        usd_vol_for_illiq = out["usd_volume"]
    elif "usd_volume_24h" in out.columns:
        usd_vol_for_illiq = out["usd_volume_24h"]
    else:
        # fallback: dollar_volume computed above if market_cap was present; else approx with close*volume
        if "dollar_volume" in out.columns:
            usd_vol_for_illiq = out["dollar_volume"]
        else:
            usd_vol_for_illiq = out["close"] * out["volume"]

    out["illiquidity"] = safe_div(out["ret_1d"].abs(), usd_vol_for_illiq)

    # Turnover (USD volume / USD market cap) — use lag to avoid leakage
    if "usd_volume_24h" in out.columns and "usd_market_cap" in out.columns:
        out["turnover"]      = safe_div(out["usd_volume_24h"], out["usd_market_cap"])
        out["turnover_lag1"] = out["turnover"].shift(1)

    # Sentiment extremes from lagged Fear & Greed
    if "fear_greed_value_lag1" in out.columns:
        fg = out["fear_greed_value_lag1"]
        out["fg_extreme_fear"]  = (fg <= 20).astype("int8")
        out["fg_extreme_greed"] = (fg >= 80).astype("int8")

    # -------------------- NEW: Calendar encodings --------------------
    if "date" in out.columns:
        dow = out["date"].dt.dayofweek
        month = out["date"].dt.month
        out["dow_sin"]   = np.sin(2*np.pi*dow/7)
        out["dow_cos"]   = np.cos(2*np.pi*dow/7)
        out["month_sin"] = np.sin(2*np.pi*(month-1)/12)
        out["month_cos"] = np.cos(2*np.pi*(month-1)/12)

    return out

# ----------------------------- finalize -----------------------------

def finalize_features(feats: pd.DataFrame, warmup=63) -> pd.DataFrame:
    feats = feats.sort_values(["symbol","date"]).reset_index(drop=True)

    # Next-day target (no lookahead leak—uses shift(-1))
    feats["y_ret_next"] = feats.groupby("symbol", observed=True)["close"].pct_change(-1, fill_method=None)

    # Warmup per symbol
    feats["age"] = feats.groupby("symbol", observed=True).cumcount() + 1
    feats = feats[feats["age"] > warmup]

    # Targeted NaN drop (keep rows needed for training)
    essential = [
        "symbol","date","open","high","low","close","volume","y_ret_next",
        "ret_1d","rsi_14","macd_line","macd_signal","bb_pct20","atr_14",
        "mom_21","vol_21","stoch_k14","adx_14"
    ]
    essential = [c for c in essential if c in feats.columns]
    feats = feats.dropna(subset=essential).reset_index(drop=True)

    # Optional compact dtypes
    for col in feats.select_dtypes(include=["float64"]).columns:
        feats[col] = feats[col].astype("float32")
    for col in feats.select_dtypes(include=["int64"]).columns:
        feats[col] = pd.to_numeric(feats[col], downcast="integer")

    return feats.copy()

def latest_day_snapshot(feats_raw: pd.DataFrame, warmup=63) -> pd.DataFrame:
    """
    Build an inference matrix for the GLOBAL latest date in feats_raw.
    Applies the SAME warmup and essential-feature NaN rules as training,
    but does NOT require y_ret_next (target).
    """
    fr = feats_raw.sort_values(["symbol","date"]).reset_index(drop=True)

    # Warmup per symbol
    fr["age"] = fr.groupby("symbol", observed=True).cumcount() + 1
    fr = fr[fr["age"] > warmup]

    # Use the same essential set MINUS y_ret_next
    essential = [
        "symbol","date","open","high","low","close","volume",
        "ret_1d","rsi_14","macd_line","macd_signal","bb_pct20","atr_14",
        "mom_21","vol_21","stoch_k14","adx_14"
    ]
    essential = [c for c in essential if c in fr.columns]
    fr = fr.dropna(subset=essential).reset_index(drop=True)

    # Keep only rows on the global latest date
    if fr.empty:
        return fr
    last_date = fr["date"].max()
    fr = fr[fr["date"] == last_date].copy()

    # Drop helper column
    fr = fr.drop(columns=["age"], errors="ignore")
    return fr.reset_index(drop=True)

# ---------------------- Global context builder ----------------------

def make_date_level_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build date-level market context from duplicated per-row fields.
    We *first* collapse to one row per date using first non-null,
    then compute deltas/log-deltas and lag them by 1 to avoid leakage.
    """
    # Columns we expect may exist (some may be missing—handled safely)
    cols = [
        "btc_dominance","eth_dominance",
        "total_market_cap_usd","altcoin_market_cap_usd",
        "total_volume_24h_usd","altcoin_volume_24h_usd",
        "fear_greed_value",
        "usd_total_supply","usd_circulating_supply"
    ]
    use_cols = ["date"] + [c for c in cols if c in df.columns]
    if len(use_cols) == 1:
        # Nothing to compute, return empty frame with date
        return df[["date"]].drop_duplicates().sort_values("date")

    # Collapse to date-level by first non-null
    agg = {c: "first" for c in use_cols if c != "date"}
    ctx = (df[use_cols]
           .sort_values("date")
           .groupby("date", as_index=False).agg(agg))

    # To numeric
    for c in ctx.columns:
        if c != "date":
            ctx[c] = pd.to_numeric(ctx[c], errors="coerce")

    # Log-deltas for market breadth / dominance / supplies
    def dlog(s):
        return np.log(s.clip(lower=1e-12)).diff()

    add = {}
    if "total_market_cap_usd" in ctx.columns:
        add["dlog_total_mcap"] = dlog(ctx["total_market_cap_usd"])
    if "altcoin_market_cap_usd" in ctx.columns:
        add["dlog_alt_mcap"] = dlog(ctx["altcoin_market_cap_usd"])
    if "total_volume_24h_usd" in ctx.columns:
        add["dlog_total_vol"] = dlog(ctx["total_volume_24h_usd"])
    if "altcoin_volume_24h_usd" in ctx.columns:
        add["dlog_alt_vol"] = dlog(ctx["altcoin_volume_24h_usd"])
    if "btc_dominance" in ctx.columns:
        add["dlog_btc_dom"] = dlog(ctx["btc_dominance"])
    if "eth_dominance" in ctx.columns:
        add["dlog_eth_dom"] = dlog(ctx["eth_dominance"])
    if "usd_circulating_supply" in ctx.columns:
        add["dlog_circ_supply"] = dlog(ctx["usd_circulating_supply"])
    if "usd_total_supply" in ctx.columns:
        add["dlog_total_supply"] = dlog(ctx["usd_total_supply"])

    for k, v in add.items():
        ctx[k] = v

    # Fear & Greed change
    if "fear_greed_value" in ctx.columns:
        ctx["d_fear_greed"] = ctx["fear_greed_value"].diff()

    # Leak-safe: lag everything by 1 day
    lag_cols = [c for c in ctx.columns if c != "date"]
    for c in lag_cols:
        ctx[c + "_lag1"] = ctx[c].shift(1)

    # Keep only lagged versions to join (prevents accidental use of t info)
    keep_cols = ["date"] + [c + "_lag1" for c in lag_cols]
    ctx = ctx[keep_cols]

    return ctx

def add_market_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add BTC/ETH daily returns at date-level (lagged).
    Assumes 'symbol' identifies BTC/ETH rows. We try 'BTC'/'ETH' directly.
    """
    mkt = df[["date","symbol","close"]].copy()
    mkt["close"] = pd.to_numeric(mkt["close"], errors="coerce")
    mkt = mkt.dropna(subset=["date","symbol","close"])

    # Map BTC and ETH returns by date
    ret_map = {}
    for coin, col_name in [("BTC","mkt_ret_btc"), ("ETH","mkt_ret_eth")]:
        tmp = (mkt[mkt["symbol"].str.upper() == coin]
               .sort_values("date"))
        if not tmp.empty:
            r = tmp["close"].pct_change(1, fill_method=None)
            ret_map[col_name] = pd.Series(r.values, index=tmp["date"].values)

    # Build a small date-level frame
    dates = df["date"].dropna().drop_duplicates().sort_values()
    ctx = pd.DataFrame({"date": dates})
    for col_name, series in ret_map.items():
        s = pd.Series(series.values, index=series.index)
        # align by date
        s = s.reindex(dates).astype(float)
        ctx[col_name] = s.values

    # Lag by 1 bar for leak safety
    for c in ["mkt_ret_btc", "mkt_ret_eth"]:
        if c in ctx.columns:
            ctx[c + "_lag1"] = ctx[c].shift(1)

    # Keep only lagged
    keep = ["date"] + [c for c in ctx.columns if c.endswith("_lag1")]
    return ctx[keep]

# -------------------------------- main --------------------------------

def main():
    # Read CSV
    df = pd.read_parquet(IN_PATH)

    # Validate required
    req = {"symbol","date","open","high","low","close","volume"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Input missing required columns: {miss}")

    # Normalize date, sort, dedupe
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values(["symbol","date"])
    df = df.drop_duplicates(["symbol","date"], keep="last")

    # -------------------- NEW: build and merge date-level context --------------------
    ctx1 = make_date_level_context(df)
    ctx2 = add_market_returns(df)
    # Merge context into the row-level df (on date)
    df = df.merge(ctx1, on="date", how="left")
    df = df.merge(ctx2, on="date", how="left")

    # Also add *lagged* percent-change fields as features to avoid leakage
    for pc_col in ["usd_percent_change_1h","usd_percent_change_24h",
                   "usd_percent_change_7d","usd_percent_change_30d"]:
        if pc_col in df.columns:
            df[pc_col] = pd.to_numeric(df[pc_col], errors="coerce")
            df[pc_col + "_lag1"] = df[pc_col].shift(1)

    # Turnover (USD vol / USD mcap) at row level (raw + lag1 added in per-symbol)
    if "usd_volume_24h" in df.columns and "usd_market_cap" in df.columns:
        df["turnover_row"] = safe_div(pd.to_numeric(df["usd_volume_24h"], errors="coerce"),
                                      pd.to_numeric(df["usd_market_cap"], errors="coerce"))
        df["turnover_row_lag1"] = df["turnover_row"].shift(1)

    # Build per symbol (raw features with all columns)
    feats_raw = (
        df.groupby("symbol", observed=True, group_keys=False)
          .apply(build_features_per_symbol)
          .reset_index(drop=True)
    )

    # TRAINING SET (with target and strict filtering)
    feats_train = finalize_features(feats_raw.copy(), warmup=63)
    feats_train = feats_train.dropna().reset_index(drop=True)
    # keep your previous drops
    feats_train = feats_train.drop(columns=["age", "name", "fear_greed_label"], errors="ignore")
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    feats_train.to_csv(OUT_TRAIN, index=False)
    print(f"[TRAIN] → {OUT_TRAIN} | rows={len(feats_train):,} cols={feats_train.shape[1]}")

    # LATEST-DAY INFERENCE SNAPSHOT (no target required)
    latest = latest_day_snapshot(feats_raw.copy(), warmup=63)
    latest = latest.drop(columns=["name", "fear_greed_label"], errors="ignore")
    latest.to_csv(OUT_LATEST, index=False)
    if not latest.empty:
        print(f"[INFER]  → {OUT_LATEST} | rows={len(latest):,} date={latest['date'].iloc[0].date()}")
    else:
        print("[INFER]  → latest snapshot empty (not enough warmup/NaNs). Consider lowering warmup to 21–35 if needed.")

if __name__ == "__main__":
    main()
