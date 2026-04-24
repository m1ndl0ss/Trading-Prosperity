#!/usr/bin/env python3
"""
Analyze hand-picked chart "buy" / "sell" points against the prices tape.

You cannot paste a screenshot for reliable pixel→(day,timestamp) extraction here.
Instead, export marks from your dashboard or a small CSV:

  day,timestamp,side
  0,125000,BUY
  0,340000,SELL
  1,90000,BUY
  ...

Timestamps must match the `timestamp` column in prices_*.csv for the round.

This script:
  1) Joins each label to the product row at (day, timestamp)
  2) Builds per-tick features (mid, spread, EMA distance, RSI, position in session range, etc.)
  3) Summarizes what BUY rows have in common vs SELL rows
  4) Counts how many *other* ticks fall in a "buy box" / "sell box" (feature quantiles from
     your labels) on the same days or all days
  5) Gives a crude *proxy* PnL for "trade every extra buy-box tick" (NOT a full replay):
     assumes you buy `clip` at best ask and exit at mid `horizon` ms later using same
     mid series — use for ordering of ideas only; real PnL needs `backtest_round_pnl.py`
     with a Trader that encodes your rule.

Usage (from Trading-Prosperity repo root):

  python scripts/chart_label_analysis.py --round ROUND3 --product HYDROGEL_PACK \\
      --labels path/to/marks.csv

Optional: --horizon 5000 --clip 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _rsi_from_mids(mids: np.ndarray, period: int) -> np.ndarray:
    """RSI at each index; NaN until enough history within the series."""
    n = len(mids)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    for i in range(period, n):
        w = mids[i - period : i + 1]
        diffs = np.diff(w)
        gains = np.maximum(0.0, diffs)
        losses = np.maximum(0.0, -diffs)
        ag = gains.mean()
        al = losses.mean()
        if al < 1e-12:
            out[i] = 50.0 if ag < 1e-12 else 100.0
        else:
            rs = ag / al
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def _ema(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def load_product_tape(data_dir: Path, round_name: str, product: str) -> pd.DataFrame:
    paths = sorted(data_dir.glob(f"prices_*{round_name.lower()}*.csv"))
    if not paths:
        paths = sorted((data_dir / round_name).glob("prices_*.csv"))
    if not paths:
        raise SystemExit(f"No prices CSVs under {data_dir / round_name}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, sep=";")
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    h = allp[allp["product"] == product].copy()
    if h.empty:
        raise SystemExit(f"No rows for product {product}")
    h = h.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return h


def enrich_features(df: pd.DataFrame, ema_halflife_ticks: float = 32.0, rsi_period: int = 24) -> pd.DataFrame:
    """Add mid, spread, EMA, RSI, session range percentile, 1-tick return."""
    out = df.copy()
    mid = out["mid_price"].astype(float)
    out["mid"] = mid
    # spread proxy: from L1 columns if present
    if "bid_price_1" in out.columns and "ask_price_1" in out.columns:
        out["spread"] = (out["ask_price_1"] - out["bid_price_1"]).astype(float)
    else:
        out["spread"] = np.nan

    alpha = 1.0 - np.exp(-np.log(2.0) / max(1.0, ema_halflife_ticks))
    out["ema"] = _ema(mid, alpha)
    out["dist_ema"] = mid - out["ema"]
    out["ret1"] = mid.diff()

    lo = out.groupby("day")["mid"].cummin()
    hi = out.groupby("day")["mid"].cummax()
    out["sess_lo"] = lo
    out["sess_hi"] = hi
    rng = (hi - lo).replace(0, np.nan)
    out["dist_range"] = (mid - lo) / (rng + 1e-9)

    mids = mid.to_numpy()
    out["rsi"] = _rsi_from_mids(mids, rsi_period)
    return out


def merge_labels(
    tape: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Exact join on day+timestamp; drops labels with no row."""
    lab = labels.copy()
    lab["day"] = lab["day"].astype(int)
    lab["timestamp"] = lab["timestamp"].astype(int)
    lab["side"] = lab["side"].str.upper()
    m = lab.merge(
        tape,
        on=["day", "timestamp"],
        how="left",
        suffixes=("_label", ""),
    )
    miss = m["mid"].isna().sum()
    if miss:
        print(f"Warning: {miss} labels had no exact (day,timestamp) row; drop them.")
        m = m.dropna(subset=["mid"])
    return m


def quantile_box(
    tape: pd.DataFrame, features: List[str], side: str, lo_q: float = 0.15, hi_q: float = 0.85
) -> Tuple[pd.Series, pd.Series]:
    sub = tape[tape["side"] == side]
    low = sub[features].quantile(lo_q)
    high = sub[features].quantile(hi_q)
    return low, high


def count_in_box(
    tape: pd.DataFrame, low: pd.Series, high: pd.Series, features: List[str]
) -> pd.Series:
    m = np.ones(len(tape), dtype=bool)
    for f in features:
        m &= tape[f].between(low[f], high[f], inclusive="both")
    return m


def naive_long_pnl_proxy(
    tape: pd.DataFrame, buy_mask: pd.Series, horizon_ms: int, clip: int
) -> float:
    """
    For each index i where buy_mask, pretend we buy `clip` at mid[i] (proxy; real is ask),
    exit at mid of row with timestamp >= timestamp[i]+horizon or last row of day.
    """
    pnl = 0.0
    arr = tape.reset_index(drop=True)
    mask = buy_mask.reindex(arr.index, fill_value=False).to_numpy()
    ts = arr["timestamp"].to_numpy()
    days = arr["day"].to_numpy()
    mid = arr["mid"].to_numpy()
    n = len(arr)
    for i in np.flatnonzero(mask):
        target = ts[i] + horizon_ms
        j = i + 1
        while j < n and days[j] == days[i] and ts[j] < target:
            j += 1
        j = min(j, n - 1)
        if days[j] != days[i]:
            j = i  # EOD same row
        # long: + (mid_exit - mid_entry) * clip
        pnl += float((mid[j] - mid[i]) * clip)
    return pnl


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze chart BUY/SELL labels vs tape")
    ap.add_argument("--round", default="ROUND3", help="Data subfolder under Data/")
    ap.add_argument("--data", type=Path, default=ROOT / "Data", help="Data root")
    ap.add_argument("--product", default="HYDROGEL_PACK")
    ap.add_argument("--labels", type=Path, required=True, help="CSV: day,timestamp,side")
    ap.add_argument("--horizon", type=int, default=5000, help="ms for naive long PnL proxy")
    ap.add_argument("--clip", type=int, default=10, help="units for naive PnL proxy")
    args = ap.parse_args()

    tape0 = load_product_tape(args.data, args.round, args.product)
    tape = enrich_features(tape0)

    lab = pd.read_csv(args.labels)
    for c in ("day", "timestamp", "side"):
        if c not in lab.columns:
            raise SystemExit(f"labels CSV must have columns: day, timestamp, side — got {list(lab.columns)}")

    joined = merge_labels(tape, lab)
    if joined.empty:
        raise SystemExit("No labels joined to tape.")

    features = ["mid", "spread", "dist_ema", "dist_range", "rsi", "ret1"]
    use = [f for f in features if f in joined.columns]
    print("=== Labeled rows (joined) ===")
    print(joined[["day", "timestamp", "side"] + use].to_string())
    print()

    buys = joined[joined["side"] == "BUY"]
    sells = joined[joined["side"] == "SELL"]
    print("=== BUY vs SELL — feature means (labeled points only) ===")
    if not buys.empty:
        print("BUY count", len(buys))
        print(buys[use].mean().to_string())
    else:
        print("BUY: (none)")
    print()
    if not sells.empty:
        print("SELL count", len(sells))
        print(sells[use].mean().to_string())
    else:
        print("SELL: (none)")
    print()

    print("=== BUY vs SELL — feature std ===")
    if not buys.empty:
        print(buys[use].std().to_string())
    print("---")
    if not sells.empty:
        print(sells[use].std().to_string())
    print()

    # Quantile boxes from labels
    okf = [f for f in use if joined[f].notna().any()]
    if len(buys) >= 3 and len(okf) >= 2:
        low_b, high_b = quantile_box(joined, okf, "BUY")
        print("=== Heuristic 'BUY box' (15th–85th pct of your BUY labels) ===")
        print(pd.DataFrame({"low": low_b, "high": high_b}))
        box_b = count_in_box(tape, low_b, high_b, okf)
        n_box = int(box_b.sum())
        print(f"Ticks in BUY box (all days, product tape): {n_box} / {len(tape)}")
        lab_buy_set = set(zip(buys["day"].astype(int), buys["timestamp"].astype(int)))
        extra_mask = []
        for _, r in tape.iterrows():
            key = (int(r["day"]), int(r["timestamp"]))
            in_box = bool(
                all(
                    low_b[f] <= r[f] <= high_b[f]
                    for f in okf
                    if pd.notna(r[f]) and f in low_b
                )
            )
            extra_mask.append(in_box and key not in lab_buy_set)
        extra_mask = np.array(extra_mask, dtype=bool)
        print(f"Ticks in BUY box excluding your labeled BUY timestamps: {extra_mask.sum()}")
        proxy = naive_long_pnl_proxy(tape, pd.Series(extra_mask), args.horizon, args.clip)
        print(
            f"Naive long PnL proxy on those extra-only ticks (clip={args.clip}, horizon_ms={args.horizon}): {proxy:,.3f}"
        )
        print("(Proxy uses mid-to-mid; not real fill sim. Encode rules in a Trader + backtest_round_pnl.py for real PnL.)")
    else:
        print("Not enough BUY labels (need >=3) for a multivariate box — add more marks or lower bar manually.")

    if len(sells) >= 3 and len(okf) >= 2:
        low_s, high_s = quantile_box(joined, okf, "SELL")
        print()
        print("=== Heuristic 'SELL box' (15th–85th pct of your SELL labels) ===")
        print(pd.DataFrame({"low": low_s, "high": high_s}))
        box_s = count_in_box(tape, low_s, high_s, okf)
        print(f"Ticks in SELL box: {int(box_s.sum())} / {len(tape)}")


if __name__ == "__main__":
    main()
