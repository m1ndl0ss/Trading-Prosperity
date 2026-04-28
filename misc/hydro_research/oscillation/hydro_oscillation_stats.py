#!/usr/bin/env python3
"""
Part A: HYDROGEL_PACK oscillation stats (no strategy fitting on R4 day 3).
Days: R3 day_0, R4 day_1, day_2 only.

Outputs CSV with per-day metrics + R3 day0 percentile bands for Donchian-style ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# .../misc/hydro_research/oscillation/thisfile -> misc
_MISC = Path(__file__).resolve().parent.parent.parent
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))

from replay_engine import best_bid_ask, read_prices_csv, row_to_depth

HYDRO = "HYDROGEL_PACK"
BUY_T = 9935
SELL_T = 10030

_HYDRO_RESEARCH = Path(__file__).resolve().parent.parent
_DATA = _MISC.parent / "Data"
_OUT = _HYDRO_RESEARCH / "output" / "oscillation"

DAY_SPECS: Tuple[Tuple[str, Path, Path], ...] = (
    ("R3_day0", _DATA / "ROUND3" / "prices_round_3_day_0.csv", None),
    ("R4_day1", _DATA / "ROUND4" / "prices_round_4_day_1.csv", None),
    ("R4_day2", _DATA / "ROUND4" / "prices_round_4_day_2.csv", None),
)


def _spread_mid(row: pd.Series) -> Tuple[float, float]:
    d = row_to_depth(row)
    try:
        bb, ba = best_bid_ask(d)
    except ValueError:
        return float("nan"), float("nan")
    sp = float(ba - bb)
    mid = float(row["mid_price"]) if pd.notna(row.get("mid_price")) else (bb + ba) / 2.0
    return sp, mid


def _acf_lags(x: pd.Series, max_lag: int) -> Dict[str, float]:
    r = x.dropna()
    if len(r) < max_lag + 5:
        return {f"acf_lag_{k}": float("nan") for k in range(1, max_lag + 1)}
    out: Dict[str, float] = {}
    for k in range(1, max_lag + 1):
        out[f"acf_ret_lag_{k}"] = float(r.autocorr(lag=k)) if len(r) > k else float("nan")
    return out


def analyze_day(prices_path: Path, day_id: str) -> Dict[str, Any]:
    p = read_prices_csv(prices_path)
    h = p[p["product"] == HYDRO].copy()
    if h.empty:
        return {"day_id": day_id, "n_ticks": 0}

    spreads: List[float] = []
    mids: List[float] = []
    for _, row in h.iterrows():
        sp, mid = _spread_mid(row)
        spreads.append(sp)
        mids.append(mid)

    mid_s = pd.Series(mids, dtype=float)
    ret = mid_s.diff()

    inside_band = ((mid_s >= BUY_T) & (mid_s <= SELL_T)).mean()
    below_buy = (mid_s < BUY_T).mean()
    above_sell = (mid_s > SELL_T).mean()
    mm_neutral = ((mid_s > BUY_T) & (mid_s < SELL_T)).mean()

    row: Dict[str, Any] = {
        "day_id": day_id,
        "n_ticks": int(len(mid_s)),
        "mid_mean": float(mid_s.mean()),
        "mid_std": float(mid_s.std(ddof=0)),
        "mid_p10": float(mid_s.quantile(0.10)),
        "mid_p50": float(mid_s.quantile(0.50)),
        "mid_p90": float(mid_s.quantile(0.90)),
        "spread_mean": float(np.nanmean(spreads)),
        "spread_median": float(np.nanmedian(spreads)),
        "ret_std_1tick": float(ret.std(ddof=0)),
        "frac_inside_9935_10030": float(inside_band),
        "frac_below_buy9935": float(below_buy),
        "frac_above_sell10030": float(above_sell),
        "frac_mm_neutral_strict": float(mm_neutral),
    }
    row.update(_acf_lags(ret, 10))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="HYDRO oscillation stats (Part A)")
    ap.add_argument(
        "--out",
        type=Path,
        default=_OUT / "hydro_oscillation_stats.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--donchian-json",
        type=Path,
        default=_OUT / "hydro_donchian_r3d0.json",
        help="Write R3 day0 p10/p90 sweep band for Donchian trader",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for day_id, path, _ in DAY_SPECS:
        if not path.is_file():
            raise SystemExit(f"Missing prices file: {path}")
        rows.append(analyze_day(path, day_id))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    r3 = df[df["day_id"] == "R3_day0"].iloc[0]
    lo = int(round(r3["mid_p10"]))
    hi = int(round(r3["mid_p90"]))
    lo, hi = min(lo, hi), max(lo, hi)
    dj = {
        "source": "R3_day0_mid_percentiles",
        "buy_thresh": lo,
        "sell_thresh": hi,
        "note": "Sweep when mid < p10 or > p90 of R3 day0 mid; used only for Donchian ablation on R4.",
    }
    args.donchian_json.parent.mkdir(parents=True, exist_ok=True)
    args.donchian_json.write_text(json.dumps(dj, indent=2), encoding="utf-8")
    print(f"Wrote {args.donchian_json}  -> buy_thresh={lo} sell_thresh={hi}")


if __name__ == "__main__":
    main()
