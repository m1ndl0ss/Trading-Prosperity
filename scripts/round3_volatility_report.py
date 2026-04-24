#!/usr/bin/env python3
"""
Standard realized-vol summary for IMC Data/ROUND3 (all days in folder).

For each (day, product): consecutive mid_price log-returns, drop zeros / invalid.
Reports:
  - n_ticks, mean mid
  - std of log returns (per step)
  - sqrt(sum of r^2))  (total realized move scale for the day, comparable across days)
  - range_pct = (max - min) / mean(mid) * 100
  - max|single-step| log return in bps
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "ROUND3"


def main() -> None:
    files = sorted(DATA.glob("prices_round_3_day_*.csv"))
    if not files:
        print(f"No prices CSVs in {DATA}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for path in files:
        day = int(path.stem.split("_day_")[-1])
        df = pd.read_csv(path, sep=";")
        for prod, g in df.groupby("product", sort=True):
            g = g.sort_values("timestamp")
            m = g["mid_price"].astype(float).values
            m = m[np.isfinite(m) & (m > 0)]
            if m.size < 3:
                continue
            r = np.diff(np.log(m))
            r = r[np.isfinite(r)]
            if r.size == 0:
                continue
            std_r = float(np.std(r, ddof=1))
            rv2 = float(np.sqrt(np.sum(r**2)))
            mean_m = float(np.mean(m))
            range_pct = float((np.max(m) - np.min(m)) / mean_m * 100.0) if mean_m else 0.0
            max_abs = float(np.max(np.abs(r)) * 10000)  # bps
            total_drift_bps = float(
                (np.log(m[-1] / m[0]) * 10000) if m[0] > 0 else 0.0
            )  # open-to-close of tape mid
            rows.append(
                {
                    "day": day,
                    "product": prod,
                    "n_rets": len(r),
                    "mean_mid": mean_m,
                    "std_log_ret": std_r * 1e4,  # bps (per 1-tick step, ~not annualized)
                    "L2_rv": rv2 * 1e4,  # bps, sum in sqrt space: comparable scale
                    "range_mid_pct": range_pct,
                    "max_1step_bps": max_abs,
                    "day_drift_bps": total_drift_bps,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No data.")
        return

    # Wide comparison: HYDROGEL and VELVET (delta-1) + summary by day
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=" * 80)
    print("ROUND3 volatility - by day x product (mid_price, consecutive-row log returns)")
    print(f"Data: {DATA}")
    print("=" * 80)

    key = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
    for p in key:
        sub = out[out["product"] == p].sort_values("day")
        if sub.empty:
            continue
        print(f"\n--- {p} ---")
        print(
            sub[
                [
                    "day",
                    "n_rets",
                    "mean_mid",
                    "std_log_ret",
                    "L2_rv",
                    "range_mid_pct",
                    "day_drift_bps",
                    "max_1step_bps",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:,.4f}" if abs(x) < 1e5 else f"{x:,.2f}")
        )

    # Day aggregates: median of std across products, and hydrogel rank
    print("\n" + "=" * 80)
    print("Day-level cross-section (all products) - median std_log_ret and median range_pct")
    print("=" * 80)
    by_day = out.groupby("day", sort=True)
    gmed = by_day.agg(
        n_syms=("product", "count"),
        med_std_bps=("std_log_ret", "median"),
        p90_std_bps=("std_log_ret", lambda s: float(np.percentile(s, 90))),
        med_range_pct=("range_mid_pct", "median"),
        med_max_step=("max_1step_bps", "median"),
    ).reset_index()
    print(gmed.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # Table: HYDROGEL vs day — simple bar for user
    h = out[out["product"] == "HYDROGEL_PACK"]
    if not h.empty:
        print("\n" + "=" * 80)
        print("HYDROGEL_PACK: volatility metrics by day (higher = more jagged/variable)")
        print("=" * 80)
        for _, r in h.sort_values("day").iterrows():
            d = int(r["day"])
            print(
                f"  day {d}:  std(d ln mid) = {r['std_log_ret']:,.2f} bps/tick  |  "
                f"L2 = {r['L2_rv']:,.0f}  |  range = {r['range_mid_pct']:.3f}%  |  "
                f"first-to-last mid drift = {r['day_drift_bps']:+,.1f} bps  |  max 1-step = {r['max_1step_bps']:.1f} bps"
            )

    print("\n" + "=" * 80)
    print("Full cross-product: std(d log mid) in bps per tick, columns = day")
    print("=" * 80)
    piv = out.pivot_table(
        index="product", columns="day", values="std_log_ret", aggfunc="first"
    )
    piv = piv.sort_index()
    print(piv.to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
