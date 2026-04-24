#!/usr/bin/env python3
"""
Path-risk / structural metrics for IMC Data/ROUND3 (complements `round3_volatility_report.py`).

Per (day, product) on `mid_price` in timestamp order:
  * max_drawdown_bps: max over t of (running_peak - m_t) / running_peak  (XIR-like bps)
  * time_below_slow_ema: fraction of ticks with mid < EMA(m; alpha) after a short burn-in
  * max_down_leg_bps: largest single contiguous run of negative 1-tick log-returns, as bps
  * day_drift_bps: ln(last/first) * 1e4  (end-of-day, can hide intra-day path risk)

For nearly constant mids (e.g. stuck 0.5), some fields are set to None / skipped.

Usage (from Trading-Prosperity repo root):
  python scripts/round3_path_risk_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "ROUND3"
SLOW_EMA_ALPHA = 0.002
BURN = 200  # ticks to skip for EMA "below" counts (stabilize slow EMA)


def _max_down_leg_bps(r: np.ndarray) -> float:
    """Largest drop (bps) in one contiguous string of negative log returns `r`."""
    if r.size == 0:
        return 0.0
    worst = 0.0
    cur = 0.0
    for x in r:
        if x < 0.0:
            cur += x
        else:
            if cur < 0.0:
                worst = min(worst, cur)
            cur = 0.0
    if cur < 0.0:
        worst = min(worst, cur)
    return float(-worst * 1e4)


def _max_up_leg_bps(r: np.ndarray) -> float:
    """Largest rise (bps) in one contiguous string of positive log returns."""
    if r.size == 0:
        return 0.0
    best = 0.0
    cur = 0.0
    for x in r:
        if x > 0.0:
            cur += x
        else:
            if cur > 0.0:
                best = max(best, cur)
            cur = 0.0
    if cur > 0.0:
        best = max(best, cur)
    return float(best * 1e4)


def _max_drawdown_bps(m: np.ndarray) -> float:
    """From running peak, max (peak - m) / peak in bps."""
    if m.size < 2:
        return 0.0
    peak = np.maximum.accumulate(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (peak - m) / peak
    return float(np.nanmax(dd) * 1e4)


def _ema(m: np.ndarray, alpha: float) -> np.ndarray:
    n = m.size
    out = np.empty(n, dtype=np.float64)
    out[0] = m[0]
    o = 1.0 - float(alpha)
    a = float(alpha)
    for i in range(1, n):
        out[i] = o * out[i - 1] + a * m[i]
    return out


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
            if r.size < 1:
                continue
            mmd = _max_drawdown_bps(m)
            drift = float(np.log(m[-1] / m[0]) * 1e4) if m[0] > 0 else 0.0
            dleg = _max_down_leg_bps(r)
            uleg = _max_up_leg_bps(r)
            ema = _ema(m, SLOW_EMA_ALPHA)
            n = len(m)
            lo, hi = min(BURN, n), n
            if hi > lo:
                frac_below = float(np.mean(m[lo:hi] < ema[lo:hi]))
            else:
                frac_below = float("nan")
            rows.append(
                {
                    "day": day,
                    "product": prod,
                    "n": n,
                    "max_drawdown_bps": mmd,
                    "day_drift_bps": drift,
                    "max_down_leg_bps": dleg,
                    "max_up_leg_bps": uleg,
                    "time_below_slow_ema": frac_below,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No data.")
        return

    print("=" * 80)
    print("ROUND3 path risk - mid path vs end-of-day drift")
    print(
        f"Data: {DATA}   |  slow_ema_alpha={SLOW_EMA_ALPHA}  burn_in_ticks={BURN} for P(mid<ema)"
    )
    print("=" * 80)

    for name in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
        sub = out[out["product"] == name].sort_values("day")
        if sub.empty:
            continue
        print(f"\n--- {name} ---")
        print(
            sub[
                [
                    "day",
                    "n",
                    "max_drawdown_bps",
                    "max_down_leg_bps",
                    "max_up_leg_bps",
                    "time_below_slow_ema",
                    "day_drift_bps",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:,.4f}" if isinstance(x, float) else f"{x}",
            )
        )

    print("\n" + "=" * 80)
    print("Full grid: max_drawdown_bps  (day columns)")
    print("=" * 80)
    piv = out.pivot_table(
        index="product", columns="day", values="max_drawdown_bps", aggfunc="first"
    )
    print(piv.sort_index().to_string(float_format=lambda x: f"{x:,.1f}"))

    print("\n" + "=" * 80)
    print("Full grid: time_below_slow_ema  (day columns, 0-1 fraction)")
    print("=" * 80)
    piv2 = out.pivot_table(
        index="product", columns="day", values="time_below_slow_ema", aggfunc="first"
    )
    print(piv2.sort_index().to_string(float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    main()
