#!/usr/bin/env python3
"""
Refit the quadratic smile v_hat = A*m^2 + B*m + C from training tape (same as ROUND3/6aPlot).

Reads VELVETFRUIT_EXTRACT + VEV_* mids from Data/ROUND3, infers IV per tick, takes
per-(day, strike) median (m, v), then np.polyfit(m, v, 2).

Usage (repo root):

  python scripts/refit_vev_smile.py
  python scripts/refit_vev_smile.py --data Data --round ROUND3

Prints A, B, C and a one-line summary for pasting into trading_algos/ROUND3/tryout.py.
"""

from __future__ import annotations

import argparse
from math import log, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_N = NormalDist()

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
START_TTE_MAP = {0: 5.0, 1: 4.0, 2: 3.0}


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2)


def find_iv(market_price: float, S: float, K: float, T: float) -> float:
    low, high = 0.0001, 2.0
    for _ in range(24):
        mid = (low + high) / 2.0
        if bs_call_price(S, K, T, mid) > market_price:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "Data", help="Data root")
    ap.add_argument("--round", default="ROUND3", help="Subfolder, e.g. ROUND3")
    ap.add_argument("--subsample", type=int, default=100, help="Use every Nth row (same as 6aPlot 100)")
    args = ap.parse_args()
    ddir = args.data / args.round
    if not ddir.is_dir():
        raise SystemExit(f"Not a directory: {ddir}")

    all_medians: list[dict] = []
    for day_i in (0, 1, 2):
        path = ddir / f"prices_round_3_day_{day_i}.csv"
        if not path.is_file():
            print(f"Warning: missing {path}, skip day {day_i}")
            continue
        df = pd.read_csv(path, sep=";")
        extract = df[df["product"] == "VELVETFRUIT_EXTRACT"][["timestamp", "mid_price"]].rename(
            columns={"mid_price": "S"}
        )
        t0 = START_TTE_MAP[day_i]
        for K in STRIKES:
            opt = df[df["product"] == f"VEV_{K}"][["timestamp", "mid_price"]]
            merged = pd.merge(extract, opt, on="timestamp").dropna()
            if merged.empty:
                continue
            sample = merged.iloc[:: args.subsample]
            ivs: list = []
            ms: list = []
            for _, row in sample.iterrows():
                T_years = (t0 - (row["timestamp"] / 1_000_000.0)) / 365.0
                if T_years <= 0:
                    continue
                try:
                    iv = find_iv(float(row["mid_price"]), float(row["S"]), float(K), float(T_years))
                except (ValueError, ZeroDivisionError):
                    continue
                m = K / float(row["S"])
                ivs.append(iv)
                ms.append(m)
            if not ivs:
                continue
            all_medians.append(
                {
                    "day": day_i,
                    "K": K,
                    "m": float(np.median(ms)),
                    "v": float(np.median(ivs)),
                }
            )

    if len(all_medians) < 3:
        raise SystemExit(f"Too few clean points: {len(all_medians)}")

    clean = pd.DataFrame(all_medians)
    coeffs = np.polyfit(clean["m"], clean["v"], 2)
    A, B, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    print("=== VEV smile refit (median IV per day x strike, polyfit deg 2) ===")
    print(f"  Points: {len(clean)}  (STRIKES x days with data)")
    print()
    print("  # Paste into tryout.py:")
    print(f"  A, B, C = {A:.6f}, {B:.6f}, {C:.6f}")
    print()
    print("  v_hat = A * (m**2) + B * m + C")
    rms = float(np.sqrt(((clean["v"] - np.poly1d(coeffs)(clean["m"])) ** 2).mean()))
    print(f"  In-sample RMSE (v): {rms:.6f}")


if __name__ == "__main__":
    main()
