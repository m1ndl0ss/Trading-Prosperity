"""Print mean mid_price for HYDROGEL_PACK per day and pooled (ROUND3 Data by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=ROOT / "Data" / "ROUND3",
        help="Folder with prices_*.csv",
    )
    ap.add_argument("--product", default="HYDROGEL_PACK")
    args = ap.parse_args()
    folder = args.data
    product = args.product

    files = sorted(folder.glob("prices_*.csv"))
    if not files:
        raise SystemExit(f"No prices_*.csv in {folder}")

    rows: list[tuple[int, str, float, int]] = []
    for f in files:
        df = pd.read_csv(f, sep=";")
        h = df[df["product"] == product]
        if h.empty:
            continue
        day = int(h["day"].iloc[0])
        mids = h["mid_price"].astype(float)
        rows.append((day, f.name, float(mids.mean()), len(mids)))

    rows.sort(key=lambda x: x[0])
    print(f"{product} - mean mid_price (from prices CSV)")
    print(f"{'day':<6} {'file':<36} {'n_ticks':>10} {'mean_mid':>14}")
    print("-" * 70)
    total_w = 0.0
    total_n = 0
    for day, name, m, n in rows:
        print(f"{day:<6} {name:<36} {n:>10} {m:>14.4f}")
        total_w += m * n
        total_n += n
    if total_n == 0:
        raise SystemExit(f"No rows for {product} in {folder}")
    print("-" * 70)
    grand = total_w / total_n
    print(f"{'ALL':<6} {'(all ticks, unweighted mean of mids)':<36} {total_n:>10} {grand:>14.4f}")


if __name__ == "__main__":
    main()
