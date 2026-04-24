#!/usr/bin/env python3
"""
Visualise ROUND3 price tapes: delta-1 products and VEV (voucher) options.

VEV series have wildly different price levels (OTM ~0.5 vs ATM ~1200), so we use:
  - Facet grid: mid vs time, one y-scale per strike (honest levels).
  - Heatmap: row-wise min–max normalisation so all strikes are comparable on one panel.

Usage (repo root):

  python visualize_round3_prices.py
  python visualize_round3_prices.py --data Data/ROUND3 --out misc/round3_viz --step 5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _strike_key(product: str) -> int:
    m = re.match(r"VEV_(\d+)$", product)
    return int(m.group(1)) if m else -1


def _discover_price_csvs(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("prices_round_*_day_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices_round_*_day_*.csv under {data_dir}")
    return files


def _day_index_from_name(path: Path) -> int:
    m = re.search(r"day_(\d+)\.csv$", path.name)
    if not m:
        raise ValueError(f"Cannot parse day from {path.name}")
    return int(m.group(1))


def load_all_prices(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in _discover_price_csvs(data_dir):
        day = _day_index_from_name(p)
        df = pd.read_csv(p, sep=";")
        df["tape_day"] = day
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Global time for plotting: preserve order across days
    ts_span = out.groupby("tape_day")["timestamp"].max().max() + 1000
    out["t_plot"] = out["tape_day"].astype(np.int64) * int(ts_span) + out["timestamp"]
    return out


def plot_delta1(df: pd.DataFrame, out: Path, step: int) -> None:
    sub = df[df["product"].isin(["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"])].copy()
    sub = sub.sort_values(["t_plot", "product"])
    if step > 1:
        sub = sub.iloc[::step]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, prod in zip(axes, ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]):
        g = sub[sub["product"] == prod]
        ax.plot(g["t_plot"], g["mid_price"], lw=0.6, color="C0")
        ax.set_ylabel("mid")
        ax.set_title(prod)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t (day * offset + timestamp)")
    fig.suptitle("Delta-1 mids (ROUND3)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_vev_facets(df: pd.DataFrame, out: Path, step: int) -> None:
    vev_syms = sorted(
        [p for p in df["product"].unique() if p.startswith("VEV_")],
        key=_strike_key,
    )
    sub = df[df["product"].isin(vev_syms)].copy()
    sub = sub.sort_values(["product", "t_plot"])
    if step > 1:
        sub = sub.iloc[::step]

    n = len(vev_syms)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for i, sym in enumerate(vev_syms):
        ax = axes_flat[i]
        g = sub[sub["product"] == sym]
        ax.plot(g["t_plot"], g["mid_price"], lw=0.5, color="C1")
        ax.set_title(sym, fontsize=10)
        ax.grid(True, alpha=0.3)
    for j in range(len(vev_syms), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("VEV mids — separate y-axis per strike (levels not comparable across panels)", y=1.01)
    fig.supxlabel("t (day * offset + timestamp)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_vev_heatmap_relative(df: pd.DataFrame, out: Path, max_cols: int) -> None:
    """
    Pivot: rows = t_plot (downsampled), columns = strike, values = mid.
    Each column min-max normalised so OTM and ATM shapes show on one colour scale.
    """
    vev_syms = sorted(
        [p for p in df["product"].unique() if p.startswith("VEV_")],
        key=_strike_key,
    )
    sub = df[df["product"].isin(vev_syms)].copy()
    piv = sub.pivot_table(index="t_plot", columns="product", values="mid_price", aggfunc="last")
    piv = piv.sort_index()
    strikes = [f"K={_strike_key(s)}" for s in vev_syms]
    piv = piv[[s for s in vev_syms]]
    piv.columns = strikes

    # Downsample rows if huge
    if len(piv) > max_cols:
        idx = np.linspace(0, len(piv) - 1, max_cols, dtype=int)
        piv = piv.iloc[idx]

    arr = piv.to_numpy(dtype=float)
    col_min = np.nanmin(arr, axis=0)
    col_max = np.nanmax(arr, axis=0)
    denom = np.where((col_max - col_min) < 1e-12, 1.0, col_max - col_min)
    norm = (arr - col_min) / denom
    norm = np.clip(norm, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(norm.T, aspect="auto", interpolation="nearest", cmap="magma", origin="lower")
    ax.set_yticks(np.arange(len(strikes)))
    ax.set_yticklabels(strikes, fontsize=9)
    ax.set_xlabel(f"time index (downsampled; {len(piv)} columns)")
    ax.set_title("VEV relative mids (column-wise min–max): compare dynamics, not levels")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="normalised mid")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_vev_smile_snapshots(df: pd.DataFrame, out: Path) -> None:
    """mid(K) at start / mid / end of each tape day — one column per day."""
    vev_syms = sorted(
        [p for p in df["product"].unique() if p.startswith("VEV_")],
        key=_strike_key,
    )
    ks = [_strike_key(s) for s in vev_syms]
    days = sorted(df["tape_day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 4.5), squeeze=False)
    axes = axes[0]

    for ax, day in zip(axes, days):
        gday = df[(df["tape_day"] == day) & (df["product"].isin(vev_syms))]
        if gday.empty:
            continue
        times = sorted(gday["t_plot"].unique())
        picks = [("start", times[0]), ("mid", times[len(times) // 2]), ("end", times[-1])]
        for label, t in picks:
            snap = gday[gday["t_plot"] == t].set_index("product")["mid_price"].reindex(vev_syms)
            ax.plot(ks, snap.values, lw=1.4, label=label)
        ax.set_title(f"tape day {day}")
        ax.set_xlabel("strike K")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mid (VEV)")
    fig.suptitle("VEV strike cross-sections (vs extract, check parity / monotonicity by eye)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ROUND3 price CSVs (delta-1 + VEV).")
    ap.add_argument("--data", type=Path, default=ROOT / "Data" / "ROUND3")
    ap.add_argument("--out", type=Path, default=ROOT / "misc" / "round3_viz")
    ap.add_argument(
        "--step",
        type=int,
        default=3,
        help="Keep every Nth row for line plots (speed; 1 = all points).",
    )
    ap.add_argument(
        "--heatmap-cols",
        type=int,
        default=800,
        help="Max time columns for heatmap (downsampled).",
    )
    args = ap.parse_args()

    data_dir = args.data.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_prices(data_dir)
    step = max(1, args.step)

    plot_delta1(df, out_dir / "round3_delta1_mids.png", step)
    plot_vev_facets(df, out_dir / "round3_vev_facets.png", step)
    plot_vev_heatmap_relative(df, out_dir / "round3_vev_heatmap_relative.png", args.heatmap_cols)
    plot_vev_smile_snapshots(df, out_dir / "round3_vev_smile_slices.png")

    print(f"Wrote PNGs under {out_dir}")


if __name__ == "__main__":
    main()
