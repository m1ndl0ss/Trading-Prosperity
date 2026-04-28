"""
Merged wallet of all counterparties on the other side of Mark 55's prints
(optional single-symbol filter; default = all products Mark 55 appears in).

In the R4 training tape, Mark 55 only prints on VELVETFRUIT_EXTRACT, so
"all products" and "VELVET only" coincide here. EOD = cash + M2M at last
print per symbol from the full file.

Sanity: Mark 55 full-book EOD + merged contra = 0 (bilateral, same mark).

Run: python misc/contra_against_m55.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from counterparty_eod_pnl import counterparty_eod  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "ROUND4")
DAYS = (
    (1, "trades_round_4_day_1.csv"),
    (2, "trades_round_4_day_2.csv"),
    (3, "trades_round_4_day_3.csv"),
)
MARK = "Mark 55"
VELVET = "VELVETFRUIT_EXTRACT"


def last_prices_full(df: pd.DataFrame) -> dict[str, float]:
    d = df.reset_index().sort_values(["timestamp", "index"], kind="mergesort")
    return {
        str(sym): float(p)
        for sym, p in d.groupby("symbol", sort=False)["price"].last().items()
    }


def aggregate_contra_merged(
    df: pd.DataFrame,
    mark: str,
    symbol_filter: str | None = None,
) -> tuple[float, float, dict[str, int], float]:
    """
    One merged book for everyone on the non-`mark` side of any trade where
    `mark` is buyer or seller. Optional `symbol_filter` to one product.
    """
    d = df.reset_index().sort_values(["timestamp", "index"], kind="mergesort")
    last = last_prices_full(df)
    cash = 0.0
    pos: dict[str, int] = defaultdict(int)

    for r in d.itertuples(index=False):
        s = str(r.symbol)
        if symbol_filter is not None and s != symbol_filter:
            continue
        b, zl = r.buyer, r.seller
        p = float(r.price)
        q = int(r.quantity)
        if b == mark and zl == mark:
            continue
        if b == mark and zl != mark:
            cash += p * q
            pos[s] -= q
        elif zl == mark and b != mark:
            cash -= p * q
            pos[s] += q

    m2m = 0.0
    for s, q in list(pos.items()):
        if q == 0:
            del pos[s]
            continue
        m2m += q * last.get(s, 0.0)
    pnl = cash + m2m
    return pnl, cash, dict(pos), m2m


def main() -> int:
    print(f"Merged contra vs {MARK} (all 3 R4 days in capsule)\n")

    for label, flt in (
        ("All products (in tape, Mark 55 rows)", None),
        (f"VELVET only ({VELVET})", VELVET),
    ):
        print("-" * 72)
        print(f"  {label}")
        print("-" * 72)
        print(
            f"  {'day':>3}  {'EOD PnL':>14}  {'cash':>14}  {'M2M':>14}  "
            f"{'EOD pos':<28}  {''}"
        )
        for day, fn in DAYS:
            path = os.path.join(BASE, fn)
            df = pd.read_csv(path, sep=";")
            pnl, cash, pos, m2m = aggregate_contra_merged(df, MARK, flt)
            pos_s = " ".join(f"{k}:{q:+d}" for k, q in sorted(pos.items()) if q)
            if not pos_s:
                pos_s = "flat"
            print(
                f"  {day:>3}  {pnl:>14,.2f}  {cash:>14,.2f}  {m2m:>14,.2f}  {pos_s}"
            )
        print()

    print("Sanity: full-book EOD for Mark 55 + merged contra (all products)  ->  0.00")
    for day, fn in DAYS:
        df = pd.read_csv(os.path.join(BASE, fn), sep=";")
        c, m, _l, _p, _m2 = counterparty_eod(df, MARK, None)
        m55 = c + m
        contra, _, _, _ = aggregate_contra_merged(df, MARK, None)
        print(f"  Day {day}:  {m55 + contra:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
