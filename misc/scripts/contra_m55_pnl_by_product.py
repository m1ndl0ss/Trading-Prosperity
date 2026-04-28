"""
Per-product EOD for the *merged* counterparty vs Mark 55: each product is
its own sub-book (only trades in that symbol), cash + M2M @ last for that
symbol. Sum of product PnLs = merged all-product contra for that day.

Run: python misc/contra_m55_pnl_by_product.py
Output: table + misc/contra_m55_pnl_by_product.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contra_against_m55 import (  # noqa: E402
    MARK,
    aggregate_contra_merged,
)
BASE = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "ROUND4")
DAYS = (
    (1, "trades_round_4_day_1.csv"),
    (2, "trades_round_4_day_2.csv"),
    (3, "trades_round_4_day_3.csv"),
)
OUT_CSV = os.path.join(os.path.dirname(__file__), "contra_m55_pnl_by_product.csv")


def m55_symbols(df: pd.DataFrame) -> list[str]:
    s = df[(df["buyer"] == MARK) | (df["seller"] == MARK)]["symbol"].dropna()
    return sorted(s.astype(str).unique().tolist())


def main() -> int:
    rows: list[dict] = []

    for day, fn in DAYS:
        path = os.path.join(BASE, fn)
        df = pd.read_csv(path, sep=";")
        total, _, _, _ = aggregate_contra_merged(df, MARK, None)
        syms = m55_symbols(df)
        ssum = 0.0
        for sym in syms:
            pnl, cash, pos, m2m = aggregate_contra_merged(df, MARK, sym)
            q = int(pos.get(sym, 0))
            ssum += pnl
            rows.append(
                {
                    "day": day,
                    "product": sym,
                    "eod_pnl": round(pnl, 2),
                    "cash": round(cash, 2),
                    "eod_m2m": round(m2m, 2),
                    "eod_pos": q,
                }
            )
        if abs(ssum - total) > 0.01:
            raise SystemExit(
                f"Day {day}: sum of product PnL {ssum} != merged {total}"
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {os.path.normpath(OUT_CSV)}\n")

    # Pretty: pivot pnl
    m = out[~out["product"].str.startswith("_")]
    pvt = m.pivot_table(
        index="product",
        columns="day",
        values="eod_pnl",
        aggfunc="first",
    )
    print("EOD PnL (merged counterparty to Mark 55) by product and day  (Shells)\n")
    print(pvt.to_string())
    print("\nEOD pos (same book, at last print) by product and day  (lots)\n")
    m2 = out[~out["product"].str.startswith("_")].copy()
    pvt2 = m2.pivot_table(
        index="product",
        columns="day",
        values="eod_pos",
        aggfunc="first",
    )
    print(pvt2.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
