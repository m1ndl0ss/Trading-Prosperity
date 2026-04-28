"""
EOD M2M PnL for:
  - Mark 14 on HYDROGEL_PACK only (per day)
  - Mark 49 on VELVETFRUIT_EXTRACT only (per day)

Uses counterparty_eod() with symbol_filter; last price from full tape.

Run: python misc/mark14_m49_product_pnl.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from counterparty_eod_pnl import counterparty_eod  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "ROUND4")
HYDRO = frozenset({"HYDROGEL_PACK"})
VELVET = frozenset({"VELVETFRUIT_EXTRACT"})
DAYS = (
    (1, "trades_round_4_day_1.csv"),
    (2, "trades_round_4_day_2.csv"),
    (3, "trades_round_4_day_3.csv"),
)


def main() -> int:
    print("Single-product EOD PnL (cash + M2M @ last print for that product)\n")
    rows = []
    for day, fn in DAYS:
        path = os.path.join(BASE, fn)
        df = pd.read_csv(path, sep=";")

        for label, name, flt in (
            ("Mark 14 x HYDRO", "Mark 14", HYDRO),
            ("Mark 49 x VELVET", "Mark 49", VELVET),
        ):
            cash, m2m, last, pos, _ = counterparty_eod(df, name, flt)
            pnl = cash + m2m
            sym = next(iter(flt))
            q = pos.get(sym, 0)
            lp = last.get(sym, 0.0)
            rows.append((day, label, pnl, cash, m2m, q, lp))

    # wide table for user
    print(f"  {'day':>3}  {'slice':<20}  {'EOD PnL':>12}  {'cash':>14}  {'M2M':>12}  {'pos':>6}  last")
    for day, label, pnl, cash, m2m, q, lp in rows:
        print(
            f"  {day:>3}  {label:<20}  {pnl:>12,.2f}  {cash:>14,.2f}  {m2m:>12,.2f}  {q:>+6.0f}  {lp:>10.2f}"
        )

    # compact
    print("\n--- copy-friendly ---\n")
    m14 = [r for r in rows if "Mark 14" in r[1]]
    m49 = [r for r in rows if "Mark 49" in r[1]]
    print("Mark 14 (HYDRO only):")
    for d, _, pnl, _, _, q, _ in m14:
        print(f"  Day {d}:  EOD = {pnl:,.2f}  (pos = {q:+d})")
    print("\nMark 49 (VELVET only):")
    for d, _, pnl, _, _, q, _ in m49:
        print(f"  Day {d}:  EOD = {pnl:,.2f}  (pos = {q:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
