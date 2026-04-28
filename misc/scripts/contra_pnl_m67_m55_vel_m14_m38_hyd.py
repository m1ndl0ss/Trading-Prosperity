"""
EOD (cash + M2M) for four slices (Round 4 days 1-3 in Data/ROUND4):

  * WITH  = that Mark's *own* book on the product (counterparty_eod, one symbol).
  * AGAINST = merged *counterparties* (everyone on the other side of that Mark
    on that product) — same as aggregate_contra_merged in contra_against_m55.py

Columns requested:
  - WITH  Mark 67  on VELVET,  AGAINST Mark 55 on VELVET
  - WITH  Mark 14  on HYDRO,  AGAINST Mark 38 on HYDRO

Run:  python misc/contra_pnl_m67_m55_vel_m14_m38_hyd.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contra_against_m55 import aggregate_contra_merged  # noqa: E402
from counterparty_eod_pnl import counterparty_eod  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "ROUND4")
DAYS = (
    (1, "trades_round_4_day_1.csv"),
    (2, "trades_round_4_day_2.csv"),
    (3, "trades_round_4_day_3.csv"),
)
VELVET = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"

# (column title, mode "own" | "contra", mark, product)
# own  = that Mark's PnL on the product; contra = other side of that Mark
SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("WITH  M67  VELVET (67 own book)", "own", "Mark 67", VELVET),
    ("AGAINST M55  VELVET (contra 55)", "contra", "Mark 55", VELVET),
    ("WITH  M14  HYDRO  (14 own book)", "own", "Mark 14", HYDRO),
    ("AGAINST M38  HYDRO (contra 38)", "contra", "Mark 38", HYDRO),
)


def pnl_for(df: pd.DataFrame, mode: str, m: str, prd: str) -> float:
    flt = frozenset({prd})
    if mode == "own":
        c, m2m, _l, _p, _mb = counterparty_eod(df, m, flt)
        return float(c + m2m)
    if mode == "contra":
        p, _c, _pos, _m2m = aggregate_contra_merged(df, m, prd)
        return float(p)
    raise ValueError(mode)


def main() -> int:
    rows: list[dict] = []
    for day, fn in DAYS:
        path = os.path.join(BASE, fn)
        df = pd.read_csv(path, sep=";")
        r: dict = {"day": day}
        pvals: list[float] = []
        for label, mode, m, prd in SPECS:
            pnl = pnl_for(df, mode, m, prd)
            r[label] = pnl
            pvals.append(pnl)
        r["Subtotal VELVET (M67_+_contra55)"] = pvals[0] + pvals[1]
        r["Subtotal HYDRO (M14_+_contra38)"] = pvals[2] + pvals[3]
        r["Sum all 4"] = sum(pvals)
        rows.append(r)

    out = pd.DataFrame(rows)
    p = out.set_index("day")

    col_order = [x[0] for x in SPECS] + [
        "Subtotal VELVET (M67_+_contra55)",
        "Subtotal HYDRO (M14_+_contra38)",
        "Sum all 4",
    ]
    p = p[col_order]

    print("EOD PnL (Shells)\n")
    print("  WITH  = that Mark's own book on the product (follow their side).")
    print("  AGAINST = everyone on the other side of that Mark (merged contra).\n")
    s = p.to_string(
        float_format=lambda x: f"{x:,.0f}" if abs(x) >= 0.5 else f"{x:,.2f}"
    )
    print(s)
    out_path = os.path.join(
        os.path.dirname(__file__), "contra_pnl_m67_55_14_38_by_day.csv"
    )
    p.to_csv(out_path)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
