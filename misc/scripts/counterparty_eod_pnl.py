"""
End-of-day mark-to-market PnL for any counterparty in Round 4 trade CSVs
(same model as the former mark38_eod_pnl: flat at open, cash + M2M @ last print).

Optional: --hydro  reports a second block using only HYDROGEL_PACK rows for that
counterparty (M2M still uses the tape's last HYDRO price for the day).

Examples:
  python misc/counterparty_eod_pnl.py "Mark 14"
  python misc/counterparty_eod_pnl.py "Mark 14" --hydro
  python misc/counterparty_eod_pnl.py "Mark 38" "Mark 14"
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd

# misc/scripts/ -> two levels up to repo root, then Data/ROUND4
_DATA_R4 = os.path.join(
    os.path.dirname(__file__), "..", "..", "Data", "ROUND4"
)
DATA_GLOB = os.path.join(_DATA_R4, "trades_round_4_day_*.csv")
HYDRO = "HYDROGEL_PACK"


def day_from_path(path: str) -> int:
    m = re.search(r"day_(\d+)\.csv$", path, re.I)
    return int(m.group(1)) if m else -1


def counterparty_eod(
    df: pd.DataFrame,
    name: str,
    symbol_filter: frozenset | None = None,
) -> tuple[float, float, dict[str, float], dict[str, int], dict[str, float]]:
    """
    EOD: cash from name's trades (+ optional symbol filter) + M2M of their position.
    last[s] = last price of s in the full file (unfiltered) for the day.
    """
    d = df.reset_index().sort_values(["timestamp", "index"], kind="mergesort")
    last: dict[str, float] = {}
    for sym, p in d.groupby("symbol", sort=False)["price"].last().items():
        last[str(sym)] = float(p)

    cash = 0.0
    pos: dict[str, int] = defaultdict(int)
    flt = symbol_filter

    for r in d.itertuples(index=False):
        s = str(r.symbol)
        if flt is not None and s not in flt:
            continue
        if r.buyer == name and r.seller == name:
            continue
        p = float(r.price)
        q = int(r.quantity)
        if r.buyer == name:
            cash -= p * q
            pos[s] += q
        elif r.seller == name:
            cash += p * q
            pos[s] -= q

    m2m_by: dict[str, float] = {}
    m2m = 0.0
    for s, q in list(pos.items()):
        if q == 0:
            del pos[s]
            continue
        lp = last[s] if s in last else 0.0
        v = q * lp
        m2m += v
        m2m_by[s] = v

    return cash, m2m, last, dict(pos), m2m_by


def print_block(
    mark: str, day: int, path: str, cash: float, m2m: float, last: dict[str, float], pos: dict, m2m_by: dict
) -> None:
    total = cash + m2m
    print("=" * 64)
    print(f"  {mark}  -  Day {day}  -  {os.path.basename(path)}")
    print("=" * 64)
    print(f"  Cash after all trades:           {cash:>16,.2f}  (Shells)")
    print(f"  M2M of open positions @ last px:  {m2m:>16,.2f}")
    print(f"  EOD mark-to-market PnL:            {total:>16,.2f}")
    if pos:
        print("  Open positions (symbol | qty | last px | M2M):")
        for s in sorted(pos.keys()):
            q = pos[s]
            lp = last.get(s, 0.0)
            v = m2m_by.get(s, q * lp)
            print(
                f"    {s:24s}  {q:>+6d}  @ {lp:>12.2f}  ->  {v:>16,.2f}"
            )
    else:
        print("  Open positions: (flat)")


def main() -> int:
    ap = argparse.ArgumentParser(description="EOD M2M PnL for one or more Marks")
    ap.add_argument("marks", nargs="*", default=["Mark 14"], help='e.g. "Mark 14"')
    ap.add_argument(
        "--hydro",
        action="store_true",
        help="Also print a block with only HYDROGEL_PACK trades for that Mark",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(DATA_GLOB), key=day_from_path)
    if not paths:
        print("No trade files found under Data/ROUND4", file=__import__("sys").stderr)
        return 1

    marks = list(args.marks) if args.marks else ["Mark 14"]
    hset = frozenset({HYDRO})

    for mark in marks:
        print(f"\n{'#' * 64}\n#  {mark}\n{'#' * 64}")
        for path in paths:
            day = day_from_path(path)
            df = pd.read_csv(path, sep=";")
            cash, m2m, last, pos, m2m_by = counterparty_eod(df, mark)
            print_block(mark, day, path, cash, m2m, last, pos, m2m_by)
            if args.hydro:
                c2, m2, l2, p2, mb2 = counterparty_eod(df, mark, hset)
                print(f"\n  --- {HYDRO} only (trades in this product; M2M @ tape last) ---")
                print_block(
                    f"{mark} [{HYDRO}]", day, path, c2, m2, l2, p2, mb2
                )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
