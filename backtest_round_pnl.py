#!/usr/bin/env python3
"""
Offline PnL for ROUND-style CSV folders (same tape-based replay as gui/dashboard).

Runs one fresh Trader() per product per day — matches Streamlit's per-tab replay.

Products are read from the prices CSVs in the selected data folder (any round).

Usage (from Trading-Prosperity repo root):

  python backtest_round_pnl.py --trader trading_algos/ROUND1/test3_fair_ref_regime.py

  python backtest_round_pnl.py --trader trading_algos/ROUND2/224853_root_only.py --round ROUND2

  # Same Python file, different tape → set --round explicitly (independent of folder name):
  python backtest_round_pnl.py --trader trading_algos/ROUND2/foo.py --round ROUND1
  python backtest_round_pnl.py --trader trading_algos/ROUND2/foo.py --round ROUND2

  # If --round is omitted, the data folder is inferred from the trader path
  # (e.g. .../trading_algos/ROUND2/foo.py -> Data/ROUND2), else defaults to ROUND1.

  python backtest_round_pnl.py --trader path/to/trader.py --data Data/TUTORIAL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tape_benchmark import (
    discover_days,
    discover_products_union,
    infer_round_from_trader_path,
    load_trader,
    pnl_for_day,
)

# Display names for common products (optional)
_SHORT_LABELS: Dict[str, str] = {
    "ASH_COATED_OSMIUM": "Osmium",
    "INTARIAN_PEPPER_ROOT": "Root",
}

def _label(sym: str, width: int) -> str:
    s = _SHORT_LABELS.get(sym, sym)
    if len(s) > width:
        s = s[: max(0, width - 2)] + ".."
    return s


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tape-replay PnL for all products in the round's prices CSVs."
    )
    ap.add_argument(
        "--trader",
        type=Path,
        required=True,
        help="Path to trader .py (e.g. trading_algos/ROUND2/224853_root_only.py)",
    )
    ap.add_argument(
        "--round",
        default=None,
        metavar="NAME",
        help="Which dataset under Data/ to replay (e.g. ROUND1, ROUND2). "
        "This chooses the CSVs only — it does not need to match the folder of --trader. "
        "If omitted: inferred from a ROUNDn (or TUTORIAL) segment in the trader path, else ROUND1.",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Data root (default: <repo>/Data)",
    )
    ap.add_argument(
        "--cash",
        type=float,
        default=0.0,
        help="Initial cash for replay (default: 0)",
    )
    args = ap.parse_args()

    data_root = args.data if args.data is not None else ROOT / "Data"

    trader_path = args.trader
    if not trader_path.is_file():
        alt = ROOT / trader_path
        if alt.is_file():
            trader_path = alt
        else:
            raise SystemExit(f"Trader file not found: {args.trader}")

    data_round_hint = ""
    if args.round:
        round_name = args.round
        data_round_hint = "data round from --round (same .py can use --round ROUND1 vs ROUND2)"
    else:
        inferred = infer_round_from_trader_path(trader_path)
        if inferred:
            round_name = inferred
            data_round_hint = (
                f"data round inferred as {round_name} from trader path "
                f"(override with --round ROUND1 / ROUND2)"
            )
        else:
            round_name = "ROUND1"
            data_round_hint = "data round defaulted to ROUND1 (set --round or use .../trading_algos/ROUNDn/...)"

    round_dir = data_root / round_name
    if not round_dir.is_dir():
        raise SystemExit(f"Data directory not found: {round_dir}")

    trader_cls = load_trader(trader_path.resolve())
    days = discover_days(round_dir)
    products = discover_products_union(days)
    if not products:
        raise SystemExit(f"No products found under {round_dir}")

    totals = {p: 0.0 for p in products}
    rows: List[Tuple[str, Dict[str, float], float]] = []

    for day_key, p_csv, t_csv in days:
        day_pnls: Dict[str, float] = {}
        for sym in products:
            pnl = pnl_for_day(trader_cls, p_csv, t_csv, sym, args.cash)["pnl"]
            day_pnls[sym] = float(pnl)
            totals[sym] += float(pnl)
        combined = sum(day_pnls.values())
        rows.append((day_key, day_pnls, combined))

    grand_total = sum(totals.values())

    col_w = 14
    prod_headers = "".join(f"{_label(p, col_w - 4):>{col_w}}" for p in products)

    print(f"Trader:     {trader_path}")
    print(f"Data:       {round_dir}")
    print(f"Note:       {data_round_hint}")
    print(f"Products:   {', '.join(products)}")
    print(f"Initial $:  {args.cash:g}")
    print()
    hdr = f"{'Day':<8}{prod_headers}{'Day combined':>{col_w}}"
    print(hdr)
    print("-" * len(hdr))
    for day_key, day_pnls, day_c in rows:
        cells = "".join(f"{day_pnls[p]:{col_w},.3f}" for p in products)
        print(f"{day_key:<8}{cells}{day_c:{col_w},.3f}")
    print("-" * len(hdr))
    tot_cells = "".join(f"{totals[p]:{col_w},.3f}" for p in products)
    print(f"{'TOTAL':<8}{tot_cells}{grand_total:{col_w},.3f}")
    print()
    for p in products:
        lbl = _SHORT_LABELS.get(p, p)
        print(f"{lbl} ({p}): {totals[p]:,.3f}")
    print(f"Combined: {grand_total:,.3f}")


if __name__ == "__main__":
    main()
