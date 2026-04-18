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
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_engine import read_prices_csv, replay_summary, replay_to_dataframe, read_trades_csv

# Display names for common products (optional)
_SHORT_LABELS: Dict[str, str] = {
    "ASH_COATED_OSMIUM": "Osmium",
    "INTARIAN_PEPPER_ROOT": "Root",
}

_ROUND_DIR_RE = re.compile(r"^ROUND\d+$", re.IGNORECASE)


def _load_trader(trader_path: Path) -> Type:
    spec = importlib.util.spec_from_file_location("trader_under_test", trader_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load trader module: {trader_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, "Trader", None)
    if not isinstance(cls, type):
        raise SystemExit(f"No Trader class in {trader_path}")
    return cls


def _infer_round_from_trader_path(trader_path: Path) -> str | None:
    """Return ROUND2 / TUTORIAL if that segment appears in the trader path."""
    for part in trader_path.resolve().parts:
        if part == "TUTORIAL":
            return "TUTORIAL"
        if _ROUND_DIR_RE.match(part):
            return part.upper()
    return None


def _discover_days(data_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    for prices in sorted(data_dir.glob("prices_*.csv")):
        trades = data_dir / prices.name.replace("prices_", "trades_", 1)
        if not trades.is_file():
            continue
        stem = prices.stem
        day_key = stem.split("_day_")[-1] if "_day_" in stem else stem
        pairs.append((day_key, prices, trades))
    if not pairs:
        raise SystemExit(f"No prices_*.csv + trades_*.csv pairs in {data_dir}")

    def _sort_key(item: Tuple[str, Path, Path]) -> Tuple[int, str]:
        dk = item[0]
        try:
            return (int(dk), dk)
        except ValueError:
            return (10**9, dk)

    pairs.sort(key=_sort_key)
    return pairs


def _discover_products(prices_path: Path) -> List[str]:
    """All product symbols present in a prices file (stable sort)."""
    prices = read_prices_csv(prices_path)
    if "product" not in prices.columns:
        raise SystemExit(f"No 'product' column in {prices_path}")
    u = prices["product"].dropna().astype(str).unique().tolist()
    return sorted(u)


def _discover_products_union(day_pairs: List[Tuple[str, Path, Path]]) -> List[str]:
    """Union of products across all price files for the round (sorted)."""
    seen: set[str] = set()
    for _dk, p_csv, _t_csv in day_pairs:
        seen.update(_discover_products(p_csv))
    return sorted(seen)


def _label(sym: str, width: int) -> str:
    s = _SHORT_LABELS.get(sym, sym)
    if len(s) > width:
        s = s[: max(0, width - 2)] + ".."
    return s


def _pnl_for_day(
    trader_cls: Type,
    prices_path: Path,
    trades_path: Path,
    product: str,
    initial_cash: float,
) -> Dict[str, Any]:
    prices = read_prices_csv(prices_path)
    trades = read_trades_csv(trades_path)
    df = replay_to_dataframe(prices, trades, product, trader_cls, initial_cash=initial_cash)
    summ = replay_summary(df, initial_cash)
    return summ


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
        inferred = _infer_round_from_trader_path(trader_path)
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

    trader_cls = _load_trader(trader_path.resolve())
    days = _discover_days(round_dir)
    products = _discover_products_union(days)
    if not products:
        raise SystemExit(f"No products found under {round_dir}")

    totals = {p: 0.0 for p in products}
    rows: List[Tuple[str, Dict[str, float], float]] = []

    for day_key, p_csv, t_csv in days:
        day_pnls: Dict[str, float] = {}
        for sym in products:
            pnl = _pnl_for_day(trader_cls, p_csv, t_csv, sym, args.cash)["pnl"]
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
