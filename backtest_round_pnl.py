#!/usr/bin/env python3
"""
Offline PnL for ROUND-style CSV folders (same tape-based replay as gui/dashboard).

Runs one fresh Trader() per product per day — matches Streamlit's per-tab replay.

Usage (from Trading-Prosperity repo root):

  python backtest_round_pnl.py --trader trading_algos/ROUND1/test3_fair_ref_regime.py

  python backtest_round_pnl.py --trader trading_algos/ROUND1/test3_fair_ref_regime.py --round ROUND1

  python backtest_round_pnl.py --trader path/to/trader.py --data Data/TUTORIAL
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_engine import read_prices_csv, read_trades_csv, replay_summary, replay_to_dataframe

OSMIUM = "ASH_COATED_OSMIUM"
ROOT_PEPPER = "INTARIAN_PEPPER_ROOT"
PRODUCTS: Tuple[str, ...] = (OSMIUM, ROOT_PEPPER)


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
    ap = argparse.ArgumentParser(description="Tape-replay PnL (Osmium, Root, combined).")
    ap.add_argument(
        "--trader",
        type=Path,
        required=True,
        help="Path to trader .py file (e.g. trading_algos/ROUND1/test3_fair_ref_regime.py)",
    )
    ap.add_argument(
        "--round",
        default="ROUND1",
        help="Subfolder under --data (default: ROUND1)",
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
    round_dir = data_root / args.round
    if not round_dir.is_dir():
        raise SystemExit(f"Data directory not found: {round_dir}")

    trader_path = args.trader
    if not trader_path.is_file():
        alt = ROOT / trader_path
        if alt.is_file():
            trader_path = alt
        else:
            raise SystemExit(f"Trader file not found: {args.trader}")

    trader_cls = _load_trader(trader_path.resolve())
    days = _discover_days(round_dir)

    totals = {OSMIUM: 0.0, ROOT_PEPPER: 0.0}
    rows: List[Tuple[str, float, float, float]] = []

    for day_key, p_csv, t_csv in days:
        p_os = _pnl_for_day(trader_cls, p_csv, t_csv, OSMIUM, args.cash)["pnl"]
        p_root = _pnl_for_day(trader_cls, p_csv, t_csv, ROOT_PEPPER, args.cash)["pnl"]
        totals[OSMIUM] += p_os
        totals[ROOT_PEPPER] += p_root
        rows.append((day_key, p_os, p_root, p_os + p_root))

    combined = totals[OSMIUM] + totals[ROOT_PEPPER]

    print(f"Trader:     {trader_path}")
    print(f"Data:       {round_dir}")
    print(f"Initial $:  {args.cash:g}")
    print()
    hdr = f"{'Day':<8} {'Osmium PnL':>14} {'Root PnL':>14} {'Day combined':>14}"
    print(hdr)
    print("-" * len(hdr))
    for day_key, p_os, p_root, day_c in rows:
        print(f"{day_key:<8} {p_os:14,.3f} {p_root:14,.3f} {day_c:14,.3f}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<8} {totals[OSMIUM]:14,.3f} {totals[ROOT_PEPPER]:14,.3f} {combined:14,.3f}")
    print()
    print(f"Osmium (ASH_COATED_OSMIUM): {totals[OSMIUM]:,.3f}")
    print(f"Root (INTARIAN_PEPPER_ROOT): {totals[ROOT_PEPPER]:,.3f}")
    print(f"Combined: {combined:,.3f}")


if __name__ == "__main__":
    main()
