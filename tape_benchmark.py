"""
Shared tape-replay helpers for backtests and parameter tuning.

Keeps day discovery, trader loading, and per-day PnL aligned with
``backtest_round_pnl.py`` / the dashboard replay.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay_engine import read_prices_csv, replay_summary, replay_to_dataframe, read_trades_csv

_ROUND_DIR_RE = re.compile(r"^ROUND\d+$", re.IGNORECASE)


def load_trader(trader_path: Path) -> Type:
    spec = importlib.util.spec_from_file_location("trader_under_test", trader_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load trader module: {trader_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, "Trader", None)
    if not isinstance(cls, type):
        raise SystemExit(f"No Trader class in {trader_path}")
    return cls


def infer_round_from_trader_path(trader_path: Path) -> str | None:
    """Return ROUND2 / TUTORIAL if that segment appears in the trader path."""
    for part in trader_path.resolve().parts:
        if part == "TUTORIAL":
            return "TUTORIAL"
        if _ROUND_DIR_RE.match(part):
            return part.upper()
    return None


def discover_days(data_dir: Path) -> List[Tuple[str, Path, Path]]:
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


def discover_products(prices_path: Path) -> List[str]:
    """All product symbols present in a prices file (stable sort)."""
    prices = read_prices_csv(prices_path)
    if "product" not in prices.columns:
        raise SystemExit(f"No 'product' column in {prices_path}")
    u = prices["product"].dropna().astype(str).unique().tolist()
    return sorted(u)


def discover_products_union(day_pairs: List[Tuple[str, Path, Path]]) -> List[str]:
    """Union of products across all price files (sorted)."""
    seen: set[str] = set()
    for _dk, p_csv, _t_csv in day_pairs:
        seen.update(discover_products(p_csv))
    return sorted(seen)


def pnl_for_day(
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
