"""
Strategy list for the dashboard: auto-discovered from `trading_algos/*.py`
(flat) or `trading_algos/{subdir}/*.py` (round-scoped).
"""

from __future__ import annotations

from typing import FrozenSet, List, Tuple, Type

from gui.algorithm_discovery import discover_strategies, find_strategy, strategy_goal_hint

StrategyEntry = Tuple[str, str, Type]


def list_all_strategies(
    products: FrozenSet[str],
    subdir: str | None = None,
) -> List[StrategyEntry]:
    return [
        (d.strategy_id, d.label, d.trader_cls)
        for d in discover_strategies(products, subdir=subdir)
    ]


def list_strategies(
    product: str,
    products: FrozenSet[str],
    subdir: str | None = None,
) -> List[StrategyEntry]:
    return [
        (d.strategy_id, d.label, d.trader_cls)
        for d in discover_strategies(products, subdir=subdir)
        if product in d.products
    ]


def get_strategy_entry(
    product: str,
    strategy_id: str,
    products: FrozenSet[str],
    subdir: str | None = None,
) -> StrategyEntry | None:
    d = find_strategy(products, strategy_id, product, subdir=subdir)
    if d is None:
        return None
    return (d.strategy_id, d.label, d.trader_cls)


def goal_hint(strategy_id: str | None, subdir: str | None = None) -> str | None:
    if not strategy_id:
        return None
    return strategy_goal_hint(strategy_id, subdir=subdir)
