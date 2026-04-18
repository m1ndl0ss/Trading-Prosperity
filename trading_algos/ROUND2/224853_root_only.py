"""
INTARIAN_PEPPER_ROOT — buy and hold (minimal).

Each tick: if position is below the cap, lift the best ask for the remaining size.
No sells, no trend/boot logic, no traderData.

Tune `_LIMIT` only.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from datamodel import Order, TradingState

INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_LIMIT = 80


class Trader:
    def __init__(self) -> None:
        self.limit = _LIMIT

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        for product, od in state.order_depths.items():
            if product != INTARIAN_PEPPER_ROOT:
                result[product] = []
                continue

            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if pos < self.limit and od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                want = self.limit - pos
                if want > 0:
                    orders.append(Order(product, best_ask, want))

            result[product] = orders

        return result, conversions, ""
