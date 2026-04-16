"""Minimal strategy: buy exactly1 ASH_COATED_OSMIUM at the first timestamp only.

Lifts the best ask at ``timestamp == 0`` when the book has offers. All other
products and all later ticks: no orders.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from datamodel import Order, TradingState

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
_FIRST_TIMESTAMP = 0


class Trader:
    def __init__(self) -> None:
        self._handled_first_ts = False

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        td = state.traderData if state.traderData else ""
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            if product != ASH_COATED_OSMIUM:
                result[product] = []
                continue

            depth = state.order_depths[product]
            orders: List[Order] = []

            if (
                not self._handled_first_ts
                and int(state.timestamp) == _FIRST_TIMESTAMP
                and depth.sell_orders
            ):
                self._handled_first_ts = True
                best_ask = min(depth.sell_orders.keys())
                orders.append(Order(ASH_COATED_OSMIUM, best_ask, 1))

            result[product] = orders

        return result, conversions, td
