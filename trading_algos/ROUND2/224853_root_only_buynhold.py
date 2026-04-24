"""
INTARIAN_PEPPER_ROOT — buy and hold (minimal).

Each tick: if position is below the cap, lift the best ask for the remaining size.
No sells, no trend/boot logic.

EWMA of mid (pepper) is updated every tick and stored in traderData as ``ema`` forreplay/GUI overlay (same key as ``fair_and_ema_from_traderdata`` in replay_engine).

Tune ``_LIMIT`` and ``_EMA_SPAN``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_LIMIT = 80

# EWMA span on mid (pandas ``ewm(span=..., adjust=False).mean()`` equivalent)
_EMA_SPAN = 20

def _ewma_alpha(span: int) -> float:
    return 2.0 / (float(span) + 1.0)
def _mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
def _next_ema(prev: Optional[float], mid: float, span: int) -> float:
    a = _ewma_alpha(span)
    if prev is None:
        return mid
    return a * mid + (1.0 - a) * prev


class Trader:
    def __init__(self) -> None:
        self.limit = _LIMIT
        self.ema_span = _EMA_SPAN

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        result: Dict[str, List[Order]] = {}
        conversions = 0

        for product, od in state.order_depths.items():
            if product != INTARIAN_PEPPER_ROOT:
                result[product] = []
                continue

            mid = _mid(od)
            if mid is not None:
                prev_ema = td.get("ema")
                prev_f = float(prev_ema) if prev_ema is not None else None
                td["ema"] = _next_ema(prev_f, float(mid), self.ema_span)

            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if pos < self.limit and od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                want = self.limit - pos
                if want > 0:
                    orders.append(Order(product, best_ask, want))

            result[product] = orders

        new_td = json.dumps(td, separators=(",", ":"))
        return result, conversions, new_td
