"""
INTARIAN_PEPPER_ROOT — staged buy-and-hold with EMA dip leg.

1. **Phase 1:** Lift best ask until position reaches ``_FIRST_TARGET`` (67).
2. **Phase 2:** Until ``_LIMIT`` (80), only lift when ``best_ask < ema(mid)``
   (mid EWMA — same formula as ``224853_root_only_buynhold.py``).
3. **Fallback:** If still short of ``_LIMIT`` and ``timestamp >= _FORCE_COMPLETE_TS``,
   lift best ask for the remainder (avoid ending under 80 if no dip).

``ema`` is stored in ``traderData`` for replay / GUI overlay.

Tune: ``_FIRST_TARGET``, ``_LIMIT``, ``_EMA_SPAN``, ``_FORCE_COMPLETE_TS``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_LIMIT = 80
_FIRST_TARGET = 67
_EMA_SPAN = 20
# Late-day safety: complete position even if ask never prints below EMA again.
_FORCE_COMPLETE_TS = 900_000

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
        self.first_target = _FIRST_TARGET
        self.ema_span = _EMA_SPAN
        self.force_complete_ts = _FORCE_COMPLETE_TS

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
        ts = int(state.timestamp)

        for product, od in state.order_depths.items():
            if product != INTARIAN_PEPPER_ROOT:
                result[product] = []
                continue

            mid = _mid(od)
            ema_val: Optional[float] = None
            if mid is not None:
                prev_ema = td.get("ema")
                prev_f = float(prev_ema) if prev_ema is not None else None
                ema_val = _next_ema(prev_f, float(mid), self.ema_span)
                td["ema"] = ema_val

            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if not od.sell_orders:
                result[product] = orders
                continue

            best_ask = min(od.sell_orders.keys())

            if pos < self.first_target:
                want = self.first_target - pos
                if want > 0:
                    orders.append(Order(product, best_ask, want))
            elif pos < self.limit:
                want = self.limit - pos
                dip_ok = ema_val is not None and float(best_ask) < float(ema_val)
                force_ok = ts >= self.force_complete_ts
                if want > 0 and (dip_ok or force_ok):
                    orders.append(Order(product, best_ask, want))

            result[product] = orders

        new_td = json.dumps(td, separators=(",", ":"))
        return result, conversions, new_td
