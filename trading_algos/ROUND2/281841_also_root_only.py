"""
INTARIAN_PEPPER_ROOT only — pepper leg from the prime_combo / 224853-style combo.

After `_BOOT_LEN` mids, sets a trend label (`short` / `long` / `flat`) and trades:
  • trend ≥ 0 (up or flat): lift the ask up to the position limit (accumulate short / sell flow).
  • trend < 0 (down): hit the bid up to the position limit (accumulate long / buy flow).

Other products: no orders. Matches `prime_combo_algo.py` root block; osmium logic removed.

Tune: `_BOOT_LEN`, `_LIMIT` below.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_BOOT_LEN = 10
_LIMIT = 80


class Trader:
    def __init__(self) -> None:
        self.limits: Dict[str, int] = {INTARIAN_PEPPER_ROOT: _LIMIT}
        self._pepper_boot: list[float] = []
        self._pepper_hold: Optional[str] = None

    def _get_mid(self, od: OrderDepth) -> Optional[float]:
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

    @staticmethod
    def _pepper_trend(mids: list[float]) -> int:
        if len(mids) < _BOOT_LEN:
            return 0
        d = mids[-1] - mids[0]
        if d > 0:
            return 1
        if d < 0:
            return -1
        step_sum = sum(mids[i + 1] - mids[i] for i in range(_BOOT_LEN - 1))
        if step_sum > 0:
            return 1
        if step_sum < 0:
            return -1
        return 0

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
            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if product == INTARIAN_PEPPER_ROOT:
                mid = self._get_mid(od)
                if mid is not None:
                    if self._pepper_hold is None:
                        self._pepper_boot.append(mid)
                        if len(self._pepper_boot) >= _BOOT_LEN:
                            t = self._pepper_trend(self._pepper_boot)
                            self._pepper_hold = (
                                "short" if t > 0 else "long" if t < 0 else "flat"
                            )

                    lim = self.limits[product]
                    if self._pepper_hold in ("short", None) and pos < lim and od.sell_orders:
                        ba = min(od.sell_orders)
                        want = lim - pos
                        if want > 0:
                            orders.append(Order(product, ba, want))
                    elif self._pepper_hold == "long" and pos > -lim and od.buy_orders:
                        bb = max(od.buy_orders)
                        want = pos + lim
                        if want > 0:
                            orders.append(Order(product, bb, -want))

            else:
                result[product] = []
                continue

            result[product] = orders

        return result, conversions, json.dumps(td, separators=(",", ":"))