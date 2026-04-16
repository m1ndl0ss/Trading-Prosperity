"""Osmium regime filter + deviation-scaled order size (FAIR_REF).

Same regime logic as test3_fair_ref_regime.py, but when the market is in a
one-sided regime (cheap → buy-only, expensive → sell-only), order size ramps
from MIN_SCALE_SIZE up to full remaining room as |EMA - FAIR_REF| grows beyond
REGIME_THRESHOLD.  This preserves inventory capacity for deeper dislocations
instead of maxing out the limit as soon as the threshold is crossed.

Neutral regime (|fair_dev| <= REGIME_THRESHOLD): full room on both sides
(v3-style two-sided MM).

Pepper: unchanged.
"""

from __future__ import annotations

import json

from datamodel import Order, OrderDepth, TradingState
from typing import Any, Dict, List, Optional

ASH_COATED_OSMIUM    = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_BOOT_LEN = 10
_LIMIT    = 80
_TD_KEY   = "frrs"  # traderData key (distinct from test3_fair_ref_regime "frr")

# ── Osmium constants ──────────────────────────────────────────────────────────
FAIR_REF         = 10_000.0
EWMA_ALPHA       = 0.111
REGIME_THRESHOLD = 5.748

# Scaling: excess beyond threshold → intensity in [0,1], then size in
# [MIN_SCALE_SIZE, room].  excess >= SCALE_RANGE → full room.
MIN_SCALE_SIZE = 36
SCALE_RANGE      = 23.31


def _td_load(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _vol_scaled(room: int, excess: float) -> int:
    """Map remaining room to a capped size; excess = how far into one-sided regime."""
    if room <= 0:
        return 0
    if SCALE_RANGE <= 0:
        return min(room, max(1, MIN_SCALE_SIZE))
    intensity = min(1.0, max(0.0, excess) / SCALE_RANGE)
    lo = min(MIN_SCALE_SIZE, room)
    cap = int(round(lo + intensity * (room - lo)))
    return min(room, max(1, cap))


class Trader:
    def __init__(self):
        self.limits = {
            ASH_COATED_OSMIUM:    _LIMIT,
            INTARIAN_PEPPER_ROOT: _LIMIT,
        }
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

    def _orders_osmium(
        self, od: OrderDepth, pos: int, blob: Dict[str, Any]
    ) -> List[Order]:
        orders: List[Order] = []
        has_bid = bool(od.buy_orders)
        has_ask = bool(od.sell_orders)
        if not has_bid and not has_ask:
            return orders

        best_bid = max(od.buy_orders) if has_bid else None
        best_ask = min(od.sell_orders) if has_ask else None

        if has_bid and has_ask:
            mid = (best_bid + best_ask) / 2.0
        elif has_ask:
            mid = float(best_ask)
        else:
            mid = float(best_bid)

        prev_ema = blob.get("ema", FAIR_REF)
        ema = (1.0 - EWMA_ALPHA) * prev_ema + EWMA_ALPHA * mid
        blob["ema"] = ema

        fair_dev = ema - FAIR_REF
        can_buy  = fair_dev <= REGIME_THRESHOLD
        can_sell = fair_dev >= -REGIME_THRESHOLD
        neutral  = abs(fair_dev) <= REGIME_THRESHOLD

        excess_buy  = max(0.0, -fair_dev - REGIME_THRESHOLD)   # cheap regime depth
        excess_sell = max(0.0, fair_dev - REGIME_THRESHOLD)   # expensive regime depth

        # ── Two-sided book ────────────────────────────────────────────────────
        if has_bid and has_ask and best_bid < best_ask:
            bid_px = best_bid + 1
            ask_px = best_ask - 1
            if bid_px < ask_px:
                if can_buy:
                    room_buy = _LIMIT - pos
                    if room_buy > 0:
                        if neutral:
                            vol = room_buy
                        else:
                            vol = _vol_scaled(room_buy, excess_buy)
                        if vol > 0:
                            orders.append(Order(ASH_COATED_OSMIUM, bid_px, vol))
                if can_sell:
                    room_sell = pos + _LIMIT
                    if room_sell > 0:
                        if neutral:
                            vol = room_sell
                        else:
                            vol = _vol_scaled(room_sell, excess_sell)
                        if vol > 0:
                            orders.append(Order(ASH_COATED_OSMIUM, ask_px, -vol))
            return orders

        if has_ask and not has_bid:
            if can_sell:
                ask_px = best_ask - 1
                room_sell = pos + _LIMIT
                if ask_px > 0 and room_sell > 0:
                    vol = room_sell if neutral else _vol_scaled(room_sell, excess_sell)
                    if vol > 0:
                        orders.append(Order(ASH_COATED_OSMIUM, ask_px, -vol))
            return orders

        if has_bid and not has_ask:
            if can_buy:
                bid_px = best_bid + 1
                room_buy = _LIMIT - pos
                if room_buy > 0:
                    vol = room_buy if neutral else _vol_scaled(room_buy, excess_buy)
                    if vol > 0:
                        orders.append(Order(ASH_COATED_OSMIUM, bid_px, vol))
            return orders

        return orders

    def run(self, state: TradingState):
        td = _td_load(state.traderData)
        blob = td.get(_TD_KEY, {})
        if not isinstance(blob, dict):
            blob = {}

        result: Dict[str, List[Order]] = {}
        conversions = 0

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if product == ASH_COATED_OSMIUM:
                orders = self._orders_osmium(od, pos, blob)

            elif product == INTARIAN_PEPPER_ROOT:
                mid = self._get_mid(od)
                if mid is not None:
                    if self._pepper_hold is None:
                        self._pepper_boot.append(mid)
                        if len(self._pepper_boot) >= _BOOT_LEN:
                            t = self._pepper_trend(self._pepper_boot)
                            self._pepper_hold = (
                                "long" if t > 0 else "short" if t < 0 else "flat"
                            )

                    lim = self.limits[product]
                    if self._pepper_hold == "long" and pos < lim and od.sell_orders:
                        ba = min(od.sell_orders)
                        want = lim - pos
                        if want > 0:
                            orders.append(Order(product, ba, want))
                    elif self._pepper_hold == "short" and pos > -lim and od.buy_orders:
                        bb = max(od.buy_orders)
                        want = pos + lim
                        if want > 0:
                            orders.append(Order(product, bb, -want))

            result[product] = orders

        td[_TD_KEY] = blob
        return result, conversions, json.dumps(td, separators=(",", ":"))
