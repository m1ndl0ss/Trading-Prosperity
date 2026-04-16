"""Osmium regime-filter strategy — dual-EMA anchor (no hard-coded fair value).

Instead of pinning fair value to a constant (FAIR_REF), this version derives
it from a very slow exponential moving average.  The slow EMA acts as the
"long-run fair price" and is compared against a faster EMA that tracks the
current market level.

  fast_ema: alpha = FAST_ALPHA  (~0.06)   — current market temperature
  slow_ema: alpha = SLOW_ALPHA  (~0.008)  — long-run fair value proxy

  fair_dev = fast_ema - slow_ema

  cheap  regime: fair_dev < -REGIME_THRESHOLD  →  buy-only  (asks suppressed)
  exp    regime: fair_dev >  REGIME_THRESHOLD  →  sell-only (bids suppressed)
  neutral regime: |fair_dev| <= REGIME_THRESHOLD → two-sided MM (v3 logic)

Why a slow EMA instead of FAIR_REF?
  The slow EMA self-calibrates to the actual long-run mean observed so far,
  making the strategy robust if osmium's structural price level drifts over
  the competition rather than sitting perfectly at 10 000 forever.

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
_TD_KEY   = "ser"  # traderData key for osmium state

# ── Osmium regime constants (tune these) ──────────────────────────────────────
FAST_ALPHA       = 0.06    # fast EMA — tracks current market level
SLOW_ALPHA       = 0.008   # slow EMA — proxy for long-run fair value
                            # ~125-tick half-life, adapts over many minutes
REGIME_THRESHOLD = 5.0     # |fast_ema - slow_ema| above this → one-sided mode
INIT_PRICE       = 10_000.0  # cold-start value for both EMAs


def _td_load(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class Trader:
    def __init__(self):
        self.limits = {
            ASH_COATED_OSMIUM:    _LIMIT,
            INTARIAN_PEPPER_ROOT: _LIMIT,
        }
        self._pepper_boot: list[float] = []
        self._pepper_hold: Optional[str] = None

    # ── helpers ───────────────────────────────────────────────────────────────

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

    # ── osmium logic ──────────────────────────────────────────────────────────

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

        # Compute mid (fall back to whichever side exists)
        if has_bid and has_ask:
            mid = (best_bid + best_ask) / 2.0
        elif has_ask:
            mid = float(best_ask)
        else:
            mid = float(best_bid)

        # Update both EMAs (initialise to INIT_PRICE on first tick)
        prev_fast = blob.get("fast_ema", INIT_PRICE)
        prev_slow = blob.get("slow_ema", INIT_PRICE)
        fast_ema  = (1.0 - FAST_ALPHA) * prev_fast + FAST_ALPHA * mid
        slow_ema  = (1.0 - SLOW_ALPHA) * prev_slow + SLOW_ALPHA * mid
        blob["fast_ema"] = fast_ema
        blob["slow_ema"] = slow_ema

        # ── Regime decision ───────────────────────────────────────────────────
        fair_dev  = fast_ema - slow_ema
        can_buy   = fair_dev <= REGIME_THRESHOLD    # suppressed in expensive regime
        can_sell  = fair_dev >= -REGIME_THRESHOLD   # suppressed in cheap regime

        # ── Two-sided book ────────────────────────────────────────────────────
        if has_bid and has_ask and best_bid < best_ask:
            bid_px = best_bid + 1
            ask_px = best_ask - 1
            if bid_px < ask_px:
                if can_buy:
                    room_buy = _LIMIT - pos
                    if room_buy > 0:
                        orders.append(Order(ASH_COATED_OSMIUM, bid_px, room_buy))
                if can_sell:
                    room_sell = pos + _LIMIT
                    if room_sell > 0:
                        orders.append(Order(ASH_COATED_OSMIUM, ask_px, -room_sell))
            return orders

        # ── One-sided book ────────────────────────────────────────────────────
        if has_ask and not has_bid:
            if can_sell:
                ask_px    = best_ask - 1
                room_sell = pos + _LIMIT
                if ask_px > 0 and room_sell > 0:
                    orders.append(Order(ASH_COATED_OSMIUM, ask_px, -room_sell))
            return orders

        if has_bid and not has_ask:
            if can_buy:
                bid_px   = best_bid + 1
                room_buy = _LIMIT - pos
                if room_buy > 0:
                    orders.append(Order(ASH_COATED_OSMIUM, bid_px, room_buy))
            return orders

        return orders

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, state: TradingState):
        td   = _td_load(state.traderData)
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
