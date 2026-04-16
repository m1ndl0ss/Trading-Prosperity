from datamodel import OrderDepth, TradingState, Order
from typing import List, Optional

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_BOOT_LEN = 10
_LIMIT_ASH = 80

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic findings (osmium, all 3 ROUND1 days combined):
#
# WHY enhanced/v2 are WORSE than simple
# ─────────────────────────────────────
# 1. Aggressive takes destroy spread earnings.
#    Enhanced does 250 buys at best_ask and 116 sells at best_bid.
#    That costs ~15 ticks per fill vs a passive quote at best_bid+1 / best_ask-1.
#    Total lost PnL from takes alone: ~5,490 ticks (3,750 + 1,740).
#
# 2. Smaller base size (26 vs 80) in v2 reduces fill count without any benefit.
#    v2 loses ~244 fills vs simple, enhanced loses only ~50 (extra fills from
#    one-sided ticks offset the reduction elsewhere).
#
# 3. One-sided ticks have VERY little tape volume (~233 units across all 3 days)
#    so handling them gives minimal uplift locally.
#    (On the real exchange they matter more since any incoming order fills us.)
#
# CORRECT diagnosis
# ─────────────────
# simple is near-optimal for the local tape-based replay because:
#   - It always posts MAXIMUM size → captures all available tape on every tick.
#   - It quotes at the tightest valid improvement (+1 bid, -1 ask) → best prices.
#   - Any trickery that reduces size OR posts at a worse price hurts PnL.
#
# The ONLY structural gap vs simple: ~2,307 one-sided ticks get NO orders.
# This version adds a one-sided handler at full room size — same offset logic.
# ──────────────────────────────────────────────────────────────────────────────


class Trader:
    def __init__(self):
        self.limits = {
            ASH_COATED_OSMIUM: _LIMIT_ASH,
            INTARIAN_PEPPER_ROOT: 80,
        }

        self._pepper_boot: list[float] = []
        self._pepper_hold: Optional[str] = None

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2

    @staticmethod
    def _pepper_trend_from_first_10(mids: list[float]) -> int:
        """+1 uptrend, -1 downtrend, 0 ambiguous."""
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

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = state.traderData if state.traderData else ""

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            current_position = state.position.get(product, 0)
            max_limit = self.limits.get(product, 20)

            if product == ASH_COATED_OSMIUM:
                pos = current_position
                has_bid = bool(order_depth.buy_orders)
                has_ask = bool(order_depth.sell_orders)

                if has_bid and has_ask:
                    best_bid = max(order_depth.buy_orders)
                    best_ask = min(order_depth.sell_orders)

                    if best_bid < best_ask:
                        bid_px = best_bid + 1
                        ask_px = best_ask - 1

                        if bid_px < ask_px:
                            bid_vol = _LIMIT_ASH - pos
                            ask_vol = -_LIMIT_ASH - pos
                            if bid_vol > 0:
                                orders.append(Order(ASH_COATED_OSMIUM, bid_px, bid_vol))
                            if ask_vol < 0:
                                orders.append(Order(ASH_COATED_OSMIUM, ask_px, ask_vol))

                elif has_ask and not has_bid:
                    # No bids in the book — post a sell at best_ask-1 (same offset).
                    # ~62% of one-sided ask ticks snap back next tick (data analysis).
                    best_ask = min(order_depth.sell_orders)
                    ask_px = best_ask - 1
                    room_sell = pos + _LIMIT_ASH
                    if ask_px > 0 and room_sell > 0:
                        orders.append(Order(ASH_COATED_OSMIUM, ask_px, -room_sell))

                elif has_bid and not has_ask:
                    # No asks in the book — post a buy at best_bid+1 (same offset).
                    best_bid = max(order_depth.buy_orders)
                    bid_px = best_bid + 1
                    room_buy = _LIMIT_ASH - pos
                    if room_buy > 0:
                        orders.append(Order(ASH_COATED_OSMIUM, bid_px, room_buy))

            elif product == INTARIAN_PEPPER_ROOT:
                mid_price = self.get_mid_price(order_depth)
                if mid_price is not None:
                    if self._pepper_hold is None:
                        self._pepper_boot.append(mid_price)
                        if len(self._pepper_boot) >= _BOOT_LEN:
                            t = self._pepper_trend_from_first_10(self._pepper_boot)
                            if t > 0:
                                self._pepper_hold = "long"
                            elif t < 0:
                                self._pepper_hold = "short"
                            else:
                                self._pepper_hold = "flat"

                    if self._pepper_hold == "long":
                        if current_position < max_limit and order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            want = max_limit - current_position
                            if want > 0:
                                orders.append(Order(product, best_ask, want))

                    elif self._pepper_hold == "short":
                        if current_position > -max_limit and order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            want = current_position + max_limit
                            if want > 0:
                                orders.append(Order(product, best_bid, -want))

            result[product] = orders

        return result, conversions, trader_data
