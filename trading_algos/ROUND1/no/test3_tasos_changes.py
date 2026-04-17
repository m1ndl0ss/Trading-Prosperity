from datamodel import OrderDepth, TradingState, Order
from typing import List, Optional

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

_BOOT_LEN = 10

# Replay note (gui/replay_engine.py): fills come only from **trades CSV** at this
# timestamp (tape), not from eating the displayed book depth. So position ramps
# over many ticks when each tick has little/no matching tape volume. Order size
# should be the full **want** (room to limit); do not cap by L1 `avail` or you
# discard fills when tape size exceeds top-of-book size on a tick.


class Trader:
    def __init__(self):
        self.limits = {
            ASH_COATED_OSMIUM: 80,
            INTARIAN_PEPPER_ROOT: 80,
        }

        # Pepper: first BOOT_LEN mids decide trend, then buy-and-hold or sell-and-hold.
        self._pepper_boot: list[float] = []
        self._pepper_hold: Optional[str] = None  # None until decided, then "long" | "short" | "flat"

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    @staticmethod
    def _pepper_trend_from_first_10(mids: list[float]) -> int:
        """+1 uptrend, -1 downtrend, 0 ambiguous (flat)."""
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
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders)
                    best_ask = min(order_depth.sell_orders)

                    if best_bid < best_ask:
                        bid_px = best_bid + 1
                        ask_px = best_ask - 1

                        if bid_px < ask_px:
                            pos = state.position.get(ASH_COATED_OSMIUM, 0)
                            bid_vol = 80 - pos
                            ask_vol = -80 - pos
                            if bid_vol > 0:
                                orders.append(Order(ASH_COATED_OSMIUM, bid_px, bid_vol))
                            if ask_vol < 0:
                                orders.append(Order(ASH_COATED_OSMIUM, ask_px, ask_vol))

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
