from __future__ import annotations

import collections
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from datamodel import Order, OrderDepth, TradingState

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

# ASH_COATED_OSMIUM ~ sit one tick inside the best market on each side
# Observed spread (Day -1): best bid ~9985 / best ask ~10001 → ~16 ticks wide.
# Verify POSITION_LIMIT against the official competition rules.
POSITION_LIMIT_ASH = 80


class Trader:
    def __init__(self):
        self.limits = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80,
        }
        self.pepper_prices = collections.deque(maxlen=80)
        self.osmium_fair_value = 10000

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        try:
            trader_data_dict: Dict[str, Any] = (
                json.loads(state.traderData) if state.traderData else {}
            )
        except (json.JSONDecodeError, TypeError):
            trader_data_dict = {}

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []

            current_position = state.position.get(product, 0)
            max_limit = self.limits.get(product, 20)

            # ASH_COATED_OSMIUM
            # Quotes one tick inside the live best bid and best ask every tick,
            # sizing each order to bring inventory back to the position limit.
            # No state carried between ticks.
            if product == ASH_COATED_OSMIUM:
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders)
                    best_ask = min(order_depth.sell_orders)

                    # Skip ticks with a crossed or zero-width spread.
                    if best_bid < best_ask:
                        bid_px = best_bid + 1
                        ask_px = best_ask - 1

                        # Skip if the one-tick improvement would cross itself.
                        if bid_px < ask_px:
                            pos = state.position.get(ASH_COATED_OSMIUM, 0)
                            bid_vol = POSITION_LIMIT_ASH - pos
                            ask_vol = -POSITION_LIMIT_ASH - pos
                            if bid_vol > 0:
                                orders.append(Order(ASH_COATED_OSMIUM, bid_px, bid_vol))
                            if ask_vol < 0:
                                orders.append(Order(ASH_COATED_OSMIUM, ask_px, ask_vol))

            # INTARIAN_PEPPER_ROOT — same logic as test3.py (80-tick MA trend)
            elif product == INTARIAN_PEPPER_ROOT:
                mid_price = self.get_mid_price(order_depth)
                if mid_price is not None:
                    self.pepper_prices.append(mid_price)

                    if len(self.pepper_prices) == self.pepper_prices.maxlen:
                        ema = np.mean(self.pepper_prices)  # Simple MA for stability

                        # Trend is Up: Buy
                        if mid_price > ema + 1 and current_position < max_limit:
                            best_ask = min(order_depth.sell_orders.keys())
                            buy_vol = min(
                                -order_depth.sell_orders[best_ask],
                                max_limit - current_position,
                            )
                            if buy_vol != 0:
                                orders.append(Order(product, best_ask, buy_vol))

                        # Trend is Down: Sell
                        elif mid_price < ema - 1 and current_position > -max_limit:
                            best_bid = max(order_depth.buy_orders.keys())
                            sell_vol = max(
                                -order_depth.buy_orders[best_bid],
                                -max_limit - current_position,
                            )
                            if sell_vol != 0:
                                orders.append(Order(product, best_bid, sell_vol))

            result[product] = orders

        new_trader_data = json.dumps(trader_data_dict, separators=(",", ":"))
        return result, conversions, new_trader_data
