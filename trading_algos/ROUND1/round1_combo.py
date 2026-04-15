from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Order, TradingState

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

# ASH_COATED_OSMIUM ~ sit one tick inside the best market on each side
# Observed spread (Day -1): best bid ~9985 / best ask ~10001 → ~16 ticks wide.
# Verify POSITION_LIMIT against the official competition rules.
POSITION_LIMIT_ASH = 80

# INTARIAN_PEPPER_ROOT ~ placeholder (spread ~10994–11006, mid ~11000)
# Observed spread (Day -1): best bid ~10994 / best ask ~11006 → ~12 ticks wide.
POSITION_LIMIT_PEPPER = 80


class Trader:
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

            # INTARIAN_PEPPER_ROOT
            # TODO: implement strategy.
            elif product == INTARIAN_PEPPER_ROOT:
                pass

            result[product] = orders

        new_trader_data = json.dumps(trader_data_dict, separators=(",", ":"))
        return result, conversions, new_trader_data
