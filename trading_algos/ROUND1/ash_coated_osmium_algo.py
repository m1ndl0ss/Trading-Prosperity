"""
ASH_COATED_OSMIUM: quote one tick inside the best market on each side.

Strategy
--------
Each tick, read the live order book:
  - bid price  = best_bid  + 1   (one above the best existing bid)
  - ask price  = best_ask  - 1   (one below the best existing ask)

This undercuts competitors on both sides without crossing the spread.
Volumes are sized to bring position back to the limit in one order.

Observed spread (Day -1): best bid ~9985 / best ask ~10001 → ~16 ticks wide,
so the quoted spread of 14 ticks is well inside the market.

Position limit: set to 50 — verify against the official competition rules.
"""

from typing import Dict, List, Tuple

from datamodel import Order, TradingState

SYMBOL = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 50


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td_in = getattr(state, "traderData", "") or ""

        depth = state.order_depths.get(SYMBOL)
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return {}, 0, td_in

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        # Crossed or touching book — skip this tick to avoid adverse fills.
        if best_bid >= best_ask:
            return {}, 0, td_in

        bid_px = best_bid + 1
        ask_px = best_ask - 1

        # Skip if our quotes would cross (e.g. one-tick spread).
        if bid_px >= ask_px:
            return {}, 0, td_in

        pos = state.position.get(SYMBOL, 0)
        bid_vol = POSITION_LIMIT - pos   # positive → buy order
        ask_vol = -POSITION_LIMIT - pos  # negative → sell order

        orders: List[Order] = []
        if bid_vol > 0:
            orders.append(Order(SYMBOL, bid_px, bid_vol))
        if ask_vol < 0:
            orders.append(Order(SYMBOL, ask_px, ask_vol))

        return {SYMBOL: orders}, 0, td_in
