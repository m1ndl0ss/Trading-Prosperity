"""EMERALDS: bid 10_000 − 7, ask 10_000 + 7 (fixed fair)."""

from typing import Dict, List, Tuple

from datamodel import Order, TradingState

EMERALDS = "EMERALDS"
POSITION_LIMIT_EMERALDS = 80

FAIR = 10_000
SPREAD_HALF = 7

#We know that fair is 10k
#Day 1, 8 of 208 ≈ 3.85% are filled at fair value
#Day 2, 3 / 191 ≈ 1.57% are filled at fair value
#So we are ok to no look at fair value in order to maximize PnL
#Sitting at max ask, min bid would be the best strategy iff we where the only ones on the order book
#But we are not so sitting at one above for bid and one below for ask would is the best strategy for this round

#Simulated PnL: +7567.00
#IMC tester PnL: +1050.00

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td_in = getattr(state, "traderData", "") or ""

        depth = state.order_depths.get(EMERALDS)
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return {}, 0, td_in

        bid_px = FAIR - SPREAD_HALF
        ask_px = FAIR + SPREAD_HALF

        pos = state.position.get(EMERALDS, 0)
        bid_vol = POSITION_LIMIT_EMERALDS - pos
        ask_vol = -POSITION_LIMIT_EMERALDS - pos

        orders: List[Order] = []
        if bid_vol > 0:
            orders.append(Order(EMERALDS, bid_px, bid_vol))
        if ask_vol < 0:
            orders.append(Order(EMERALDS, ask_px, ask_vol))

        return {EMERALDS: orders}, 0, td_in