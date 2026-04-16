from datamodel import OrderDepth, TradingState, Order
from typing import List
import collections
import numpy as np

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"

# Bot-hunter findings for ASH_COATED_OSMIUM:
#   - Overall directional accuracy 41.5% → strong mean reversion (trades tend to be wrong)
#   - Trades at offset 0 (right at mid) have only 8% accuracy → extreme reversion signal
#   - No dominant qty or single-offset signal → the whole asset mean-reverts
#   Strategy: anchor quotes to a rolling fair value, skew by inventory,
#             and aggressively take when price overshoots fair by REVERT_THRESHOLD.
OSM_EDGE = 2               # passive quote ± ticks from fair value
OSM_REVERT_THRESHOLD = 8   # ticks beyond fair to trigger aggressive mean-reversion take

class Trader:
    def __init__(self):
        # Configuration
        self.limits = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }

        # State tracking
        self.pepper_prices = collections.deque(maxlen=80)
        self.osmium_mids = collections.deque(maxlen=200)  # rolling fair value window

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def run(self, state: TradingState):
        """
        Main entry point for the Prosperity exchange.
        """
        result = {}
        # 1. Conversions must be an integer (0)
        conversions = 0
        # 2. Safely capture existing trader data to pass to the next tick
        trader_data = state.traderData if state.traderData else ""

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Current inventory position
            current_position = state.position.get(product, 0)
            max_limit = self.limits.get(product, 20)
            
            # --- STRATEGY 1: ASH_COATED_OSMIUM (Mean Reversion / Market Making) ---
            if product == ASH_COATED_OSMIUM:
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders)
                    best_ask = min(order_depth.sell_orders)
                    mid = (best_bid + best_ask) / 2
                    self.osmium_mids.append(mid)

                    # Rolling fair value — mean of recent mids (mean-reversion anchor)
                    fair = round(np.mean(self.osmium_mids))

                    pos = state.position.get(ASH_COATED_OSMIUM, 0)

                    # Inventory skew: shift both quotes slightly against position
                    # so we naturally lean to close out when holding inventory
                    skew = -round(pos / 15)

                    bid_px = fair - OSM_EDGE + skew
                    ask_px = fair + OSM_EDGE + skew

                    # Don't post outside the current best — no point leaving the book
                    bid_px = min(bid_px, best_bid + 1)
                    ask_px = max(ask_px, best_ask - 1)

                    if best_bid < best_ask and bid_px < ask_px:
                        bid_vol = max_limit - pos
                        ask_vol = -max_limit - pos
                        if bid_vol > 0:
                            orders.append(Order(ASH_COATED_OSMIUM, bid_px, bid_vol))
                        if ask_vol < 0:
                            orders.append(Order(ASH_COATED_OSMIUM, ask_px, ask_vol))

                    # Aggressive mean-reversion take: when price has overshot fair by
                    # OSM_REVERT_THRESHOLD ticks, lift/hit the book — the bot-hunter
                    # showed that large offsets from mid reliably revert.
                    if best_ask <= fair - OSM_REVERT_THRESHOLD and pos < max_limit:
                        # Book is offering well below fair → buy, expect reversion up
                        take_vol = min(-order_depth.sell_orders[best_ask], max_limit - pos)
                        if take_vol > 0:
                            orders.append(Order(ASH_COATED_OSMIUM, best_ask, take_vol))
                    elif best_bid >= fair + OSM_REVERT_THRESHOLD and pos > -max_limit:
                        # Book is bidding well above fair → sell, expect reversion down
                        take_vol = max(-order_depth.buy_orders[best_bid], -max_limit - pos)
                        if take_vol < 0:
                            orders.append(Order(ASH_COATED_OSMIUM, best_bid, take_vol))

            # --- STRATEGY 2: INTARIAN_PEPPER_ROOT (Trend Following) ---
            elif product == "INTARIAN_PEPPER_ROOT":
                mid_price = self.get_mid_price(order_depth)
                if mid_price is not None:
                    self.pepper_prices.append(mid_price)
                    
                    if len(self.pepper_prices) == self.pepper_prices.maxlen:
                        ema = np.mean(self.pepper_prices) # Simple MA for stability
                        
                        # Trend is Up: Buy
                        if mid_price > ema + 1 and current_position < max_limit:
                            best_ask = min(order_depth.sell_orders.keys())
                            buy_vol = min(-order_depth.sell_orders[best_ask], max_limit - current_position)
                            if buy_vol != 0:
                                orders.append(Order(product, best_ask, buy_vol))
                        
                        # Trend is Down: Sell
                        elif mid_price < ema - 1 and current_position > -max_limit:
                            best_bid = max(order_depth.buy_orders.keys())
                            sell_vol = max(-order_depth.buy_orders[best_bid], -max_limit - current_position)
                            if sell_vol != 0:
                                orders.append(Order(product, best_bid, sell_vol))

            result[product] = orders

        # Correct return signature: (dict, int, str)
        return result, conversions, trader_data