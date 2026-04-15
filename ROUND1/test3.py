from datamodel import OrderDepth, TradingState, Order
from typing import List
import collections
import numpy as np

class Trader:
    def __init__(self):
        # Configuration
        self.limits = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }
        
        # State tracking
        self.pepper_prices = collections.deque(maxlen=80)
        self.osmium_fair_value = 10000

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
            if product == "ASH_COATED_OSMIUM":
                # Sell if price > 10000
                if len(order_depth.buy_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    if best_bid > self.osmium_fair_value:
                        # Market sell into existing bids
                        sell_quantity = max(-order_depth.buy_orders[best_bid], -max_limit - current_position)
                        if sell_quantity != 0:
                            orders.append(Order(product, best_bid, sell_quantity))
                
                # Buy if price < 10000
                if len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    if best_ask < self.osmium_fair_value:
                        # Market buy into existing asks
                        buy_quantity = min(-order_depth.sell_orders[best_ask], max_limit - current_position)
                        if buy_quantity != 0:
                            orders.append(Order(product, best_ask, buy_quantity))

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