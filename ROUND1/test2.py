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
        # We keep your 80-tick window for Pepper Root
        self.pepper_prices = collections.deque(maxlen=80)
        
        # NEW: A 20-tick window for Osmium to calculate dynamic fair value
        self.osmium_prices = collections.deque(maxlen=20) 

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
        conversions = 0
        trader_data = state.traderData if state.traderData else ""

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            current_position = state.position.get(product, 0)
            max_limit = self.limits.get(product, 20)
            
            # --- IMPROVED STRATEGY: ASH_COATED_OSMIUM (Dynamic Mean Reversion) ---
            if product == "ASH_COATED_OSMIUM":
                mid_price = self.get_mid_price(order_depth)
                
                if mid_price is not None:
                    self.osmium_prices.append(mid_price)
                    
                    # Calculate dynamic fair value based on recent history
                    # Fallback to 10000 if we don't have enough data yet
                    if len(self.osmium_prices) > 5:
                        dynamic_fair_value = np.mean(self.osmium_prices)
                    else:
                        dynamic_fair_value = 10000

                    # Sell if the best bid is higher than our dynamic fair value
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        if best_bid > dynamic_fair_value:
                            sell_quantity = max(-order_depth.buy_orders[best_bid], -max_limit - current_position)
                            if sell_quantity != 0:
                                orders.append(Order(product, best_bid, sell_quantity))
                    
                    # Buy if the best ask is lower than our dynamic fair value
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        if best_ask < dynamic_fair_value:
                            buy_quantity = min(-order_depth.sell_orders[best_ask], max_limit - current_position)
                            if buy_quantity != 0:
                                orders.append(Order(product, best_ask, buy_quantity))

            # --- ORIGINAL STRATEGY: INTARIAN_PEPPER_ROOT (Trend Following from test3.py) ---
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

        return result, conversions, trader_data