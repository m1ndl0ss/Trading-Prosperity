from datamodel import Order, TradingState, Symbol
from typing import List, Dict
import json

class Trader:
    def run(self, state: TradingState):
        """
        The simulation calls this method every timestep.
        Returns: (result: Dict[Symbol, List[Order]], conversions: int, traderData: str)
        """
        result = {}
        conversions = 0  # Default to 0 unless trading Round 4/Location Arbitrage assets
        
        # --- Persistence Logic (traderData) ---
        # If you need to track state (e.g., price history), decode it here.
        # If no data exists yet, initialize it.
        try:
            trader_data_dict = json.loads(state.traderData) if state.traderData else {}
        except:
            trader_data_dict = {}

        # --- Product Logic ---
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)

            # --- 1. EMERALDS (Fixed Fair Value Strategy) ---
            if product == "EMERALDS":
                FAIR_VALUE = 10000
                LIMIT = 80
                
                # Market Taking
                for price, vol in order_depth.sell_orders.items():
                    if price < FAIR_VALUE and current_position < LIMIT:
                        num = min(abs(vol), LIMIT - current_position)
                        orders.append(Order(product, price, num))
                        current_position += num
                for price, vol in order_depth.buy_orders.items():
                    if price > FAIR_VALUE and current_position > -LIMIT:
                        num = max(-vol, -LIMIT - current_position)
                        orders.append(Order(product, price, num))
                        current_position += num
                
                # Market Making (9999 / 10001)
                if current_position < LIMIT:
                    orders.append(Order(product, FAIR_VALUE - 1, LIMIT - current_position))
                if current_position > -LIMIT:
                    orders.append(Order(product, FAIR_VALUE + 1, -LIMIT - current_position))

            # --- 2. TOMATOES (Wall-Mid Strategy) ---
            elif product == "TOMATOES":
                LIMIT = 80
                EDGE = 6  # Quoting edge relative to Wall-Mid

                if order_depth.buy_orders and order_depth.sell_orders:
                    # Find deep liquidity walls
                    bid_wall = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
                    ask_wall = max(order_depth.sell_orders, key=lambda k: abs(order_depth.sell_orders[k]))
                    wall_mid = (bid_wall + ask_wall) / 2
                    
                    # Market Taking against Wall-Mid
                    for price, vol in sorted(order_depth.sell_orders.items()):
                        if price < wall_mid and current_position < LIMIT:
                            num = min(abs(vol), LIMIT - current_position)
                            orders.append(Order(product, price, num))
                            current_position += num
                    for price, vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if price > wall_mid and current_position > -LIMIT:
                            num = max(-vol, -LIMIT - current_position)
                            orders.append(Order(product, price, num))
                            current_position += num

                    # Market Making around Wall-Mid
                    if current_position < LIMIT:
                        orders.append(Order(product, int(wall_mid - EDGE), LIMIT - current_position))
                    if current_position > -LIMIT:
                        orders.append(Order(product, int(wall_mid + EDGE), -LIMIT - current_position))

            result[product] = orders

        # --- Save State (traderData) ---
        # Serialize your dictionary back to a string for the next timestep
        new_trader_data = json.dumps(trader_data_dict)

        return result, conversions, new_trader_data