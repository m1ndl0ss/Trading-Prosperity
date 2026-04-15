from datamodel import Order, TradingState, Symbol
from typing import List
import json

class Trader:
    def run(self, state: TradingState):
        result = {}
        
        # 1. PERSISTENCE & STATE MANAGEMENT
        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except:
            trader_data = {}

        # Initialize extrema tracking for Osmium 'Olivia' Signal
        if 'osmium_min' not in trader_data: trader_data['osmium_min'] = 1e9
        if 'osmium_max' not in trader_data: trader_data['osmium_max'] = 0
        if 'osmium_bias' not in trader_data: trader_data['osmium_bias'] = 0

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_pos = state.position.get(product, 0)
            LIMIT = 80 # Per Round 1 Manual

            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            # 2. CALCULATE WALL-MID (FAIR VALUE)
            bid_wall_price = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
            ask_wall_price = max(order_depth.sell_orders, key=lambda k: abs(order_depth.sell_orders[k]))
            fair_value = (bid_wall_price + ask_wall_price) / 2

            # 3. OLIVIA SIGNAL DETECTION (OSMIUM ONLY)
            if product == "ASH_COATED_OSMIUM":
                mid_p = (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
                trader_data['osmium_min'] = min(trader_data['osmium_min'], mid_p)
                trader_data['osmium_max'] = max(trader_data['osmium_max'], mid_p)
                
                for trade in state.market_trades.get(product, []):
                    if trade.quantity == 15:
                        if trade.price == trader_data['osmium_min']: trader_data['osmium_bias'] = 1
                        elif trade.price == trader_data['osmium_max']: trader_data['osmium_bias'] = -1

            # 4. AGGRESSIVE MARKET TAKING (Capture Immediate Profit)
            for price, vol in sorted(order_depth.sell_orders.items()):
                if price < fair_value and current_pos < LIMIT:
                    num = min(abs(vol), LIMIT - current_pos)
                    orders.append(Order(product, price, num))
                    current_pos += num
            for price, vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if price > fair_value and current_pos > -LIMIT:
                    num = max(-vol, -LIMIT - current_pos)
                    orders.append(Order(product, price, num))
                    current_pos += num

            # 5. 0 EV INVENTORY NEUTRALIZATION (Unwind Skew for Free)
            if current_pos > 30 and int(fair_value) in order_depth.buy_orders:
                clear_vol = max(-order_depth.buy_orders[int(fair_value)], -current_pos)
                orders.append(Order(product, int(fair_value), clear_vol))
                current_pos += clear_vol
            elif current_pos < -30 and int(fair_value) in order_depth.sell_orders:
                clear_vol = min(abs(order_depth.sell_orders[int(fair_value)]), -current_pos)
                orders.append(Order(product, int(fair_value), clear_vol))
                current_pos += clear_vol

            # 6. MARKET MAKING WITH DYNAMIC BIAS
            edge = 1 if product == "INTARIAN_PEPPER_ROOT" else 2
            bid_edge = edge - (1 if (product == "ASH_COATED_OSMIUM" and trader_data['osmium_bias'] == 1) else 0)
            ask_edge = edge + (1 if (product == "ASH_COATED_OSMIUM" and trader_data['osmium_bias'] == 1) else 0)

            # Soft Liquidation: Narrow edge near limits
            if current_pos > 50: ask_edge -= 1
            if current_pos < -50: bid_edge -= 1

            if current_pos < LIMIT:
                orders.append(Order(product, int(fair_value - bid_edge), LIMIT - current_pos))
            if current_pos > -LIMIT:
                orders.append(Order(product, int(fair_value + ask_edge), -LIMIT - current_pos))

            result[product] = orders

        return result, 0, json.dumps(trader_data)