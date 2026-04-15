import json
from datamodel import Order, ProsperityEncoder, Symbol, TradingState, OrderDepth
from typing import Any, List, Dict

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]]) -> None:
        # 1. Official Platform Style JSON Log
        # The visualizer needs this specific JSON structure for the 'Sandbox' section
        print(json.dumps({
            "state": {
                "t": state.timestamp,
                "p": state.position
            },
            "orders": [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr],
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))
        self.logs = ""
        
        # 2. Official Platform Style CSV Table (Activities log)
        # The visualizer's code scans for THIS EXACT STRING below
        if state.timestamp == 0:
            print("\nActivities log:")
            # The visualizer maps its charts to THESE exact column names in THIS order
            print("day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss")
        
        for product in state.order_depths:
            d = state.order_depths[product]
            bids = sorted(d.buy_orders.items(), key=lambda x: x[0], reverse=True)
            asks = sorted(d.sell_orders.items(), key=lambda x: x[0])
            
            # Formatting the data row for each product at each timestamp
            # prosperity4bt round is used here as a placeholder for 'day'
            print(f"1;{state.timestamp};{product};"
                  f"{bids[0][0] if len(bids)>0 else ''};{bids[0][1] if len(bids)>0 else ''};"
                  f"{bids[1][0] if len(bids)>1 else ''};{bids[1][1] if len(bids)>1 else ''};"
                  f"{bids[2][0] if len(bids)>2 else ''};{bids[2][1] if len(bids)>2 else ''};"
                  f"{asks[0][0] if len(asks)>0 else ''};{asks[0][1] if len(asks)>0 else ''};"
                  f"{asks[1][0] if len(asks)>1 else ''};{asks[1][1] if len(asks)>1 else ''};"
                  f"{asks[2][0] if len(asks)>2 else ''};{asks[2][1] if len(asks)>2 else ''};"
                  f"0;0") # placeholders for midprice and pnl

logger = Logger()

class Trader:
    def run(self, state: TradingState):
        result = {}
        # Simple test strategy to ensure trades are generated
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                orders.append(Order(product, best_ask, 1))
            result[product] = orders

        # This call creates the visualizer-ready log content
        logger.flush(state, result)
        return result, 0, ""