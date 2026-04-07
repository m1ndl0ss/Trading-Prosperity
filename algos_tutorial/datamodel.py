from typing import Dict, List, Any

# Basic Types
Symbol = str

class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __repr__(self) -> str:
        return f"Order({self.symbol}, {self.price}, {self.quantity})"

class OrderDepth:
    def __init__(self, buy_orders: Dict[int, int] = None, sell_orders: Dict[int, int] = None):
        # Maps price to quantity
        self.buy_orders = buy_orders or {}
        self.sell_orders = sell_orders or {}

class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: str = "", seller: str = "", timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

class TradingState:
    def __init__(self,
                 traderData: str,
                 timestamp: int,
                 listings: Dict[Symbol, Any],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Symbol, int],
                 observations: Dict[Symbol, int]):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations