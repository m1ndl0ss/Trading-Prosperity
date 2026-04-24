"""
HYDROGEL_PACK only — jmerle IMC Prosperity **2** Round 2 *market-making* pattern.

Source idea:
  https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round2.py
  (`MarketMakingStrategy` + `StarfruitStrategy` true value = popular buy/sell mid)

In Prosperity 2 Round 2, the big edge on ORCHIDS was **conversions** (flatten, then
short to limit at a price that stays profitable after reconverting). This repo’s
`datamodel` (Prosperity 3) does not expose `conversionObservations` on products like
the P2 engine, so **that** half of jmerle’s `round2.py` is not ported here.

What *is* ported: the Starfruit-style MM — lift asks / hit bids when mispriced vs
`true_value`, join one tick inside the *popular* (max-size) levels for residual size,
and soft/hard **liquidation** when stuck at the position limit (deque window).
Optional **Amethysts** mode: fixed `true_value` (default 10_000) instead of the
starfruit mid.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"
_TD_WINDOW = "jm_w"

PARAMS: Dict[str, Any] = {
    # True value: "starfruit" = popular mid (jmerle StarfruitStrategy), "amethyst" = fixed
    "fair_mode": "amethyst",
    "fixed_fair": 9000,
    "window_size": 9,
}


class Trader:
    """jmerle-style market maker for one product, Round 3 position cap."""

    LIMIT: Dict[str, int] = {HYDROGEL_PACK: 200}

    def __init__(self) -> None:
        self.params = dict(PARAMS)

    def _get_true_value(self, order_depth: OrderDepth) -> int | None:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        mode = str(self.params.get("fair_mode", "starfruit")).lower()
        if mode == "amethyst":
            return int(self.params["fixed_fair"])

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return int(round((popular_buy_price + popular_sell_price) / 2.0))

    def _run_market_making(
        self,
        state: TradingState,
        window: List[bool],
        wsize: int,
    ) -> List[Order]:
        product = HYDROGEL_PACK
        lim = self.LIMIT[product]
        order_depth = state.order_depths[product]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        true_value = self._get_true_value(order_depth)
        if true_value is None:
            return []

        buy_book = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_book = sorted(order_depth.sell_orders.items())

        position = state.position.get(product, 0)
        to_buy = lim - position
        to_sell = lim + position

        window.append(bool(abs(position) == lim))
        while len(window) > wsize:
            window.pop(0)

        soft_liquidate = (
            len(window) == wsize
            and sum(window) >= wsize / 2
            and (window[-1] if window else False)
        )
        hard_liquidate = len(window) == wsize and all(window)

        max_buy_price = true_value - 1 if position > lim * 0.5 else true_value
        min_sell_price = true_value + 1 if position < lim * -0.5 else true_value

        orders: List[Order] = []

        for price, volume in sell_book:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                if quantity > 0:
                    orders.append(Order(product, int(price), int(quantity)))
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            if quantity > 0:
                orders.append(Order(product, int(true_value), int(quantity)))
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            if quantity > 0:
                orders.append(Order(product, int(true_value) - 2, int(quantity)))
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_book, key=lambda tup: tup[1])[0]
            price = min(int(max_buy_price), int(popular_buy_price) + 1)
            orders.append(Order(product, price, to_buy))

        for price, volume in buy_book:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                if quantity > 0:
                    orders.append(Order(product, int(price), -int(quantity)))
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            if quantity > 0:
                orders.append(Order(product, int(true_value), -int(quantity)))
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            if quantity > 0:
                orders.append(Order(product, int(true_value) + 2, -int(quantity)))
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_book, key=lambda tup: tup[1])[0]
            price = max(int(min_sell_price), int(popular_sell_price) - 1)
            orders.append(Order(product, price, -to_sell))

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        wsize = int(self.params["window_size"])
        window: List[bool] = [bool(x) for x in td.get(_TD_WINDOW, [])]
        while len(window) > wsize:
            window.pop(0)

        result: Dict[str, List[Order]] = {}
        for product in state.order_depths:
            if product != HYDROGEL_PACK:
                result[product] = []
                continue
            o = self._run_market_making(state, window, wsize)
            result[product] = o

        td[_TD_WINDOW] = window[-wsize:]
        return result, conversions, json.dumps(td, separators=(",", ":"))
