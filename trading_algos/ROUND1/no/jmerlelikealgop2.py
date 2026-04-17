"""
ASH_COATED_OSMIUM — jmerle-style rolling market maker (Prosperity 2 pattern).

Port of MarketMakingStrategy from:
  https://github.com/jmerle/imc-prosperity/blob/master/src/strategies/hybrid.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"

_TD_PRICES = "jm_prices"
_TD_SPREADS = "jm_spreads"

PARAMS: Dict[str, Any] = {
    "boot_len": 10,
    "max_history": 4000,
    "common_fraction": 1.0 / 3.0,
}


class Trader:
    LIMIT: Dict[str, int] = {ASH_COATED_OSMIUM: 80}

    def __init__(self) -> None:
        self.params = dict(PARAMS)

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

        prices: List[int] = [int(x) for x in td.get(_TD_PRICES, [])]
        spreads: List[int] = [int(x) for x in td.get(_TD_SPREADS, [])]

        boot = int(self.params["boot_len"])
        cap = int(self.params["max_history"])
        frac = float(self.params["common_fraction"])

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            if product != ASH_COATED_OSMIUM:
                result[product] = []
                continue

            od = state.order_depths[product]
            if not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = (best_bid + best_ask) // 2
            sprd = best_ask - best_bid

            prices.append(mid_price)
            spreads.append(sprd)
            if len(prices) > cap:
                prices = prices[-cap:]
            if len(spreads) > cap:
                spreads = spreads[-cap:]

            if len(prices) < boot:
                result[product] = []
                continue

            ctr = Counter(prices)
            common_price, common_count = ctr.most_common(1)[0]
            if common_count > len(prices) * frac:
                true_value = common_price
            else:
                true_value = round(statistics.mean(prices[-boot:]))

            spread_med = round(statistics.median(spreads[-boot:]))
            delta = max(1, spread_med // 2 - 1)

            pos = state.position.get(product, 0)
            lim = self.LIMIT[product]
            to_buy = lim - pos
            to_sell = lim + pos

            orders: List[Order] = []
            if to_buy > 0:
                orders.append(Order(product, true_value - delta, to_buy))
            if to_sell > 0:
                orders.append(Order(product, true_value + delta, -to_sell))

            result[product] = orders

        td[_TD_PRICES] = prices
        td[_TD_SPREADS] = spreads
        new_td = json.dumps(td, separators=(",", ":"))
        return result, conversions, new_td