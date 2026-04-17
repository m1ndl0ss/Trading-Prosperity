from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
_TD_KEY = "mm_mid_prev"

# Starfruit-equivalent params from round_1_v6.py (tune for osmium if needed)
PARAMS: Dict[str, Any] = {
    "take_width": 1.8,
    "clear_width": 1,
    "prevent_adverse": False,
    "adverse_volume": 15,
    "reversion_beta": -0.2376,
    "disregard_edge": 1,
    "join_edge": 1, #2
    "default_edge": 2,
}


def _clone_depth(d: OrderDepth) -> OrderDepth:
    n = OrderDepth()
    n.buy_orders = dict(d.buy_orders)
    n.sell_orders = dict(d.sell_orders)
    return n


class Trader:
    """Osmium-only. LIMIT matches other ROUND1 algos."""

    LIMIT: Dict[str, int] = {ASH_COATED_OSMIUM: 80}

    def __init__(self) -> None:
        self.params = dict(PARAMS)

    def _fair_value(self, order_depth: OrderDepth, td: Dict[str, Any]) -> float | None:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        adv = int(self.params["adverse_volume"])

        filtered_ask = [
            p
            for p in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[p]) >= adv
        ]
        filtered_bid = [
            p
            for p in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[p]) >= adv
        ]
        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None

        if mm_ask is None or mm_bid is None:
            prev = td.get(_TD_KEY)
            if prev is None:
                mmmid_price = (best_ask + best_bid) / 2.0
            else:
                mmmid_price = float(prev)
        else:
            mmmid_price = (mm_ask + mm_bid) / 2.0

        beta = float(self.params["reversion_beta"])
        prev_mid = td.get(_TD_KEY)
        if prev_mid is not None:
            last_price = float(prev_mid)
            if last_price != 0:
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * beta
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
        else:
            fair = mmmid_price

        td[_TD_KEY] = mmmid_price
        return fair

    def _take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool,
        adverse_volume: int,
    ) -> Tuple[int, int]:
        lim = self.LIMIT[product]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(int(best_ask_amount), lim - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(int(best_bid_amount), lim + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def _market_make(
        self,
        product: str,
        orders: List[Order],
        bid: float,
        ask: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, int(round(bid)), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(round(ask)), -sell_quantity))

        return buy_order_volume, sell_order_volume

    def _clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = int(round(fair_value - width))
        fair_for_ask = int(round(fair_value + width))

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def _take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_v = sell_v = 0
        buy_v, sell_v = self._take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_v,
            sell_v,
            bool(self.params["prevent_adverse"]),
            int(self.params["adverse_volume"]),
        )
        return orders, buy_v, sell_v

    def _clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self._clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def _make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = True,
        soft_position_limit: int = 40,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        asks_above_fair = [
            p for p in order_depth.sell_orders.keys() if p > fair_value + disregard_edge
        ]
        bids_below_fair = [
            p for p in order_depth.buy_orders.keys() if p < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        self._market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

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

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            if product != ASH_COATED_OSMIUM:
                result[product] = []
                continue

            depth_orig = state.order_depths[product]
            pos = state.position.get(ASH_COATED_OSMIUM, 0)
            fair = self._fair_value(depth_orig, td)
            if fair is None:
                result[product] = []
                continue

            depth = _clone_depth(depth_orig)

            take_orders, buy_v, sell_v = self._take_orders(
                ASH_COATED_OSMIUM,
                depth,
                fair,
                float(self.params["take_width"]),
                pos,
            )
            clear_orders, buy_v, sell_v = self._clear_orders(
                ASH_COATED_OSMIUM,
                depth,
                fair,
                int(self.params["clear_width"]),
                pos,
                buy_v,
                sell_v,
            )
            make_orders, _, _ = self._make_orders(
                ASH_COATED_OSMIUM,
                depth,
                fair,
                pos,
                buy_v,
                sell_v,
                float(self.params["disregard_edge"]),
                float(self.params["join_edge"]),
                float(self.params["default_edge"]),
            )

            result[product] = take_orders + clear_orders + make_orders

        new_td = json.dumps(td, separators=(",", ":"))
        return result, conversions, new_td