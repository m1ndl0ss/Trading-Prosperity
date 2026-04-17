"""
ASH_COATED_OSMIUM — v2: fixes for the actual osmium book structure.

Key differences from starfruit_style_osmium.py
-----------------------------------------------
1. Reversion is additive on tick-level moves (not multiplicative on returns).
   The data shows lag-1 autocorr = -0.50; the old formula applied beta to a
   0.0005 return which moved fair by ~1 tick. The new formula applies beta
   directly to the raw tick move: fair = mmmid + move * beta, where beta ≈ -0.5.

2. take_width is set to ~7 (matching half the 16-tick spread) so the take
   logic can actually fire when the book moves toward our fair. Previously it
   was set to ~1, meaning best_ask (at fair+8) could never satisfy best_ask <=
   fair - 1, so the take block was dead code every tick.

3. clear_width is set to ~8 so the clear logic looks for bids >= fair+8 (the
   actual best bid location). Previously it searched for bids >= fair+1, which
   never matched since bids are always ~8 ticks below fair.

4. manage_position=True with soft_position_limit (default 20) skews make
   quotes 1 tick against the heavy side to prevent end-of-day inventory buildup.

5. adverse_volume lowered to 5: level-1 volumes average 14 but the p10 is 10.
   At adverse_volume=10, ~4% of ticks silently fall back to level-2 prices
   (which sit 2.5 ticks further from fair), injecting noise into the fair
   estimate. At 5, this drops to ~2%.

6. join_edge raised to 8: with the best bid/ask at ~fair±8 and the old
   join_edge=2, the bot always pennied (best±1) rather than joining the best
   level. Raising to 8 means the bot joins when the existing best is within 8
   ticks of fair — i.e., almost always — keeping it at the best price queue.

PARAMS are all tunable; the MC preset "osmium_v2" searches their ranges.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
_TD_KEY = "mm_mid_prev"

PARAMS: Dict[str, Any] = {
    # ── fair-value ────────────────────────────────────────────────────────────
    "adverse_volume": 10,           # min level-1 size to count as MM quote
    "reversion_beta": -0.3067,       # applied additively: fair = mmmid + move*beta
    # ── take (cross spread on misprice) ───────────────────────────────────────
    "take_width": 7.162,             # take if best_ask <= fair-7 or best_bid >= fair+7
    "prevent_adverse": True,      # skip take when size > adverse_volume
    # ── clear (reduce inventory near fair) ───────────────────────────────────
    "clear_width": 6,              # clear sells at fair+8, clear buys at fair-8
    # ── make (passive quotes) ─────────────────────────────────────────────────
    "disregard_edge": 1,           # ignore existing quotes within 1 tick of fair
    "join_edge": 6,                # join existing quote if within 8 ticks of fair
    "default_edge": 9,             # fallback edge when no existing quote to join
    # ── inventory management ──────────────────────────────────────────────────
    "soft_position_limit": 34,     # skew make quote 1 tick when |pos| > this
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

    # ── Fair value ──────────────────────────────────────────────────────────
    def _fair_value(self, order_depth: OrderDepth, td: Dict[str, Any]) -> float | None:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        adv = int(self.params["adverse_volume"])

        filtered_ask = [
            p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= adv
        ]
        filtered_bid = [
            p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= adv
        ]
        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None

        if mm_ask is None or mm_bid is None:
            prev = td.get(_TD_KEY)
            mmmid_price = float(prev) if prev is not None else (best_ask + best_bid) / 2.0
        else:
            mmmid_price = (mm_ask + mm_bid) / 2.0

        beta = float(self.params["reversion_beta"])
        prev_mid = td.get(_TD_KEY)
        if prev_mid is not None:
            # Additive tick-level reversion: moves are ~1-5 ticks, not tiny returns.
            # old formula: fair = mmmid + mmmid * (return * beta)  → ~0.2 tick effect
            # new formula: fair = mmmid + move * beta              → 0.5-2.5 tick effect
            move = mmmid_price - float(prev_mid)
            fair = mmmid_price + move * beta
        else:
            fair = mmmid_price

        td[_TD_KEY] = mmmid_price
        return fair

    # ── Take ────────────────────────────────────────────────────────────────
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
            if not prevent_adverse or best_ask_amount <= adverse_volume:
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
            if not prevent_adverse or best_bid_amount <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(int(best_bid_amount), lim + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    # ── Clear ───────────────────────────────────────────────────────────────
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
        # With width ~8, fair_for_ask ≈ best_ask and fair_for_bid ≈ best_bid.
        # Clear thus exits near the existing best price rather than crossing mid.
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

    # ── Make ────────────────────────────────────────────────────────────────
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
        soft_position_limit: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []

        asks_above_fair = [
            p for p in order_depth.sell_orders if p > fair_value + disregard_edge
        ]
        bids_below_fair = [
            p for p in order_depth.buy_orders if p < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair      # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair      # join
            else:
                bid = best_bid_below_fair + 1  # penny

        # Skew quotes 1 tick to help unwind heavy inventory.
        if position > soft_position_limit:
            ask -= 1
        elif position < -soft_position_limit:
            bid += 1

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, int(round(bid)), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, int(round(ask)), -sell_quantity))

        return orders, buy_order_volume, sell_order_volume

    # ── Helpers ─────────────────────────────────────────────────────────────
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

    # ── Main loop ────────────────────────────────────────────────────────────
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
                int(self.params["soft_position_limit"]),
            )

            result[product] = take_orders + clear_orders + make_orders

        new_td = json.dumps(td, separators=(",", ":"))
        return result, conversions, new_td
