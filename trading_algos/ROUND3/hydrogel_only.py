"""
HYDROGEL_PACK only — compact market making aimed at *robust* behaviour (few knobs).

Why the old stack under-performed live: many interacting parameters (reversion, two
EMAs, top-fade, skews) amount to **fitting noise** on the CSV.  This version uses a
single slow EMA + session peak, a **deadband** before we call the regime "bear", and
only then: lower *fair*, asymmetric takes, smaller buy quotes, and an extra sell tick
when long.

Tuning surface (see `PARAMS`): 10 numbers.  Book / join / adverse / clear-boost are fixed
in `_BOOK` (not swept).  Drawdown-from-peak nudge uses small **constants** in code so
they are not another grid dimension.

**Live vs backtest** will still differ: tape replay, partial fills, adversaries, and
path dependence — aim for stable rules, not max CSV PnL.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"
_TD_MID = "m_prev"
_TD_EMA = "ema"
_TD_PEAK = "peak"

# Book behaviour: fixed, not part of the tuning grid.
_BOOK: Dict[str, Any] = {
    "adverse_volume": 19,
    "join_edge": 3,
    "disregard_edge": 1,
    "default_edge": 0,
    "prevent_adverse": True,
}

# Only these keys are meant to be swept / tuned.
PARAMS: Dict[str, Any] = {
    "take_width": 3.0,
    "clear_width": 1,
    # Slow EMA on micro mid; ~0.02 ≈ 50-tick e-folding on typical tape density.
    "ema_alpha": 0.02,
    # "Bear" only when mid is *meaningfully* below EMA (not every tick under the line).
    "bear_ema_deadband": 2.5,
    # When mid < EMA - deadband, fair -= min(max_bear, (EMA - mid) * k) — pull quotes with trend.
    "bear_nudge_k": 0.28,
    "bear_nudge_max": 7.0,
    # In bear: add to min edge before we lift; subtract before we hit a bid to sell.
    "bear_bid_lift": 0.55,
    "bear_ask_tighten": 0.35,
    # In bear, scale down *buy* quote size (sell size unchanged).
    "bear_buy_scale": 0.58,
    "soft_position_fract": 0.4,
}
_SOFT_CLEAR_EXTRA = 1  # extra clear_width when |pos| beyond soft band (fixed)


def _clone_depth(d: OrderDepth) -> OrderDepth:
    n = OrderDepth()
    n.buy_orders = dict(d.buy_orders)
    n.sell_orders = dict(d.sell_orders)
    return n


def _soft_limit(lim: int, fract: float) -> int:
    return max(1, int(round(lim * fract)))


class Trader:
    """Hydrogel only; position cap 200 (Round 3)."""

    LIMIT: Dict[str, int] = {HYDROGEL_PACK: 200}

    def __init__(self) -> None:
        self.p = {**_BOOK, **PARAMS}
        self.params = self.p  # alias for notebooks / external tools

    def _microstructure_mid(
        self, order_depth: OrderDepth, td: Dict[str, Any]
    ) -> Optional[float]:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        adv = int(self.p["adverse_volume"])
        f_ask = [
            p
            for p in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[p]) >= adv
        ]
        f_bid = [
            p
            for p in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[p]) >= adv
        ]
        mm_ask = min(f_ask) if f_ask else None
        mm_bid = max(f_bid) if f_bid else None
        if mm_ask is None or mm_bid is None:
            pr = td.get(_TD_MID)
            if pr is None:
                return (best_ask + best_bid) / 2.0
            return float(pr)
        return (mm_ask + mm_bid) / 2.0

    def _fair_bear(
        self, mmmid: float, td: Dict[str, Any]
    ) -> Tuple[float, bool, float]:
        """Return (fair, is_bear, ema) after updating EMA and session peak in td."""
        a = float(self.p["ema_alpha"])
        ema = float(td.get(_TD_EMA, mmmid))
        ema = (1.0 - a) * ema + a * mmmid
        td[_TD_EMA] = ema
        peak = max(float(td.get(_TD_PEAK, mmmid)), mmmid)
        td[_TD_PEAK] = peak

        db = float(self.p["bear_ema_deadband"])
        is_bear = mmmid < ema - db

        fair = mmmid
        if is_bear:
            k = float(self.p["bear_nudge_k"])
            cap = float(self.p["bear_nudge_max"])
            fair -= min(cap, max(0.0, (ema - mmmid) * k))
        if peak > mmmid + 1.0:
            # Fixed: extra fair cut when we have come off the session high (one leg).
            fair -= min(2.5, max(0.0, (peak - mmmid) * 0.08))

        td[_TD_MID] = mmmid
        return fair, is_bear, ema

    def _take_best_orders(
        self,
        product: str,
        fair: float,
        take_buy: float,
        take_sell: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        lim = self.LIMIT[product]
        adv = int(self.p["adverse_volume"])
        prev = bool(self.p["prevent_adverse"])

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prev or abs(best_ask_amount) <= adv:
                if best_ask <= fair - take_buy:
                    q = min(int(best_ask_amount), lim - position)
                    if q > 0:
                        orders.append(Order(product, best_ask, q))
                        buy_order_volume += q
                        order_depth.sell_orders[best_ask] += q
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prev or abs(best_bid_amount) <= adv:
                if best_bid >= fair + take_sell:
                    q = min(int(best_bid_amount), lim + position)
                    if q > 0:
                        orders.append(Order(product, best_bid, -q))
                        sell_order_volume += q
                        order_depth.buy_orders[best_bid] -= q
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
        buy_scale: float,
    ) -> Tuple[int, int]:
        lim = self.LIMIT[product]
        bq = lim - (position + buy_order_volume)
        if bq > 0:
            bq = int(max(1, round(bq * buy_scale)))
            orders.append(Order(product, int(round(bid)), bq))
        sell_q = lim + (position - sell_order_volume)
        if sell_q > 0:
            orders.append(Order(product, int(round(ask)), -sell_q))
        return buy_order_volume, sell_order_volume

    def _clear_position_order(
        self,
        product: str,
        fair: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        pa = position + buy_order_volume - sell_order_volume
        w = width
        fair_bid = int(round(fair - w))
        fair_ask = int(round(fair + w))
        bq = self.LIMIT[product] - (position + buy_order_volume)
        sq = self.LIMIT[product] + (position - sell_order_volume)
        if pa > 0:
            cq = min(
                sum(
                    v
                    for p, v in order_depth.buy_orders.items()
                    if p >= fair_ask
                ),
                pa,
            )
            s = min(sq, cq)
            if s > 0:
                orders.append(Order(product, fair_ask, -abs(s)))
                sell_order_volume += abs(s)
        if pa < 0:
            cq = min(
                sum(abs(v) for p, v in order_depth.sell_orders.items() if p <= fair_bid),
                abs(pa),
            )
            s = min(bq, cq)
            if s > 0:
                orders.append(Order(product, fair_bid, abs(s)))
                buy_order_volume += abs(s)
        return buy_order_volume, sell_order_volume

    def _make_join_quote(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        soft_limit: int,
        buy_scale: float,
        is_bear: bool,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        de = float(self.p["disregard_edge"])
        je = float(self.p["join_edge"])
        dde = float(self.p["default_edge"])
        asks_ok = [p for p in order_depth.sell_orders.keys() if p > fair + de]
        bids_ok = [p for p in order_depth.buy_orders.keys() if p < fair - de]
        baa = min(asks_ok) if asks_ok else None
        bbb = max(bids_ok) if bids_ok else None
        ask = round(fair + dde)
        if baa is not None:
            ask = baa if abs(baa - fair) <= je else baa - 1
        bid = round(fair - dde)
        if bbb is not None:
            bid = bbb if abs(fair - bbb) <= je else bbb + 1
        if position > soft_limit:
            ask -= 1
        elif position < -soft_limit:
            bid += 1
        if is_bear and position > 0:
            ask -= 1
        self._market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume, buy_scale
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
        lim = self.LIMIT[HYDROGEL_PACK]
        soft = _soft_limit(lim, float(self.p["soft_position_fract"]))

        for product in state.order_depths:
            if product != HYDROGEL_PACK:
                result[product] = []
                continue
            d0 = state.order_depths[product]
            pos = state.position.get(HYDROGEL_PACK, 0)
            m = self._microstructure_mid(d0, td)
            if m is None:
                result[product] = []
                continue
            fair, is_bear, _ema = self._fair_bear(m, td)
            d = _clone_depth(d0)

            tw = float(self.p["take_width"])
            bl = float(self.p["bear_bid_lift"])
            st = float(self.p["bear_ask_tighten"])
            take_buy = tw + (bl if is_bear else 0.0)
            take_sell = max(0.6, tw - (st if (is_bear and pos > 0) else 0.0))

            take_list: List[Order] = []
            buy_v, sell_v = self._take_best_orders(
                HYDROGEL_PACK,
                fair,
                take_buy,
                take_sell,
                take_list,
                d,
                pos,
                0,
                0,
            )
            take_orders = take_list

            cw0 = int(self.p["clear_width"])
            cwx = _SOFT_CLEAR_EXTRA
            cw = cw0
            if abs(pos) > soft:
                cw += cwx
            clear_list: List[Order] = []
            buy_v2, sell_v2 = self._clear_position_order(
                HYDROGEL_PACK,
                fair,
                cw,
                clear_list,
                d,
                pos,
                buy_v,
                sell_v,
            )
            clear_orders = clear_list

            bsc = 1.0 if not is_bear else float(self.p["bear_buy_scale"])
            make_orders, _, _ = self._make_join_quote(
                HYDROGEL_PACK,
                d,
                fair,
                pos,
                buy_v2,
                sell_v2,
                soft,
                bsc,
                is_bear,
            )
            result[product] = take_orders + clear_orders + make_orders

        return result, conversions, json.dumps(td, separators=(",", ":"))
