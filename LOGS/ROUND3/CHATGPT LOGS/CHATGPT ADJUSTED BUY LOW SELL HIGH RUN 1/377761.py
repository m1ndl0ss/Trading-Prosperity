"""
HYDROGEL_PACK — buy low / sell high vs a slow fair (EMA of micro mid).

Idea: treat a smoothed mid as fair value. When the book trades *cheap* (mid well
below EMA), lean into buys (lift + bid). When *rich* (mid well above EMA), lean
into sells (hit + ask). Small deviations → light two-sided quotes to stay engaged.

Separate from `hydrogel_only.py` (regime MM + bear skew). Tune `PARAMS` for tape.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"
_TD_EMA = "ema"
_TD_MID = "m_prev"

#1       +28701.00  adverse_volume=12  aggr_extra=5.896  band=9.589  clear_width=2  disregard_edge=1.538  ema_alpha=0.0283  join_edge=4.781  mm_edge=1.981  order_size=9  prevent_adverse=True  soft_position_fract=0.4796  take_width=1.571
PARAMS: Dict[str, Any] = {
    "ema_alpha": 0.0283,
    # |mid - ema| must exceed this before we bias directionally (price units).
    "band": 9.589,
    # Extra edge beyond `band` before aggressive lift / hit (pick off stale quotes).
    "aggr_extra": 5.896,
    "take_width": 1.571,
    "mm_edge": 1.981,
    "join_edge": 4.781,
    "clear_width": 2,
    "adverse_volume": 12,
    "prevent_adverse": True,
    "disregard_edge": 1.538,
    "order_size": 9,
    "soft_position_fract": 0.4796,
}


def _clone_depth(d: OrderDepth) -> OrderDepth:
    n = OrderDepth()
    n.buy_orders = dict(d.buy_orders)
    n.sell_orders = dict(d.sell_orders)
    return n


def _soft_limit(lim: int, fract: float) -> int:
    return max(1, int(round(lim * fract)))


class Trader:
    LIMIT: Dict[str, int] = {HYDROGEL_PACK: 200}

    def __init__(self) -> None:
        self.p = dict(PARAMS)
        self.params = self.p

    def _micro_mid(self, order_depth: OrderDepth, td: Dict[str, Any]) -> Optional[float]:
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
            prev = td.get(_TD_MID)
            if prev is None:
                return (best_ask + best_bid) / 2.0
            return float(prev)
        return (mm_ask + mm_bid) / 2.0

    def _update_ema(self, mid: float, td: Dict[str, Any]) -> float:
        a = float(self.p["ema_alpha"])
        ema = float(td.get(_TD_EMA, mid))
        ema = (1.0 - a) * ema + a * mid
        td[_TD_EMA] = ema
        td[_TD_MID] = mid
        return ema

    def _take(
        self,
        fair: float,
        take_buy: float,
        take_sell: float,
        orders: List[Order],
        d: OrderDepth,
        pos: int,
        buy_v: int,
        sell_v: int,
    ) -> Tuple[int, int]:
        lim = self.LIMIT[HYDROGEL_PACK]
        adv = int(self.p["adverse_volume"])
        prev = bool(self.p["prevent_adverse"])

        if d.sell_orders:
            best_ask = min(d.sell_orders.keys())
            amt = -d.sell_orders[best_ask]
            if not prev or abs(amt) <= adv:
                if best_ask <= fair - take_buy:
                    q = min(int(amt), lim - pos - buy_v)
                    if q > 0:
                        orders.append(Order(HYDROGEL_PACK, best_ask, q))
                        buy_v += q
                        d.sell_orders[best_ask] += q
                        if d.sell_orders[best_ask] == 0:
                            del d.sell_orders[best_ask]

        if d.buy_orders:
            best_bid = max(d.buy_orders.keys())
            amt = d.buy_orders[best_bid]
            if not prev or abs(amt) <= adv:
                if best_bid >= fair + take_sell:
                    q = min(int(amt), lim + pos - sell_v)
                    if q > 0:
                        orders.append(Order(HYDROGEL_PACK, best_bid, -q))
                        sell_v += q
                        d.buy_orders[best_bid] -= q
                        if d.buy_orders[best_bid] == 0:
                            del d.buy_orders[best_bid]

        return buy_v, sell_v

    def _clear(
        self,
        fair: float,
        width: int,
        orders: List[Order],
        d: OrderDepth,
        pos: int,
        buy_v: int,
        sell_v: int,
    ) -> Tuple[int, int]:
        lim = self.LIMIT[HYDROGEL_PACK]
        pa = pos + buy_v - sell_v
        fb = int(round(fair - width))
        fa = int(round(fair + width))
        bq = lim - pos - buy_v
        sq = lim + pos - sell_v
        if pa > 0:
            cq = min(
                sum(v for p, v in d.buy_orders.items() if p >= fa),
                pa,
            )
            s = min(sq, cq)
            if s > 0:
                orders.append(Order(HYDROGEL_PACK, fa, -abs(s)))
                sell_v += abs(s)
        if pa < 0:
            cq = min(
                sum(abs(v) for p, v in d.sell_orders.items() if p <= fb),
                abs(pa),
            )
            s = min(bq, cq)
            if s > 0:
                orders.append(Order(HYDROGEL_PACK, fb, abs(s)))
                buy_v += abs(s)
        return buy_v, sell_v

    def _quote_pair(
        self,
        fair: float,
        d: OrderDepth,
        pos: int,
        buy_v: int,
        sell_v: int,
        dev: float,
        soft: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        de = float(self.p["disregard_edge"])
        je = float(self.p["join_edge"])
        edge = float(self.p["mm_edge"])
        lim = self.LIMIT[HYDROGEL_PACK]
        size = max(1, int(self.p["order_size"]))

        asks_ok = [p for p in d.sell_orders.keys() if p > fair + de]
        bids_ok = [p for p in d.buy_orders.keys() if p < fair - de]
        baa = min(asks_ok) if asks_ok else None
        bbb = max(bids_ok) if bids_ok else None
        ask = round(fair + edge)
        if baa is not None:
            ask = baa if abs(baa - fair) <= je else baa - 1
        bid = round(fair - edge)
        if bbb is not None:
            bid = bbb if abs(fair - bbb) <= je else bbb + 1

        band = float(self.p["band"])
        if dev < -band:
            ask -= 1
        elif dev > band:
            bid += 1
        if pos > soft:
            ask -= 1
        elif pos < -soft:
            bid += 1

        bq = lim - pos - buy_v
        sq = lim + pos - sell_v
        if bq > 0:
            q = min(size, bq)
            if dev > band:
                q = max(1, q // 2)
            orders.append(Order(HYDROGEL_PACK, int(bid), q))
        if sq > 0:
            q = min(size, sq)
            if dev < -band:
                q = max(1, q // 2)
            orders.append(Order(HYDROGEL_PACK, int(ask), -q))

        return orders, buy_v, sell_v

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
            mid = self._micro_mid(d0, td)
            if mid is None:
                result[product] = []
                continue

            ema = self._update_ema(mid, td)
            dev = mid - ema
            fair = ema

            band = float(self.p["band"])
            ag = float(self.p["aggr_extra"])
            tw = float(self.p["take_width"])

            take_buy = tw
            take_sell = tw
            if dev < -band:
                take_buy = tw + ag + min(abs(dev + band), 12.0) * 0.15
            elif dev > band:
                take_sell = tw + ag + min(dev - band, 12.0) * 0.15

            d = _clone_depth(d0)
            buy_v, sell_v = 0, 0
            take_list: List[Order] = []
            buy_v, sell_v = self._take(
                fair, take_buy, take_sell, take_list, d, pos, buy_v, sell_v
            )

            cw = int(self.p["clear_width"])
            if abs(pos) > soft:
                cw += 1
            clear_list: List[Order] = []
            buy_v, sell_v = self._clear(
                fair, cw, clear_list, d, pos, buy_v, sell_v
            )

            make_orders, _, _ = self._quote_pair(
                fair, d, pos, buy_v, sell_v, dev, soft
            )

            result[product] = take_list + clear_list + make_orders

        return result, conversions, json.dumps(td, separators=(",", ":"))