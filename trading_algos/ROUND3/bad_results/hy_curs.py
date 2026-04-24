"""
HYDROGEL_PACK only. Pipeline: estimate fair from *Wall Mid* (deep-liquidity bid/ask),
then take -> clear -> passive quotes (see Frankfurt Hedgehogs Prosperity 3 write-up).
Also borrows: halflife EMA, short trend extrap, undercut (Stanford Cardinal 2021).

- https://github.com/TimoDiehm/imc-prosperity-3 (2nd place Prosperity 3): Wall Mid FAQ,
  Kelp-style take then overbid/undercut, inventory flattening at fair.
- https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"
_TD_EMA = "ema"
_TD_MID = "m_prev"
_MIDS = "mids"  # last few mids for short extrap (banana-style cache)

PARAMS: Dict[str, Any] = {
    # EMA: alpha = 1 - 2^(-1/half_life) in ticks (like Cardinal's halflife_price)
    "ema_halflife": 14.0,
    "band": 6.5,
    "aggr_extra": 4.2,
    "take_width": 2.5,
    "dev_slope": 0.22,
    "inv_skew": 0.028,
    "extrap_w": 0.22,
    "mm_edge": 1.0,
    "join_edge": 2.609,
    "undercut": 1,
    "clear_width": 2,
    "adverse_volume": 22,
    # Set to e.g. 20 to use cumulative-depth walls (~10% of 200) like Hedgehogs'
    # book walk; None = per-level `adverse_volume` filter (default here).
    "wall_cum_volume": None,
    "prevent_adverse": True,
    "disregard_edge": 0.6999,
    "order_size": 19,
    "soft_position_fract": 0.4993,
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

    def _wall_mid_cumulative(
        self, order_depth: OrderDepth, target: int, td: Dict[str, Any]
    ) -> Optional[float]:
        """Hedgehogs-style: first bid/ask prices where cum depth reaches target."""
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        c = 0
        wall_bid: Optional[int] = None
        for p in sorted(order_depth.buy_orders.keys(), reverse=True):
            c += int(order_depth.buy_orders[p])
            if c >= target:
                wall_bid = p
                break
        c = 0
        wall_ask: Optional[int] = None
        for p in sorted(order_depth.sell_orders.keys()):
            c += int(abs(order_depth.sell_orders[p]))
            if c >= target:
                wall_ask = p
                break
        if wall_bid is not None and wall_ask is not None:
            return (float(wall_ask) + float(wall_bid)) / 2.0
        prev = td.get(_TD_MID)
        if prev is None:
            return (best_ask + best_bid) / 2.0
        return float(prev)

    def _wall_mid_level(self, order_depth: OrderDepth, td: Dict[str, Any]) -> Optional[float]:
        """Wall mid = mean of first bid/ask *levels* with size >= adverse_volume (repo style)."""
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

    def _wall_mid(self, order_depth: OrderDepth, td: Dict[str, Any]) -> Optional[float]:
        wcv = self.p.get("wall_cum_volume")
        if wcv is not None and int(wcv) > 0:
            return self._wall_mid_cumulative(order_depth, int(wcv), td)
        return self._wall_mid_level(order_depth, td)

    def _alpha_from_halflife(self) -> float:
        hl = max(1.0, float(self.p["ema_halflife"]))
        return 1.0 - math.exp(-math.log(2.0) / hl)

    def _update_ema(self, mid: float, td: Dict[str, Any]) -> float:
        a = self._alpha_from_halflife()
        ema = float(td.get(_TD_EMA, mid))
        ema = (1.0 - a) * ema + a * mid
        td[_TD_EMA] = ema
        td[_TD_MID] = mid
        return ema

    def _short_extrap_fair(self, mid: float, td: Dict[str, Any]) -> float:
        """1-step trend from last mids (simplified banana regression)."""
        hist: List[float] = list(td.get(_MIDS) or [])
        hist.append(mid)
        td[_MIDS] = hist[-6:]
        if len(hist) >= 2:
            d = float(hist[-1] - hist[-2])
            return float(mid + d * 0.6)
        return float(mid)

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
        uc = int(self.p.get("undercut", 0))
        ask = round(fair + edge)
        if baa is not None:
            ask = baa if abs(baa - fair) <= je else baa - 1
            ask = min(ask, baa - uc)  # undercut best ask (pearls/bananas style)
        bid = round(fair - edge)
        if bbb is not None:
            bid = bbb if abs(fair - bbb) <= je else bbb + 1
            bid = max(bid, bbb + uc)  # improve vs best bid
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
        lim_p = self.LIMIT[HYDROGEL_PACK]
        soft = _soft_limit(lim_p, float(self.p["soft_position_fract"]))

        for product in state.order_depths:
            if product != HYDROGEL_PACK:
                result[product] = []
                continue
            d0 = state.order_depths[product]
            pos = state.position.get(HYDROGEL_PACK, 0)
            mid = self._wall_mid(d0, td)
            if mid is None:
                result[product] = []
                continue
            ema = self._update_ema(mid, td)
            extrap = self._short_extrap_fair(mid, td)
            ew = float(self.p["extrap_w"])
            sk = float(self.p["inv_skew"])
            base = (1.0 - ew) * ema + ew * extrap
            fair = base - sk * pos
            dev = mid - ema
            band = float(self.p["band"])
            ag = float(self.p["aggr_extra"])
            tw = float(self.p["take_width"])
            sl = float(self.p["dev_slope"])
            take_buy = tw
            take_sell = tw
            if dev < -band:
                take_buy = tw + ag + min(abs(dev + band), 12.0) * sl
            elif dev > band:
                take_sell = tw + ag + min(dev - band, 12.0) * sl
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
