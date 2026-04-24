"""
HYDROGEL_PACK only — session extrema / range-percentile strategy (not market making).

Inspiration: Frankfurt Hedgehogs on Squid Ink (Prosperity 3 Round 1) — they traded
off *daily* running min/max and extreme prints (the “Olivia at highs/lows” story):
https://github.com/TimoDiehm/imc-prosperity-3

We do not see opponent IDs here; this bot approximates the *idea* with:
- running session low / high of a robust mid
- `dist = (mid - lo) / (hi - lo)` in [0,1]
- when price sits in the bottom/top fraction of the *day’s* range, lean long / short
- slow EMA as a second filter (only buy if “cheap” vs EMA, only sell if “rich”)
- execution: aggressive takes + optional one-sided exit, no two-sided quote blanket

Respects LIMIT 200. Tune PARAMS for your tape.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"
_K_LO = "s_lo"
_K_HI = "s_hi"
_K_EMA = "ema"
_K_MID = "m_prev"
_K_N = "n"

PARAMS: Dict[str, Any] = {
    "ema_halflife": 32.0,
    # No directional trades until range >= this (price units; avoids first minutes noise)
    "min_range": 18.0,
    "burn_in": 30,
    # Bottom / top fraction of [lo,hi] to arm signals (0.12 = lower 12% of range)
    "buy_range_pct": 0.12,
    "sell_range_pct": 0.88,
    # Must also be "cheap" / "rich" vs EMA
    "ema_buffer": 3.0,
    "take_edge": 1.1,
    "clip": 35,
    "adverse_volume": 20,
    "prevent_adverse": True,
}


def _clone_depth(d: OrderDepth) -> OrderDepth:
    n = OrderDepth()
    n.buy_orders = dict(d.buy_orders)
    n.sell_orders = dict(d.sell_orders)
    return n


def _mid_simple(d: OrderDepth) -> Optional[float]:
    if not d.sell_orders or not d.buy_orders:
        return None
    return (min(d.sell_orders.keys()) + max(d.buy_orders.keys())) / 2.0


class Trader:
    LIMIT: Dict[str, int] = {HYDROGEL_PACK: 200}

    def __init__(self) -> None:
        self.p = dict(PARAMS)
        self.params = self.p

    def _alpha_ema(self) -> float:
        hl = max(1.0, float(self.p["ema_halflife"]))
        return 1.0 - math.exp(-math.log(2.0) / hl)

    def _ema(self, mid: float, td: Dict[str, Any]) -> float:
        a = self._alpha_ema()
        e = float(td.get(_K_EMA, mid))
        e = (1.0 - a) * e + a * mid
        td[_K_EMA] = e
        return e

    def _session_extrema(self, mid: float, td: Dict[str, Any]) -> Tuple[float, float, int]:
        n = int(td.get(_K_N, 0)) + 1
        td[_K_N] = n
        lo = float(td.get(_K_LO, mid))
        hi = float(td.get(_K_HI, mid))
        lo = min(lo, mid)
        hi = max(hi, mid)
        td[_K_LO] = lo
        td[_K_HI] = hi
        td[_K_MID] = mid
        return lo, hi, n

    def _take_long(
        self,
        d: OrderDepth,
        pos: int,
        cap: int,
        max_px: float,
    ) -> List[Order]:
        out: List[Order] = []
        lim = self.LIMIT[HYDROGEL_PACK]
        adv = int(self.p["adverse_volume"])
        prev = bool(self.p["prevent_adverse"])
        if not d.sell_orders or cap <= 0:
            return out
        best_ask = min(d.sell_orders.keys())
        amt = -d.sell_orders[best_ask]
        if best_ask > max_px:
            return out
        if prev and abs(amt) > adv:
            return out
        q = min(int(amt), cap, lim - pos)
        if q > 0:
            out.append(Order(HYDROGEL_PACK, best_ask, q))
            d.sell_orders[best_ask] += q
            if d.sell_orders[best_ask] == 0:
                del d.sell_orders[best_ask]
        return out

    def _take_short(
        self,
        d: OrderDepth,
        pos: int,
        cap: int,
        min_px: float,
    ) -> List[Order]:
        out: List[Order] = []
        lim = self.LIMIT[HYDROGEL_PACK]
        adv = int(self.p["adverse_volume"])
        prev = bool(self.p["prevent_adverse"])
        if not d.buy_orders or cap <= 0:
            return out
        best_bid = max(d.buy_orders.keys())
        amt = d.buy_orders[best_bid]
        if best_bid < min_px:
            return out
        if prev and abs(amt) > adv:
            return out
        q = min(int(amt), cap, lim + pos)
        if q > 0:
            out.append(Order(HYDROGEL_PACK, best_bid, -q))
            d.buy_orders[best_bid] -= q
            if d.buy_orders[best_bid] == 0:
                del d.buy_orders[best_bid]
        return out

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
        out: Dict[str, List[Order]] = {}
        p = self.p
        for sym in state.order_depths:
            if sym != HYDROGEL_PACK:
                out[sym] = []
        if HYDROGEL_PACK not in state.order_depths:
            return out, conversions, json.dumps(td, separators=(",", ":"))
        d0 = state.order_depths[HYDROGEL_PACK]
        pos = int(state.position.get(HYDROGEL_PACK, 0))
        lim = self.LIMIT[HYDROGEL_PACK]
        mid = _mid_simple(d0)
        if mid is None:
            out[HYDROGEL_PACK] = []
            return out, conversions, json.dumps(td, separators=(",", ":"))
        ema = self._ema(mid, td)
        lo, hi, n = self._session_extrema(mid, td)
        rng = hi - lo
        clip = int(p["clip"])
        te = float(p["take_edge"])
        buf = float(p["ema_buffer"])
        burn = int(p["burn_in"])
        mnr = float(p["min_range"])
        orders: List[Order] = []
        d = _clone_depth(d0)

        if n < burn or rng < mnr:
            out[HYDROGEL_PACK] = []
            return out, conversions, json.dumps(td, separators=(",", ":"))

        dist = (mid - lo) / (rng + 1e-9)
        b_pct = float(p["buy_range_pct"])
        s_pct = float(p["sell_range_pct"])

        if dist < b_pct and mid < ema - buf and pos < lim - 5:
            # Session lower bucket + cheap vs EMA -> lift asks up to "fair" proxy
            max_px = ema - te
            take = self._take_long(
                d, pos, min(clip, lim - pos), max_px
            )
            orders.extend(take)
        elif dist > s_pct and mid > ema + buf and pos > -lim + 5:
            min_px = ema + te
            take = self._take_short(
                d, pos, min(clip, lim + pos), min_px
            )
            orders.extend(take)
        out[HYDROGEL_PACK] = orders
        return out, conversions, json.dumps(td, separators=(",", ":"))
