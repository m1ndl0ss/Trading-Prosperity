"""
Round 4 combined trader — same as r4_main.py, but counterparty *tilt* for the
two underlyings only (HYDROGEL, VELVETFRUIT_EXTRACT), on top of the existing
mean-rev / sweeps / MM:

  HYDROGEL
    * FOLLOW  Mark 14  (lean with the MM: 14 buy -> tighten bid, 14 sell -> tighten ask)
    * AGAINST Mark 38  (lean opposite the taker: 38 buy -> tighten ask, 38 sell -> tighten bid)
    Replaces: r4_main.py behaviour that only *followed* Mark 38 on HYDRO.

  VELVETFRUIT_EXTRACT
    * FOLLOW  Mark 67  (67 buy -> tighten bid, 67 sell -> tighten ask)
    * AGAINST Mark 55  (55 buy -> tighten ask, 55 sell -> tight bid)
    (r4_main.py has no Mark skew on VELVET; this file adds the neutral-zone tilt only.)

  VEV options and all parameters match r4_main.py (thresholds, limits, VEV config).

Tilt = +/- 1 on passive quotes in the *neutral* MM band (same as prior Mark-38
logic), after inventory lean, then clamped so buy < best ask, sell > best bid.
If both buy and sell appear for a given Mark in the same `market_trades` slice
for that product, that Mark contributes no net tilt (same as r4_main Mark 38).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# copied from r4_main.py — unchanged
HYDROGEL_PACK = "HYDROGEL_PACK"
HYDRO_PARAMS: Dict[str, Any] = {
    "buy_thresh":  9935,
    "sell_thresh": 10030,
    "unwind_long": 9995,
    "mm_size":     15,
}
HYDRO_LIMIT = 200

VELVET = "VELVETFRUIT_EXTRACT"
VELVET_PARAMS: Dict[str, Any] = {
    "buy_thresh":  5215,
    "sell_thresh": 5275,
    "unwind_long": 5272,
    "mm_size":     25,
}
VELVET_LIMIT = 200

_VEV_STRIKES  = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)
_VEV_MM_ONLY: Tuple[int, ...] = ()
_VEV_CFG: Dict[int, Dict[str, Any]] = {
    4000: {"buy": 1246.0, "sell": 1276.0, "clip": 300},
    4500: {"buy":  749.0, "sell":  773.0, "clip": 300},
    5000: {"buy":  246.5, "sell":  277.0, "clip": 300},
    5100: {"buy":  158.6, "sell":  184.0, "clip": 300},
    5200: {"buy":   89.3, "sell":  109.7, "clip": 300},
    5300: {"buy":   40.0, "sell":   53.5, "clip": 300},
    5400: {"buy":    8.0, "sell":   16.0, "clip": 300},
    5500: {"buy":    5.5, "sell":    8.0, "clip": 300},
}
_VEV_LIMIT = 300
_VEV_MM    = 15

_ALL_SYMBOLS: Tuple[str, ...] = (
    HYDROGEL_PACK,
    VELVET,
    *(f"VEV_{k}" for k in _VEV_STRIKES),
    "VEV_6000",
    "VEV_6500",
)


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _last_tick_mark_sides(
    state: TradingState, symbol: str, mark: str
) -> tuple[bool, bool]:
    """At least one print last tick: mark bought (as buyer) / mark sold (as seller)."""
    buy = sell = False
    for t in state.market_trades.get(symbol, []):
        if t.buyer == mark:
            buy = True
        if t.seller == mark:
            sell = True
    return buy, sell


def _hydro_sweep_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = HYDRO_LIMIT - pos
    if want <= 0 or not depth.sell_orders:
        return []
    if min(depth.sell_orders.keys()) > max_price:
        return []
    return [Order(HYDROGEL_PACK, max_price, want)]


def _hydro_sweep_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = HYDRO_LIMIT + pos
    if want <= 0 or not depth.buy_orders:
        return []
    if max(depth.buy_orders.keys()) < min_price:
        return []
    return [Order(HYDROGEL_PACK, min_price, -want)]


def _hydro_unwind_long(depth: OrderDepth, pos: int) -> List[Order]:
    if pos <= 0 or not depth.buy_orders:
        return []
    return [Order(HYDROGEL_PACK, max(depth.buy_orders.keys()), -pos)]


def _velvet_sweep_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = VELVET_LIMIT - pos
    if want <= 0 or not depth.sell_orders:
        return []
    if min(depth.sell_orders.keys()) > max_price:
        return []
    return [Order(VELVET, max_price, want)]


def _velvet_sweep_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = VELVET_LIMIT + pos
    if want <= 0 or not depth.buy_orders:
        return []
    if max(depth.buy_orders.keys()) < min_price:
        return []
    return [Order(VELVET, min_price, -want)]


class Trader:
    LIMIT: Dict[str, int] = {
        HYDROGEL_PACK: HYDRO_LIMIT,
        VELVET: VELVET_LIMIT,
        **{f"VEV_{k}": _VEV_LIMIT for k in _VEV_STRIKES},
        "VEV_6000": _VEV_LIMIT,
        "VEV_6500": _VEV_LIMIT,
    }

    def __init__(self) -> None:
        self.hydro_p  = dict(HYDRO_PARAMS)
        self.velvet_p = dict(VELVET_PARAMS)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        result: Dict[str, List[Order]] = {sym: [] for sym in _ALL_SYMBOLS}

        m14b, m14s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 14")
        m38b, m38s = _last_tick_mark_sides(state, HYDROGEL_PACK, "Mark 38")
        m67b, m67s = _last_tick_mark_sides(state, VELVET, "Mark 67")
        m55b, m55s = _last_tick_mark_sides(state, VELVET, "Mark 55")

        # ── 1. HYDROGEL ───────────────────────────────────────────────────────
        if HYDROGEL_PACK in state.order_depths:
            d = state.order_depths[HYDROGEL_PACK]
            if d.buy_orders and d.sell_orders:
                mid = _mid(d)
                if mid is not None:
                    p      = self.hydro_p
                    buy_t  = int(p["buy_thresh"])
                    sell_t = int(p["sell_thresh"])
                    uw     = int(p["unwind_long"])
                    mm     = int(p.get("mm_size", 15))
                    pos    = int(state.position.get(HYDROGEL_PACK, 0))
                    bb     = max(d.buy_orders.keys())
                    ba     = min(d.sell_orders.keys())
                    o: List[Order] = []

                    if mid > sell_t and pos > -HYDRO_LIMIT:
                        o = _hydro_sweep_sell(d, pos, bb - 10)
                    elif mid < buy_t and pos < HYDRO_LIMIT:
                        o = _hydro_sweep_buy(d, pos, ba + 10)
                    elif mid > uw and pos > 0:
                        o = _hydro_unwind_long(d, pos)
                    else:
                        bp, sp = bb + 1, ba - 1
                        if pos > 50:
                            bp -= 1; sp -= 1
                        elif pos < -50:
                            bp += 1; sp += 1
                        # FOLLOW  Mark 14
                        if m14b and not m14s:
                            bp += 1
                        elif m14s and not m14b:
                            sp -= 1
                        # AGAINST Mark 38
                        if m38b and not m38s:
                            sp -= 1
                        elif m38s and not m38b:
                            bp += 1
                        bp = min(bp, ba - 1)
                        sp = max(sp, bb + 1)
                        bq = min(mm, HYDRO_LIMIT - pos)
                        sq = min(mm, HYDRO_LIMIT + pos)
                        if bq > 0:
                            o.append(Order(HYDROGEL_PACK, int(bp), bq))
                        if sq > 0:
                            o.append(Order(HYDROGEL_PACK, int(sp), -sq))
                    result[HYDROGEL_PACK] = o

        # ── 2. VELVETFRUIT ───────────────────────────────────────────────────
        if VELVET in state.order_depths:
            d = state.order_depths[VELVET]
            if d.buy_orders and d.sell_orders:
                mid = _mid(d)
                if mid is not None:
                    p      = self.velvet_p
                    buy_t  = int(p["buy_thresh"])
                    sell_t = int(p["sell_thresh"])
                    uw_l   = int(p.get("unwind_long", 99999))
                    mm     = int(p.get("mm_size", 25))
                    pos    = int(state.position.get(VELVET, 0))
                    bb     = max(d.buy_orders.keys())
                    ba     = min(d.sell_orders.keys())
                    o: List[Order] = []

                    if mid < buy_t:
                        o = _velvet_sweep_buy(d, pos, ba + 4)
                    elif mid > sell_t:
                        o = _velvet_sweep_sell(d, pos, bb - 4)
                    elif mid > uw_l and pos > 0:
                        o = [Order(VELVET, bb, -pos)]
                    elif abs(pos) < mm * 4:
                        bp, sp = bb + 1, ba - 1
                        if pos > 50:
                            bp -= 1; sp -= 1
                        elif pos < -50:
                            bp += 1; sp += 1
                        # FOLLOW  Mark 67
                        if m67b and not m67s:
                            bp += 1
                        elif m67s and not m67b:
                            sp -= 1
                        # AGAINST Mark 55
                        if m55b and not m55s:
                            sp -= 1
                        elif m55s and not m55b:
                            bp += 1
                        bp = min(bp, ba - 1)
                        sp = max(sp, bb + 1)
                        if bp < sp:
                            bq = min(mm, VELVET_LIMIT - pos)
                            sq = min(mm, VELVET_LIMIT + pos)
                            if bq > 0:
                                o.append(Order(VELVET, int(bp), bq))
                            if sq > 0:
                                o.append(Order(VELVET, int(sp), -sq))
                    result[VELVET] = o

        # ── 3. VEV options (identical to r4_main) ────────────────────────────
        for K in _VEV_STRIKES:
            sym = f"VEV_{K}"
            cfg = _VEV_CFG[K]
            d   = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            mid = _mid(d)
            if mid is None:
                continue
            buy_t  = cfg["buy"]
            sell_t = cfg["sell"]
            clip   = cfg["clip"]
            pos    = int(state.position.get(sym, 0))
            bb     = max(d.buy_orders.keys())
            ba     = min(d.sell_orders.keys())
            spread = ba - bb
            o: List[Order] = []

            if mid < buy_t:
                want = clip - pos
                if want > 0:
                    o.append(Order(sym, ba + 4, want))
            elif mid > sell_t:
                want = clip + pos
                if want > 0:
                    o.append(Order(sym, bb - 4, -want))
            elif spread >= 3:
                bp, sp = bb + 1, ba - 1
                if pos > clip // 4:
                    bp -= 1; sp -= 1
                elif pos < -(clip // 4):
                    bp += 1; sp += 1
                bp = min(bp, ba - 1)
                sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0:
                        o.append(Order(sym, int(bp), bq))
                    if sq > 0:
                        o.append(Order(sym, int(sp), -sq))
            result[sym] = o

        return result, 0, json.dumps(td, separators=(",", ":"))