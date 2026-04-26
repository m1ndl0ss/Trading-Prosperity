"""
top_of_line.py + VELVETFRUIT unwind_long AND unwind_short.

Both unwind rules active simultaneously:
  1. mid < buy_thresh   → sweep buy (go max long)
  2. mid > sell_thresh  → sweep sell (go max short)
  3. mid > unwind_long  AND pos > 0 → sell to flat at bid   ← take profit on long
  4. mid < unwind_short AND pos < 0 → buy  to flat at ask   ← take profit on short
  5. neutral zone       → passive MM

Best combined pair from simulation (off + 3-day train at 1k res):
  uw_long=5267 uw_short=5253: off=9,000  train=101,200  (vs base off=12,400 train=115,500)
  Both unwinds consistently reduce total PnL vs baseline in simulation — the strategy
  already extracts value without them.  However live exchange behaviour may differ.
  Default uses conservative levels (fires rarely) to minimise regression risk.

Simulation note: unwind_long=5267 fires 316 ticks but TRAINING PnL is UNCHANGED (+0).
This means on training days the unwind doesn't cost anything — it exits and re-enters
at the same net level. The official-day cost (−3,400) is day-specific. Worth testing.

Tune VELVET_PARAMS independently:
  unwind_long  — higher = fires less often (approach sell_thresh from below)
  unwind_short — lower  = fires less often (approach buy_thresh from above)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"
HYDRO_PARAMS: Dict[str, Any] = {
    "buy_thresh": 9940,
    "sell_thresh": 10025,
    "unwind_long": 9990,
    "mm_size": 15,
}
HYDRO_LIMIT = 200

VELVET = "VELVETFRUIT_EXTRACT"
VELVET_PARAMS: Dict[str, Any] = {
    "buy_thresh":   5244,
    "sell_thresh":  5274,
    "unwind_long":  5272,  # sell long to flat when mid recovers above this
    "unwind_short": 5247,  # buy short to flat when mid falls below this
    "mm_size": 25,
}
VELVET_LIMIT = 200

_VEV_STRIKES  = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)
_VEV_MM_ONLY: Tuple[int, ...] = ()
_VEV_CFG: Dict[int, Dict[str, Any]] = {
    4000: {"buy": 1246.0, "sell": 1276.0, "clip": 300},
    4500: {"buy": 749.0,  "sell": 773.0,  "clip": 300},
    5000: {"buy": 246.5,  "sell": 277.0,  "clip": 300},
    5100: {"buy": 158.6,  "sell": 184.9,  "clip": 300},
    5200: {"buy":  89.3,  "sell": 109.7,  "clip": 300},
    5300: {"buy":  45.0,  "sell":  53.5,  "clip": 300},
    5400: {"buy":  15.0,  "sell":  18.0,  "clip": 300},
    5500: {"buy":   5.5,  "sell":   8.0,  "clip": 300},
}
_VEV_LIMIT = 300
_VEV_MM = 15

_ALL_SYMBOLS: Tuple[str, ...] = (
    HYDROGEL_PACK, VELVET,
    *(f"VEV_{k}" for k in _VEV_STRIKES),
    "VEV_6000", "VEV_6500",
)


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _hydro_sweep_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = HYDRO_LIMIT - pos
    if want <= 0 or not depth.sell_orders: return []
    if min(depth.sell_orders.keys()) > max_price: return []
    return [Order(HYDROGEL_PACK, max_price, want)]


def _hydro_sweep_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = HYDRO_LIMIT + pos
    if want <= 0 or not depth.buy_orders: return []
    if max(depth.buy_orders.keys()) < min_price: return []
    return [Order(HYDROGEL_PACK, min_price, -want)]


def _hydro_unwind_long(depth: OrderDepth, pos: int) -> List[Order]:
    if pos <= 0 or not depth.buy_orders: return []
    return [Order(HYDROGEL_PACK, max(depth.buy_orders.keys()), -pos)]


def _velvet_sweep_buy(depth: OrderDepth, pos: int, max_price: int) -> List[Order]:
    want = VELVET_LIMIT - pos
    if want <= 0 or not depth.sell_orders: return []
    if min(depth.sell_orders.keys()) > max_price: return []
    return [Order(VELVET, max_price, want)]


def _velvet_sweep_sell(depth: OrderDepth, pos: int, min_price: int) -> List[Order]:
    want = VELVET_LIMIT + pos
    if want <= 0 or not depth.buy_orders: return []
    if max(depth.buy_orders.keys()) < min_price: return []
    return [Order(VELVET, min_price, -want)]


class Trader:
    LIMIT: Dict[str, int] = {
        HYDROGEL_PACK: HYDRO_LIMIT,
        VELVET: VELVET_LIMIT,
        **{f"VEV_{k}": _VEV_LIMIT for k in _VEV_STRIKES},
        "VEV_6000": _VEV_LIMIT, "VEV_6500": _VEV_LIMIT,
    }

    def __init__(self) -> None:
        self.hydro_p  = dict(HYDRO_PARAMS)
        self.velvet_p = dict(VELVET_PARAMS)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict): td = raw
            except (json.JSONDecodeError, TypeError): td = {}

        result: Dict[str, List[Order]] = {sym: [] for sym in _ALL_SYMBOLS}

        # 1 — HYDROGEL
        if HYDROGEL_PACK in state.order_depths:
            d = state.order_depths[HYDROGEL_PACK]
            if d.buy_orders and d.sell_orders:
                mid = _mid(d)
                if mid is not None:
                    p = self.hydro_p
                    buy_t = int(p["buy_thresh"]); sell_t = int(p["sell_thresh"])
                    uw = int(p["unwind_long"]); mm = int(p.get("mm_size", 15))
                    pos = int(state.position.get(HYDROGEL_PACK, 0))
                    bb = max(d.buy_orders.keys()); ba = min(d.sell_orders.keys())
                    o: List[Order] = []
                    if mid > sell_t and pos > -HYDRO_LIMIT:
                        o = _hydro_sweep_sell(d, pos, bb - 10)
                    elif mid < buy_t and pos < HYDRO_LIMIT:
                        o = _hydro_sweep_buy(d, pos, ba + 10)
                    elif mid > uw and pos > 0:
                        o = _hydro_unwind_long(d, pos)
                    else:
                        bp, sp = bb + 1, ba - 1
                        if pos > 50: bp -= 1; sp -= 1
                        elif pos < -50: bp += 1; sp += 1
                        bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                        bq = min(mm, HYDRO_LIMIT - pos); sq = min(mm, HYDRO_LIMIT + pos)
                        if bq > 0: o.append(Order(HYDROGEL_PACK, int(bp), bq))
                        if sq > 0: o.append(Order(HYDROGEL_PACK, int(sp), -sq))
                    result[HYDROGEL_PACK] = o

        # 2 — VELVET (with unwind_long AND unwind_short)
        if VELVET in state.order_depths:
            d = state.order_depths[VELVET]
            if d.buy_orders and d.sell_orders:
                mid = _mid(d)
                if mid is not None:
                    p = self.velvet_p
                    buy_t  = int(p["buy_thresh"])
                    sell_t = int(p["sell_thresh"])
                    uw_l   = int(p["unwind_long"])    # unwind long  — sell to flat at bid
                    uw_s   = int(p["unwind_short"])   # unwind short — buy  to flat at ask
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
                        # Unwind long: hit bid to go flat
                        o = [Order(VELVET, bb, -pos)]
                    elif mid < uw_s and pos < 0:
                        # Unwind short: lift ask to go flat
                        o = [Order(VELVET, ba, -pos)]  # -pos is positive
                    elif abs(pos) < mm * 4:
                        bp, sp = bb + 1, ba - 1
                        if pos > 50: bp -= 1; sp -= 1
                        elif pos < -50: bp += 1; sp += 1
                        bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                        if bp < sp:
                            bq = min(mm, VELVET_LIMIT - pos)
                            sq = min(mm, VELVET_LIMIT + pos)
                            if bq > 0: o.append(Order(VELVET, int(bp), bq))
                            if sq > 0: o.append(Order(VELVET, int(sp), -sq))
                    result[VELVET] = o

        # 3 — VEV
        for K in _VEV_STRIKES:
            sym = f"VEV_{K}"; cfg = _VEV_CFG[K]
            d = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders: continue
            mid = _mid(d)
            if mid is None: continue
            buy_t = cfg["buy"]; sell_t = cfg["sell"]; clip = cfg["clip"]
            pos = int(state.position.get(sym, 0))
            bb = max(d.buy_orders.keys()); ba = min(d.sell_orders.keys())
            spread = ba - bb; o: List[Order] = []
            if mid < buy_t:
                want = clip - pos
                if want > 0: o.append(Order(sym, ba + 4, want))
            elif mid > sell_t:
                want = clip + pos
                if want > 0: o.append(Order(sym, bb - 4, -want))
            elif spread >= 3:
                bp, sp = bb + 1, ba - 1
                if pos > clip // 4: bp -= 1; sp -= 1
                elif pos < -(clip // 4): bp += 1; sp += 1
                bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0: o.append(Order(sym, int(bp), bq))
                    if sq > 0: o.append(Order(sym, int(sp), -sq))
            result[sym] = o

        return result, 0, json.dumps(td, separators=(",", ":"))
