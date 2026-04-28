"""
Round 4 combined trader: HYDROGEL + VELVETFRUIT_EXTRACT + VEV mean-rev.

=== ROUND 4 CHANGES vs ROUND 3 top_of_line.py ===

1. THRESHOLD RE-CALIBRATION (full-day R4 grid search, all 3 training days positive):
   HYDROGEL    [9935, 10030]  (+15,800 vs [9940,10025] on R4 data)
   VELVETFRUIT [5215, 5275]   (+5,600  vs [5244,5274]  on R4 data)
   VEV_5200    [89.3, 109.7]  REVERTED from [91,104]: +23,400 on full-day R4 data.
     [91,104] only won in 4/30 windows (13%). The short-bias relies on the 1k-tick
     truncated window that no longer exists in the R4 backtester (now 10k ticks/day).
   VEV_5300    [40,  53.5]    Buy 45→40: fixed Day3 negative (was −2,100).
   VEV_5400    [ 8,  16]      Buy 15→8, sell 18→16: fixed Day3 negative (was −1,350).

2. MARK 38 SIGNAL ON HYDROGEL (new counterparty intelligence):
   Mark 38 is a trend-following liquidity taker. They buy at ask (+6 to +12 above mid)
   and sell at bid (−5 to −20 below mid) — consistently paying the spread directionally.
   R4 trades: 515 buy + 507 sell on HYDROGEL across 3 days.
   Detection: check state.market_trades for "Mark 38" each tick.
   Usage: skew the passive MM by 1 tick in the direction they're trading.
     Mark 38 bought → raise bp +1 (more aggressive on buy side)
     Mark 38 sold  → lower sp −1 (more aggressive on sell side)
   This is additive on the existing threshold logic, does not replace it.

=== R4 SIMULATION RESULTS (full 10k-tick days) ===
  HYDROGEL    [9935,10030]: Day1=52,200  Day2=67,400  Day3=28,800  Total=148,400
  VELVETFRUIT [5215,5275]:  Day1=24,300  Day2=41,100  Day3=35,000  Total=100,400
  VEV_5000    [246.5,277]:  Day1=70,950  Day2=40,350  Day3=23,700  Total=135,000
  VEV_5100    [158.6,184]:  Day1=76,050  Day2=33,150  Day3=21,150  Total=130,350
  VEV_5200    [89.3,109.7]: Day1=64,050  Day2=19,200  Day3= 2,400  Total= 85,650
  VEV_5300    [40,53.5]:    Day1=24,300  Day2=14,100  Day3=   900  Total= 39,300
  VEV_5400    [8,16]:       Day1= 4,500  Day2= 8,400  Day3= 2,850  Total= 15,750
  VEV_4000    [1246,1276]:  Day1=52,050  Day2=13,500  Day3=12,300  Total= 77,850
  VEV_4500    [749,773]:    Day1=33,450  Day2=14,850  Day3= 8,100  Total= 56,400
  (VEV_5500: kept [5.5,8.0], VEV_6000/6500: no trade)

=== COUNTERPARTY MAP (R4 trade data) ===
  Mark 14:  HYDROGEL + VF market maker (buys below mid, sells above mid, ~12-21pt edge)
  Mark 38:  HYDROGEL trend follower — crosses spread directionally (OUR SIGNAL)
  Mark 01:  VELVETFRUIT neutral + OTM VEV buyer (5300-6500, never sells them)
  Mark 22:  VELVETFRUIT seller + OTM VEV seller (5200-6500, nearly never buys)
  Mark 55:  VELVETFRUIT high-freq neutral trader
  Mark 67:  VELVETFRUIT buy-only (1510 qty over 3 days, creates buying pressure)
  Mark 49:  VELVETFRUIT sell-heavy (5:1 sell ratio)

HYDROGEL & VELVET: limit 200.  VEV strikes 4000-5500: limit 300 each.  VEV_6000/6500: no trade.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── HYDROGEL ──────────────────────────────────────────────────────────────────
HYDROGEL_PACK = "HYDROGEL_PACK"
HYDRO_PARAMS: Dict[str, Any] = {
    "buy_thresh":  9935,   # R4 optimised (was 9940); +15,800 total sim vs R3 thresholds
    "sell_thresh": 10030,  # R4 optimised (was 10025)
    "unwind_long": 9995,   # carries forward from R3 v6
    "mm_size":     15,
}
HYDRO_LIMIT = 200

# ── VELVETFRUIT ───────────────────────────────────────────────────────────────
VELVET = "VELVETFRUIT_EXTRACT"
VELVET_PARAMS: Dict[str, Any] = {
    "buy_thresh":  5215,   # R4 optimised (was 5244); Day3 range 5191-5300
    "sell_thresh": 5275,   # R4 optimised (was 5274)
    "unwind_long": 5272,   # carries forward from R3 v6 (+984 confirmed)
    "mm_size":     25,
}
VELVET_LIMIT = 200

# ── VEV options ───────────────────────────────────────────────────────────────
_VEV_STRIKES  = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)
_VEV_MM_ONLY: Tuple[int, ...] = ()
_VEV_CFG: Dict[int, Dict[str, Any]] = {
    4000: {"buy": 1246.0, "sell": 1276.0, "clip": 300},  # R4 all-positive: 52k/13.5k/12.3k
    4500: {"buy":  749.0, "sell":  773.0, "clip": 300},  # R4 all-positive: 33.5k/14.9k/8.1k
    5000: {"buy":  246.5, "sell":  277.0, "clip": 300},  # R4 all-positive: 71k/40.4k/23.7k
    5100: {"buy":  158.6, "sell":  184.0, "clip": 300},  # R4 all-positive: 76k/33.2k/21.2k
    5200: {"buy":   89.3, "sell":  109.7, "clip": 300},  # REVERTED [91,104]→[89.3,109.7]: +23,400 on R4 full-day
    5300: {"buy":   40.0, "sell":   53.5, "clip": 300},  # buy 45→40: fixes Day3 negative (was −2,100 → +900)
    5400: {"buy":    8.0, "sell":   16.0, "clip": 300},  # buy 15→8, sell 18→16: fixes Day3 negative (was −1,350 → +2,850)
    5500: {"buy":    5.5, "sell":    8.0, "clip": 300},  # EOD mark strategy; sell above day max
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

        # ── Mark 38 signal: detect in previous-tick trades ────────────────────
        # Mark 38 is a trend-following liquidity taker on HYDROGEL — they cross
        # the spread directionally. Their presence signals price direction.
        mark38_buy  = False
        mark38_sell = False
        for t in state.market_trades.get(HYDROGEL_PACK, []):
            if t.buyer  == "Mark 38":
                mark38_buy  = True
            if t.seller == "Mark 38":
                mark38_sell = True

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
                        # Neutral zone: passive MM with Mark 38 directional skew.
                        # When Mark 38 bought last tick → raise bp +1 (lean more bullish).
                        # When Mark 38 sold  last tick → lower sp −1 (lean more bearish).
                        bp, sp = bb + 1, ba - 1
                        if pos > 50:
                            bp -= 1; sp -= 1
                        elif pos < -50:
                            bp += 1; sp += 1
                        # Mark 38 skew (applied after inventory lean)
                        if mark38_buy and not mark38_sell:
                            bp += 1   # tighten buy side — we lean bullish with Mark 38
                        elif mark38_sell and not mark38_buy:
                            sp -= 1   # tighten sell side — we lean bearish with Mark 38
                        bp = min(bp, ba - 1)
                        sp = max(sp, bb + 1)
                        bq = min(mm, HYDRO_LIMIT - pos)
                        sq = min(mm, HYDRO_LIMIT + pos)
                        if bq > 0:
                            o.append(Order(HYDROGEL_PACK, int(bp), bq))
                        if sq > 0:
                            o.append(Order(HYDROGEL_PACK, int(sp), -sq))
                    result[HYDROGEL_PACK] = o

        # ── 2. VELVETFRUIT ────────────────────────────────────────────────────
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
                        o = [Order(VELVET, bb, -pos)]   # unwind long at bid
                    elif abs(pos) < mm * 4:
                        bp, sp = bb + 1, ba - 1
                        if pos > 50:
                            bp -= 1; sp -= 1
                        elif pos < -50:
                            bp += 1; sp += 1
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

        # ── 3. VEV options (mean-reversion bands + passive MM) ────────────────
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
