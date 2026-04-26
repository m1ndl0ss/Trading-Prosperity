"""
Round 3 combined trader: HYDROGEL + VELVETFRUIT_EXTRACT + VEV mean-rev.  (v3)

=== LESSON FROM v2 REGRESSION (−13,614) ===
Moving HYDROGEL to [9920, 10030] was the main mistake (−11,056).
The thresholds sat at the absolute extremes of the official range (9915 min, 10031 max):
  only 5-10 ticks at each extreme, not enough time to fill 200 units at 45/tick.
The original [9940, 10025] fires on 92 buy ticks + 28 sell ticks — plenty of fill time.
Rule: threshold must sit well INSIDE the observed range, not at its edges.

=== MINSEOK'S KEY INSIGHT (teammate, +6,731 from VEV_4500) ===
The only real improvement over wombo_combo_1 is fixing VEV_4500.
Original buy=727.9 was 13 units BELOW the day's observed minimum (741) — never fired.
Fix: [749.0, 773.0] → band=24pts, spread~16pts → edge=8pts/unit → +6,730/day.

Structural rule (from minseok): directional sweep profitable only when
  band_width > bid-ask spread
  VEV_4000: band~19pts < spread~21pts → LOSS per cycle → MM-only
  VEV_4500: band=24pts > spread~16pts → +8pts/unit → directional ✓
  VEV_5000-5400: bands >> spreads → directional very profitable ✓

=== VEV_5500 ===
Sell threshold [9.0] sat above day's max (7.5) in v1 — sell never fired, tiny +43.
With [5.5, 8.0]: buy fires at deep lows, sell threshold 8.0 is still above max (7.5)
but mark-to-market at EOD with lower buy entry gives +444 vs original +43.

=== v4 IMPROVEMENTS (over v3/top_of_line 61,899) ===
  1. HYDROGEL unwind_long 9980→9990: off+2,000 AND train+7,600 (both improve simultaneously).
     Holding the long position slightly longer before unwinding captures more of the recovery.
  2. VEV_4000 directional [1246,1276]: off sim +8,866 (134→9,000). Expected actual: ~+6,400.
     Minseok's MM-only was based on OLD 49-unit band [1227,1276] which never fired.
     Tight [1246,1276] band=30 > spread=21 → net edge=9/unit. Training all-positive.
  3. VEV_5000 sell 274.2→277.0: off+2,700, training remains all-positive.
     87 sell ticks fired at 274.2; with 277 fewer fires but at 3-unit better price.

=== EXPECTED PERFORMANCE (v4) ===
  HYDROGEL_PACK:   ~21,600 (unwind=9990 → off sim 24,600)
  VELVETFRUIT:      ~9,479 (unchanged)
  VEV_4000:         ~6,500 (new directional — off sim 9,000, ~73% efficiency)
  VEV_4500:         ~6,830 (Minseok-validated [749,773])
  VEV_5000:         ~6,000 (sell 274.2→277.0, off sim 18,000)
  VEV_5100:         ~8,143 (unchanged best option)
  VEV_5200–5500:    ~8,000 (unchanged)
  TOTAL:           ~66,000–68,000

HYDROGEL & VELVET: limit 200.  VEV strikes 4000–5500: limit 300 each.  VEV_6000/6500: no trade.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── HYDROGEL (379894) ────────────────────────────────────────────────────────
HYDROGEL_PACK = "HYDROGEL_PACK"
HYDRO_PARAMS: Dict[str, Any] = {
    "buy_thresh": 9940,    # proven — 92 ticks below on official data, plenty of fill time
    "sell_thresh": 10025,  # proven — 28 ticks above on official data
    "unwind_long": 9990,   # was 9980; 9990 gives off+2,000 AND train+7,600 simultaneously
    "mm_size": 15,
}
HYDRO_LIMIT = 200

# ── VELVET (velvet_like_hydro) ───────────────────────────────────────────────
VELVET = "VELVETFRUIT_EXTRACT"
VELVET_PARAMS: Dict[str, Any] = {
    "buy_thresh": 5244,    # proven — 16 buy ticks on official data
    "sell_thresh": 5274,   # proven — 22 sell ticks on official data
    "mm_size": 25,
}
VELVET_LIMIT = 200

# ── VEV (vev_mean_rev) ───────────────────────────────────────────────────────
_VEV_STRIKES  = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)  # all directional
_VEV_MM_ONLY: Tuple[int, ...] = ()  # VEV_4000 moved back to directional (see note below)
_VEV_CFG: Dict[int, Dict[str, Any]] = {
    # VEV_4000: REINSTATED as directional with tight thresholds [1246, 1276].
    #   Old thresholds [1227,1276] never fired on official (min=1241.5). Minseok's
    #   "band<spread" rule was based on the 49-unit old band — with 30-unit tight
    #   band, band(30) > spread(21), net edge=9/unit. Off sim: +8,866 (134→9,000).
    #   All 3 training days positive: 16,950/36,750/8,850.
    4000: {"buy": 1246.0, "sell": 1276.0, "clip": 300},  # NEW: off sim 9,000 vs 134 MM-only
    4500: {"buy": 749.0,  "sell": 773.0,  "clip": 300},  # FIXED: was [727.9,776.8], buy never fired (off min 741); [749,773] validated by minseok → +6,730
    5000: {"buy": 246.5,  "sell": 277.0,  "clip": 300},  # sell 274.2→277.0: off+2,700, train all-positive
    5100: {"buy": 158.6,  "sell": 184.9,  "clip": 300},  # proven → 8,143/day (best option)
    5200: {"buy":  89.3,  "sell": 109.7,  "clip": 300},  # proven → 5,598/day
    5300: {"buy":  45.0,  "sell":  53.5,  "clip": 300},  # proven → 5,069/day
    5400: {"buy":  15.0,  "sell":  18.0,  "clip": 300},  # proven → 1,279/day
    5500: {"buy":   5.5,  "sell":   8.0,  "clip": 300},  # sell above off max(7.5) but EOD mark on lower buy entry → +444 vs original +43
}
_VEV_LIMIT = 300
_VEV_MM = 15

_ALL_SYMBOLS: Tuple[str, ...] = (
    HYDROGEL_PACK,
    VELVET,
    *(f"VEV_{k}" for k in _VEV_STRIKES),
    *(f"VEV_{k}" for k in _VEV_MM_ONLY),
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
        **{f"VEV_{k}": _VEV_LIMIT for k in _VEV_MM_ONLY},
        "VEV_6000": _VEV_LIMIT,
        "VEV_6500": _VEV_LIMIT,
    }

    def __init__(self) -> None:
        self.hydro_p = dict(HYDRO_PARAMS)
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

        # 1 — HYDROGEL
        if HYDROGEL_PACK in state.order_depths:
            d = state.order_depths[HYDROGEL_PACK]
            if d.buy_orders and d.sell_orders:
                mid = _mid(d)
                if mid is not None:
                    p = self.hydro_p
                    buy_t = int(p["buy_thresh"])
                    sell_t = int(p["sell_thresh"])
                    uw = int(p["unwind_long"])
                    mm = int(p.get("mm_size", 15))
                    pos = int(state.position.get(HYDROGEL_PACK, 0))
                    bb = max(d.buy_orders.keys())
                    ba = min(d.sell_orders.keys())
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
                            bp -= 1
                            sp -= 1
                        elif pos < -50:
                            bp += 1
                            sp += 1
                        bp = min(bp, ba - 1)
                        sp = max(sp, bb + 1)
                        bq = min(mm, HYDRO_LIMIT - pos)
                        sq = min(mm, HYDRO_LIMIT + pos)
                        if bq > 0:
                            o.append(Order(HYDROGEL_PACK, int(bp), bq))
                        if sq > 0:
                            o.append(Order(HYDROGEL_PACK, int(sp), -sq))
                    result[HYDROGEL_PACK] = o

        # 2 — VELVET
        if VELVET in state.order_depths:
            d = state.order_depths[VELVET]
            if d.buy_orders and d.sell_orders:
                mid = _mid(d)
                if mid is not None:
                    p = self.velvet_p
                    buy_t = int(p["buy_thresh"])
                    sell_t = int(p["sell_thresh"])
                    mm = int(p.get("mm_size", 25))
                    pos = int(state.position.get(VELVET, 0))
                    bb = max(d.buy_orders.keys())
                    ba = min(d.sell_orders.keys())
                    o: List[Order] = []

                    if mid < buy_t:
                        o = _velvet_sweep_buy(d, pos, ba + 4)
                    elif mid > sell_t:
                        o = _velvet_sweep_sell(d, pos, bb - 4)
                    elif abs(pos) < mm * 4:
                        bp, sp = bb + 1, ba - 1
                        if pos > 50:
                            bp -= 1
                            sp -= 1
                        elif pos < -50:
                            bp += 1
                            sp += 1
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

        # 3 — VEV
        for K in _VEV_STRIKES:
            sym = f"VEV_{K}"
            cfg = _VEV_CFG[K]
            d = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            mid = _mid(d)
            if mid is None:
                continue
            buy_t = cfg["buy"]
            sell_t = cfg["sell"]
            clip = cfg["clip"]
            pos = int(state.position.get(sym, 0))
            bb = max(d.buy_orders.keys())
            ba = min(d.sell_orders.keys())
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
                    bp -= 1
                    sp -= 1
                elif pos < -(clip // 4):
                    bp += 1
                    sp += 1
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

        # 4 — VEV MM-only (VEV_4000: spread > band → directional loses money)
        for K in _VEV_MM_ONLY:
            sym = f"VEV_{K}"
            d = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            bb = max(d.buy_orders.keys())
            ba = min(d.sell_orders.keys())
            spread = ba - bb
            if spread < 3:
                continue
            pos = int(state.position.get(sym, 0))
            bp, sp = bb + 1, ba - 1
            if pos > _VEV_LIMIT // 4:
                bp -= 1; sp -= 1
            elif pos < -(_VEV_LIMIT // 4):
                bp += 1; sp += 1
            bp = min(bp, ba - 1)
            sp = max(sp, bb + 1)
            if bp < sp:
                bq = min(_VEV_MM, _VEV_LIMIT - pos)
                sq = min(_VEV_MM, _VEV_LIMIT + pos)
                o: List[Order] = []
                if bq > 0:
                    o.append(Order(sym, int(bp), bq))
                if sq > 0:
                    o.append(Order(sym, int(sp), -sq))
                result[sym] = o

        return result, 0, json.dumps(td, separators=(",", ":"))