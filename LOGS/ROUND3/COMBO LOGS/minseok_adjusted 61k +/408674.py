"""
Round 3 combined trader: HYDROGEL_PACK + VELVETFRUIT_EXTRACT + VEV options

=== DIAGNOSIS FROM LOG 406865 ===
The IV z-score layer (added in previous iteration) caused a -14,751 regression:
  VEV_5000: teammate=5,335 → ours=40    (-5,295)
  VEV_5100: teammate=8,143 → ours=2,021 (-6,122)
  VEV_5200: teammate=5,598 → ours=1,004 (-4,593)

Root cause: IV z-score fires 10-unit trades in the neutral zone (90%+ of ticks),
creating ~4× more total trades vs the clean mean-reversion approach.
Each extra neutral-zone trade crosses the bid-ask spread (~6 pts on VEV_5000),
generating ~6 pts of friction per round-trip. With ~1,100 extra units traded on
VEV_5000 alone, that's ~6,600 in friction — matching the observed regression.

LESSON: fewer, more precise trades > more frequent, smaller trades.
IV z-score signals did not provide edge sufficient to cover their friction cost.

=== WHAT ACTUALLY WORKS (from teammate 405673 analysis) ===
Pure per-strike price-band mean-reversion:
  • BUY aggressively (ask+4) when mid < buy_t
  • SELL aggressively (bid-4) when mid > sell_t
  • Passive MM only in neutral zone when spread ≥ 3 (prevents crossing 1-2pt spreads)
  • MM fires BOTH sides (lean toward flat when |pos| > 75)

=== IMPROVEMENTS OVER TEAMMATE'S CODE ===
1. VEV_4500: buy@752  sell@771  — teammate had buy=727.9 (never fired) → now +2,704/day
2. VEV_4000: MM-only  — directional bands LOST -1,525 (spread 21pts > band 19pts)
   Passive MM earns the spread instead of paying it → expect ~+117 like teammate

Key rule discovered: directional sweep only profitable when bid-ask spread < band width.
  VEV_4000 spread 21 pts > band 19 pts → round-trip = -2 pts/unit → MM-only
  VEV_4500 spread 16 pts < band 19 pts → round-trip = +3 pts/unit → directional ✓
  VEV_5000-5400: spreads 5-7 pts << bands 8-28 pts → directional very profitable ✓

=== EXPECTED PERFORMANCE ===
  HYDROGEL_PACK       :  ~19,600/day  (unchanged — proven)
  VELVETFRUIT_EXTRACT :   ~9,500/day  (unchanged — proven)
  VEV_5000–5400       :  ~25,000/day  (same as teammate's proven bands)
  VEV_4500 directional:   ~2,700/day  (improved from teammate's +99 MM-only)
  VEV_4000 MM-only    :     ~117/day  (back to teammate's baseline, no longer losing)
  TOTAL               :  ~56,917/day  (+2,168 vs teammate's 54,749)

=== DTE (DAYS TO EXPIRATION) NOTE ===
OptionalAlpha research: with short DTE (our 7-day round), the key is:
  • Focus on strikes near ATM (VEV_5100/5200): highest time value per unit of risk
  • Passive MM earns best when spread ≥ 3 (requires discipline — skip tight spreads)
  • Don't over-trade: let the mean-reversion bands do the heavy lifting
This is already reflected in our spread≥3 guard and neutral-zone MM only approach.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ─────────────────────────────────────────────────────────────────────────────
# HYDROGEL_PACK
# ─────────────────────────────────────────────────────────────────────────────
HYDROGEL     = "HYDROGEL_PACK"
HYDRO_LIMIT  = 200
HYDRO_BUY    = 9940
HYDRO_SELL   = 10025
HYDRO_UNWIND = 9980
HYDRO_MM     = 15

# ─────────────────────────────────────────────────────────────────────────────
# VELVETFRUIT_EXTRACT
# ─────────────────────────────────────────────────────────────────────────────
VELVET        = "VELVETFRUIT_EXTRACT"
VELVET_LIMIT  = 200
VELVET_BUY    = 5244
VELVET_SELL   = 5274
VELVET_MM     = 25
VELVET_MM_CUT = 100    # suppress passive MM when |pos| >= this

# ─────────────────────────────────────────────────────────────────────────────
# VEV options — per-strike mean-reversion
# ─────────────────────────────────────────────────────────────────────────────
_VEV_LIMIT = 300
_VEV_MM    = 15

# Bands calibrated to observed mid-price p10/p90 distributions.
# 4000/4500: FIXED vs teammate (old buy thresholds were below observed minimum).
# 5000-5400: Teammate's working calibration — generating 25k/day combined.
# 5500: MM-only (range 3pts, spread too wide for profitable directional sweep).
_VEV_CFG: Dict[int, Dict] = {
    # 4000: REMOVED from directional — bid-ask spread (~21 pts) > band width (19 pts)
    #   Round-trip: sell at bid(1261) - buy at ask(1263) = -2 pts/unit → LOSS every cycle
    #   Moved to MM-only so we earn the spread passively instead.
    4500: {"buy":  749.0, "sell":  773.0},   # IMPROVED: band=24pts, spread~16pts → +8pts/unit ✓
    # Previous (752/771): 86/93 fires, +3pts/unit, ~2,704/day
    # New     (749/773): 39/44 fires, +8pts/unit, ~4,800/day est (+2,096 gain)
    # Fewer cycles but 2.7× higher edge per unit → better total at these firing rates
    5000: {"buy":  246.5, "sell":  274.2},   # teammate calibration — 5,335/day
    5100: {"buy":  158.6, "sell":  184.9},   # teammate calibration — 8,143/day ← best
    5200: {"buy":   89.3, "sell":  109.7},   # teammate calibration — 5,598/day
    5300: {"buy":   45.0, "sell":   53.5},   # teammate calibration — 5,069/day
    5400: {"buy":   15.0, "sell":   18.0},   # teammate calibration — 1,279/day
}
_VEV_MM_ONLY = (4000, 5500)  # MM-only: 4000 (spread too wide), 5500 (range too tight)

_ALL_SYMBOLS: Tuple[str, ...] = (
    HYDROGEL,
    VELVET,
    *(f"VEV_{k}" for k in list(_VEV_CFG) + list(_VEV_MM_ONLY)),
    "VEV_6000",
    "VEV_6500",
)


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


class Trader:

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

        # ── 1. HYDROGEL_PACK ─────────────────────────────────────────────────
        d = state.order_depths.get(HYDROGEL)
        if d and d.buy_orders and d.sell_orders:
            mid = _mid(d)
            if mid is not None:
                pos = int(state.position.get(HYDROGEL, 0))
                bb  = max(d.buy_orders)
                ba  = min(d.sell_orders)
                o: List[Order] = []

                if mid > HYDRO_SELL and pos > -HYDRO_LIMIT:
                    want = HYDRO_LIMIT + pos
                    if want > 0:
                        o = [Order(HYDROGEL, bb - 10, -want)]
                elif mid < HYDRO_BUY and pos < HYDRO_LIMIT:
                    want = HYDRO_LIMIT - pos
                    if want > 0:
                        o = [Order(HYDROGEL, ba + 10, want)]
                elif mid > HYDRO_UNWIND and pos > 0:
                    o = [Order(HYDROGEL, bb, -pos)]
                else:
                    bp = bb + 1; sp = ba - 1
                    if pos > 50:  bp -= 1; sp -= 1
                    elif pos < -50: bp += 1; sp += 1
                    bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                    bq = min(HYDRO_MM, HYDRO_LIMIT - pos)
                    sq = min(HYDRO_MM, HYDRO_LIMIT + pos)
                    if bq > 0: o.append(Order(HYDROGEL, int(bp),  bq))
                    if sq > 0: o.append(Order(HYDROGEL, int(sp), -sq))

                result[HYDROGEL] = o

        # ── 2. VELVETFRUIT_EXTRACT ───────────────────────────────────────────
        d = state.order_depths.get(VELVET)
        if d and d.buy_orders and d.sell_orders:
            mid = _mid(d)
            if mid is not None:
                pos = int(state.position.get(VELVET, 0))
                bb  = max(d.buy_orders)
                ba  = min(d.sell_orders)
                o   = []

                if mid < VELVET_BUY:
                    want = VELVET_LIMIT - pos
                    if want > 0:
                        o = [Order(VELVET, ba + 4, want)]
                elif mid > VELVET_SELL:
                    want = VELVET_LIMIT + pos
                    if want > 0:
                        o = [Order(VELVET, bb - 4, -want)]
                elif abs(pos) < VELVET_MM_CUT:
                    bp = bb + 1; sp = ba - 1
                    if pos > 50:  bp -= 1; sp -= 1
                    elif pos < -50: bp += 1; sp += 1
                    bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                    if bp < sp:
                        bq = min(VELVET_MM, VELVET_LIMIT - pos)
                        sq = min(VELVET_MM, VELVET_LIMIT + pos)
                        if bq > 0: o.append(Order(VELVET, int(bp),  bq))
                        if sq > 0: o.append(Order(VELVET, int(sp), -sq))

                result[VELVET] = o

        # ── 3. VEV options — pure price-band mean-reversion ──────────────────
        #
        # ONLY three modes:
        #   (a) BUY zone  (mid < buy_t)  : sweep buy to +300
        #   (b) SELL zone (mid > sell_t) : sweep sell to -300
        #   (c) NEUTRAL   (in between)   : passive MM if spread >= 3, else nothing
        #
        # No IV signals. No extra neutral-zone orders. This prevents the bid-ask
        # friction that killed VEV_5000-5400 in the previous version.

        for K, cfg in _VEV_CFG.items():
            sym = f"VEV_{K}"
            d   = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            mid = _mid(d)
            if mid is None:
                continue

            pos    = int(state.position.get(sym, 0))
            bb     = max(d.buy_orders)
            ba     = min(d.sell_orders)
            spread = ba - bb
            o      = []

            if mid < cfg["buy"]:
                # Option cheap → sweep long to position limit
                want = _VEV_LIMIT - pos
                if want > 0:
                    o.append(Order(sym, ba + 4, want))

            elif mid > cfg["sell"]:
                # Option expensive → sweep short to position limit
                want = _VEV_LIMIT + pos
                if want > 0:
                    o.append(Order(sym, bb - 4, -want))

            elif spread >= 3:
                # Neutral zone: passive MM only when spread is worth crossing
                bp = bb + 1; sp = ba - 1
                if pos > _VEV_LIMIT // 4:    bp -= 1; sp -= 1   # lean sell when long
                elif pos < -(_VEV_LIMIT // 4): bp += 1; sp += 1  # lean buy when short
                bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0: o.append(Order(sym, int(bp),  bq))
                    if sq > 0: o.append(Order(sym, int(sp), -sq))
            # spread < 3 in neutral: do nothing (avoid paying 1-2pt spread for no edge)

            result[sym] = o

        # ── 4. VEV_5500 — passive MM only (range too tight for sweeps) ────────
        for K in _VEV_MM_ONLY:
            sym = f"VEV_{K}"
            d   = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            pos    = int(state.position.get(sym, 0))
            bb     = max(d.buy_orders)
            ba     = min(d.sell_orders)
            spread = ba - bb
            o      = []
            if spread >= 3:
                bp = bb + 1; sp = ba - 1
                if pos > _VEV_LIMIT // 4:    bp -= 1; sp -= 1
                elif pos < -(_VEV_LIMIT // 4): bp += 1; sp += 1
                bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0: o.append(Order(sym, int(bp),  bq))
                    if sq > 0: o.append(Order(sym, int(sp), -sq))
            result[sym] = o

        return result, 0, json.dumps(td, separators=(",", ":"))