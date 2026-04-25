"""
Round 3 combined trader: HYDROGEL_PACK + VELVETFRUIT_EXTRACT + VEV options

=== KEY INSIGHT (from 409948.log analysis) ===
Theoretical max PnL from optimal band placement (day 2 data):
  HYDROGEL_PACK:   33,052  (we were getting 19,587 — sell at 9997, not 10025!)
  VELVETFRUIT:     13,410  (we were getting  9,479)
  VEV all strikes: 83,055  (we were getting 32,388 — bands too conservative)
  TOTAL MAX:      112,121  vs our prior 61,454

=== HYDROGEL FIX ===
Sell threshold dropped from 10025 → 9997. Price oscillates between 9915-10031
but our old sell (10025) barely fired. New sell (9997) fires 4× per day → 2× PnL.
Removed HYDRO_UNWIND (was at 9980, inside new neutral zone, killed cycles).

=== VEV BAND FIX ===
Bands now set to OBSERVED PRICE EXTREMES (p0/p100). Rule: directional OK when
range > spread. All strikes satisfy this:
  VEV_4000: range=35pts, spread=21pts ✓  (was wrongly MM-only — narrow bands lost)
  VEV_4500: range=35.5, spread=16    ✓  (extended from 749/773 to 741/775.5)
  VEV_5000: range=34pts, spread=6.2  ✓  (extended from 246.5/274.2 to 245/278)
  VEV_5100: range=29.5, spread=4.4   ✓  (extended from 158.6/184.9 to 157/186)
  VEV_5200: range=23pts, spread=3.0  ✓  (extended from 89.3/109.7 to 87.5/110)
  VEV_5300: range=16pts, spread=2.2  ✓  (extended from 45/53.5 to 41/55)
  VEV_5400: range=7pts,  spread=1.4  ✓  (extended from 15/18 to 12/18.5)
  VEV_5500: range=3pts,  spread=1.2  ✓  (moved from MM-only to directional)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ─────────────────────────────────────────────────────────────────────────────
# HYDROGEL_PACK
# ─────────────────────────────────────────────────────────────────────────────
# KEY INSIGHT: sell at 9997 (not 10025) — tighter bands fire 4x per day vs 1x
# Simulation: 9937/9997 → 33,052/day vs 9940/10025 → 19,587/day
HYDROGEL    = "HYDROGEL_PACK"
HYDRO_LIMIT = 200
HYDRO_BUY   = 9937   # was 9940 — extend to near observed min (9915)
HYDRO_SELL  = 9997   # was 10025 — tighter = more cycles (fires 4x vs 1x per day)
HYDRO_MM    = 15
# HYDRO_UNWIND removed: fired inside neutral zone (9980 < 9997), killed cycles

# ─────────────────────────────────────────────────────────────────────────────
# VELVETFRUIT_EXTRACT
# ─────────────────────────────────────────────────────────────────────────────
VELVET        = "VELVETFRUIT_EXTRACT"
VELVET_LIMIT  = 200
VELVET_BUY    = 5242   # was 5244 — extend to near observed min (5241.5)
VELVET_SELL   = 5275   # was 5274 — extend to near observed max (5276.5)
VELVET_MM     = 25
VELVET_MM_CUT = 100    # suppress passive MM when |pos| >= this

# ─────────────────────────────────────────────────────────────────────────────
# VEV options — per-strike mean-reversion
# ─────────────────────────────────────────────────────────────────────────────
_VEV_LIMIT = 300
_VEV_MM    = 15

# Bands set to OBSERVED PRICE EXTREMES (p0/p100) so every oscillation is captured.
# Rule: directional profitable when range > spread. All strikes below satisfy this.
# VEV_4000: range=35pts, spread=21pts → profitable (was wrongly MM-only)
# VEV_5500: range=3pts,  spread=1.17pts → profitable (range > spread ✓)
_VEV_CFG: Dict[int, Dict] = {
    4000: {"buy": 1241.5, "sell": 1276.0},  # restored directional: range=34.5 > spread=21 ✓
    4500: {"buy":  741.0, "sell":  775.5},  # full range (was 749/773, missed low buys)
    5000: {"buy":  245.0, "sell":  278.0},  # full range (was 246.5/274.2)
    5100: {"buy":  157.0, "sell":  186.0},  # full range (was 158.6/184.9)
    5200: {"buy":   87.5, "sell":  110.0},  # full range (was 89.3/109.7)
    5300: {"buy":   41.0, "sell":   55.0},  # full range (was 45.0/53.5)
    5400: {"buy":   12.0, "sell":   18.5},  # full range (was 15.0/18.0)
    5500: {"buy":    4.5, "sell":    7.5},  # restored directional: range=3 > spread=1.17 ✓
}
_VEV_MM_ONLY = (6000, 6500)  # only truly worthless OTM (bid=0, spread=1)

_ALL_SYMBOLS: Tuple[str, ...] = (
    HYDROGEL,
    VELVET,
    *(f"VEV_{k}" for k in list(_VEV_CFG) + list(_VEV_MM_ONLY)),
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