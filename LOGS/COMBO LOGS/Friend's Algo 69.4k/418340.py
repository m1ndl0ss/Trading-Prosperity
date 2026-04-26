"""
Round 3 combined trader: HYDROGEL_PACK + VELVETFRUIT_EXTRACT + VEV options  [v11]

=== CORE STRATEGY ===
VEV options: price-band mean reversion with BS-model drift correction.

The base bands (buy_t / sell_t) are calibrated to day-0 p25/p75 of observed
prices — symmetric zones that maximise completed buy→sell cycles.
A slow EMA of the live VELVETFRUIT price tracks genuine underlying drift while
filtering out the 5244↔5274 intraday oscillation that drove v5's false signals.

  fv_delta = BS_fair(K, EMA_S) − BS_fair(K, S_REF)
  adjusted_buy  = static_buy  + fv_delta
  adjusted_sell = static_sell + fv_delta

In the backtest (EMA barely moves) → bands ≈ static.
In round 3 if VELVETFRUIT drifts structurally, the bands shift with it.

=== VEV BAND CALIBRATION (v9) ===
Key lesson from v8 (score 61,454→45,236): price trends across 3 days matter.
  VEV_4500/5000: +9-12 pt upward trend (tracks VELVETFRUIT+13).
  VEV_5300/5400: -3-4 pt downward trend (theta decay / market-maker repricing).
Sell zones that go too far below the observed distribution cause directional losses.

v12 = v11 with VEV_4500 sell reverted 770→773:
  HYDROGEL unwind: 9980→9990 (teammate confirmed +652)
  VEV_4000: MM-only→directional [1246,1276] (teammate confirmed +3,541; old buy=1227 was below day-min 1241.5, never fired)
  VEV_5000: sell 274.2→277 (teammate confirmed +68; fewer but higher-quality fills)
  VEV_5100: sell 184.9→184.0 (v11 confirmed +142, log 417785)
  VEV_5200: sell 109.7→104, buy 89.3→91 (OUR confirmed +3,155, logs 415901+417785)
  VEV_5500: MM-only→directional [5.5,8.0] (teammate confirmed +445)
  VEV_4500: sell=773 (teammate's value; our 770 cost -1,407 in v11 vs log 416844)
v11 backtest (log 417785): 67,950. v12 projected: ~69,357.

=== IV SMILE ===
IV(m) = 7.867·m² − 15.462·m + 7.922   (m = K/S, fitted to observed prices)
T_EXPIRY = 0.010 (~3.65 day equivalent), calibrated to observed option prices.
S_REF = 5259 (calibration point).

=== HYDROGEL (v4/v7 settings retained) ===
HYDRO_SELL=10025, HYDRO_UNWIND=9980.
v6 tested HYDRO_SELL=9997 → immediate -200 short at tick 0 (price at 10011),
MTM loss -4,167 by tick 10k. Conservative 10025 + quick unwind at 9980 is best.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ─────────────────────────────────────────────────────────────────────────────
# HYDROGEL_PACK
# ─────────────────────────────────────────────────────────────────────────────
HYDROGEL     = "HYDROGEL_PACK"
HYDRO_LIMIT  = 200
HYDRO_BUY    = 9940
HYDRO_SELL   = 10025  # reverted — 9997 caused early shorts during price spikes → MTM loss
HYDRO_UNWIND = 9990   # was 9980; teammate confirmed +652 (20,239 vs 19,587) — holds long slightly longer
HYDRO_MM     = 15

# ─────────────────────────────────────────────────────────────────────────────
# VELVETFRUIT_EXTRACT
# ─────────────────────────────────────────────────────────────────────────────
VELVET        = "VELVETFRUIT_EXTRACT"
VELVET_LIMIT  = 200
VELVET_BUY    = 5244
VELVET_SELL   = 5274
VELVET_MM     = 25
VELVET_MM_CUT = 100

# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes / IV smile helpers  (r = 0, European call)
# ─────────────────────────────────────────────────────────────────────────────
_SQRT2    = 1.4142135623730951
_IV_A     =  7.867   # IV smile: IV(m) = A·m² + B·m + C, m = K / S
_IV_B     = -15.462
_IV_C     =  7.922
_T        =  0.010   # time-to-expiry (calibrated from observed option prices)
_S_REF    = 5259.0   # calibration underlying price


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price, r = 0."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(T)
    if sq < 1e-9:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sq
    return S * _ncdf(d1) - K * _ncdf(d1 - sq)


def _smile_iv(K: float, S: float) -> float:
    m = K / S
    return max(_IV_A * m * m + _IV_B * m + _IV_C, 0.05)


def _vev_fv(K: float, S: float) -> float:
    return _bs_call(S, K, _T, _smile_iv(K, S))


# ─────────────────────────────────────────────────────────────────────────────
# VEV options config
# ─────────────────────────────────────────────────────────────────────────────
_VEV_LIMIT = 300
_VEV_MM    = 15

# Static bands — calibrated to day-0 p25/p75 of observed price distribution.
# Symmetric buy/sell zone exposure maximises completed cycles; previous sell
# thresholds were above p90 for most strikes, leaving the sell zone almost
# unreachable (e.g. VEV_4500 sell zone was only 4% of ticks on day-0).
# Model: PnL ≈ min(b_tks, s_tks) * fill_rate * band_width.
# Calibrated fill rates (units/tick): 4500→0.786, 5000→0.259, 5100→0.327,
#   5200→0.279, 5300→0.341, 5400→0.514.
# Projected VEV PnL improvement: 32,265 → 76,010 (+43,745).
_VEV_CFG: Dict[int, Dict] = {
    # VEV_4000: DIRECTIONAL — teammate confirmed +3,541 vs MM-only (+134).
    #   Old buy=1227 sat below observed day-min (1241.5) → never fired.
    #   [1246,1276]: band=30 > spread=21 → net edge 9/unit. (log 416844)
    4000: {"buy": 1246.0, "sell": 1276.0},
    4500: {"buy": 749.0,  "sell": 773.0},   # REVERTED 770→773: sell=770 cost -1,407 vs teammate's 773 (v11 vs 416844)
    5000: {"buy": 246.5,  "sell": 277.0},   # teammate: sell 274.2→277 confirmed +68 (fewer but higher-quality fills)
    5100: {"buy": 158.6,  "sell": 184.0},   # ultra-conservative sell -0.9pt; day-1 sell zone 343→~450 ticks
    5200: {"buy":  91.0,  "sell": 104.0},   # OUR improvement: confirmed +3,158 (log 415901); sell 109.7→104
    5300: {"buy":  45.0,  "sell":  53.5},   # v7 unchanged (downward trend — sell reduction causes stuck longs)
    5400: {"buy":  15.0,  "sell":  18.0},   # v7 unchanged (downward trend)
    # VEV_5500: DIRECTIONAL — teammate confirmed +445 vs MM-only (0).
    #   sell=8.0 sits above day-0 max (7.5); buy=5.5 catches deep lows → long MTM gain.
    5500: {"buy":   5.5,  "sell":   8.0},
}

# BS fair values at S_REF — used to compute fv_delta when EMA drifts
_VEV_FV_REF: Dict[int, float] = {K: _vev_fv(float(K), _S_REF) for K in _VEV_CFG}

# EMA smoothing for underlying price
# alpha = 0.01 → time constant ≈ 100 ticks; ignores intraday oscillation.
# DRIFT_THRESHOLD = 20 pts: only apply BS adjustment when EMA has moved
# significantly from S_REF.  Prevents the small calibration drift from the
# backtest's mean oscillation (~7 pts) from shifting all bands and killing fills.
# In the backtest: EMA drifts ≤ 7 pts → threshold not met → exact v4 behavior.
# In round 3 with genuine level shift (e.g. VELVETFRUIT mean moves 30 pts):
#   threshold triggered → bands adapt via BS moneyness model.
_EMA_ALPHA      = 0.01
_DRIFT_THRESHOLD = 20.0

_VEV_MM_ONLY: Tuple[int, ...] = (6000, 6500)  # 4000+5500 now directional

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
                elif mid > HYDRO_UNWIND and pos > 0:
                    # Exit longs quickly once in neutral zone — captures gains, frees position
                    o = [Order(HYDROGEL, bb, -pos)]
                else:
                    # Neutral: passive MM
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
        S_now: Optional[float] = None
        if d and d.buy_orders and d.sell_orders:
            mid = _mid(d)
            if mid is not None:
                S_now = mid
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

        # Update slow EMA of VELVETFRUIT — filters oscillation, tracks drift
        ema_S: float = td.get("ema_S", _S_REF)
        if S_now is not None:
            ema_S = _EMA_ALPHA * S_now + (1.0 - _EMA_ALPHA) * ema_S
        td["ema_S"] = ema_S

        # ── 3. VEV options — BS-corrected band mean-reversion ─────────────────
        #
        # Bands shift by Δ(BS fair value) when EMA_S has drifted from S_REF.
        # Three modes:
        #   (a) BUY zone  (mid < buy_t)  : sweep long  to +300
        #   (b) SELL zone (mid > sell_t) : sweep short to −300
        #   (c) NEUTRAL                  : passive MM if spread ≥ 3

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

            # BS drift correction: only apply when EMA has genuinely shifted
            # (not just the 5-10 pt calibration noise from intraday oscillation)
            drift = ema_S - _S_REF
            if abs(drift) >= _DRIFT_THRESHOLD:
                fv_delta = _vev_fv(float(K), ema_S) - _VEV_FV_REF[K]
            else:
                fv_delta = 0.0
            buy_t  = cfg["buy"]  + fv_delta
            sell_t = cfg["sell"] + fv_delta

            if mid < buy_t:
                want = _VEV_LIMIT - pos
                if want > 0:
                    o.append(Order(sym, ba + 4, want))

            elif mid > sell_t:
                want = _VEV_LIMIT + pos
                if want > 0:
                    o.append(Order(sym, bb - 4, -want))

            elif spread >= 3:
                bp = bb + 1; sp = ba - 1
                if pos > _VEV_LIMIT // 4:      bp -= 1; sp -= 1
                elif pos < -(_VEV_LIMIT // 4): bp += 1; sp += 1
                bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0: o.append(Order(sym, int(bp),  bq))
                    if sq > 0: o.append(Order(sym, int(sp), -sq))

            result[sym] = o

        # ── 4. VEV MM_ONLY ────────────────────────────────────────────────────
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
                if pos > _VEV_LIMIT // 4:      bp -= 1; sp -= 1
                elif pos < -(_VEV_LIMIT // 4): bp += 1; sp += 1
                bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                if bp < sp:
                    bq = min(_VEV_MM, _VEV_LIMIT - pos)
                    sq = min(_VEV_MM, _VEV_LIMIT + pos)
                    if bq > 0: o.append(Order(sym, int(bp),  bq))
                    if sq > 0: o.append(Order(sym, int(sp), -sq))
            result[sym] = o

        return result, 0, json.dumps(td, separators=(",", ":"))