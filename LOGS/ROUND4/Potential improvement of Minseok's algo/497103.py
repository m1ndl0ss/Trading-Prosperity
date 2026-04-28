"""
Round 4 trader: HYDROGEL_PACK + VELVETFRUIT_EXTRACT + VEV options  [v12]

=== WHAT WE LEARNED FROM v6 (score 26,254 vs v5's 48,238) ===

v6 applied a continuous real-time FV anchor to ALL VEV options. This broke
ATM/OTM (5000-5200) badly: spread capture dropped by ~50% because the smooth
FV tracking kept buy/sell thresholds glued to the market price, so bands were
almost never crossed. v5's threshold discontinuity (EMA drift ≥ 15 fires a
band jump) was inadvertently generating the oscillation cycles that produced
the spread capture. Removing it killed the trades.

What v6 DID fix correctly: VEV_4000 spread capture reversed from −1,072 to
+6,735. The real-time intrinsic FV (buy_t = ema_S − K − 12) exits the sell
zone after ~50 ticks instead of ~223, avoiding 170 ticks of aggressive
below-bid sells at bb−4 that v5 incurred before the threshold fired.

=== v7: REGIME SPLIT ===

Deep ITM (K ≤ 4500):   real-time intrinsic FV, continuous correction.
                        buy_t = (ema_S − K) − buy_margin  (always live)
                        sell_t = (ema_S − K) + sell_margin
                        Directional entry: ba+2 / bb−2  (not ba+4 / bb−4)

ATM/OTM (K ≥ 5000):    v5's static p25/p75 bands + threshold=15 drift gate.
                        Old IV smile (7.867/−15.462/7.922) preserved here
                        because the static bands were calibrated against it.
                        Directional entry: ba+4 / bb−4  (unchanged — the
                        oscillation volume is more important than slip cost)

=== v8: VEV_5300 SELL-BIASED (new calibrated IV smile) ===

Mispricing analysis on R4 data (3 days):
  Day 1: VEV_5300 overpriced +9.80 pts vs BS FV
  Day 2: VEV_5300 overpriced +4.47 pts vs BS FV
  Day 3: VEV_5300 underpriced −1.71 pts vs BS FV (VELVET elevated, near ATM)
  Average: +4.19 pts overpriced

Strategy: sell when market > new_smile_FV + 2, hold short.
  If VEV_5300 expires OTM (VELVETFRUIT < 5300, historically likely),
  we keep the full sale price (options settle at 0).
  Buy-back threshold set deep (FV − 12) so we rarely close early.

New IV smile coefficients recalibrated from R4 trade data:
  A=14.535, B=−29.732, C=15.424  (old: 7.867/−15.462/7.922)
  These are used ONLY for VEV_5300 FV computation; ATM/OTM static
  bands still use old smile to preserve calibrated thresholds.

=== v9: ADD VEV_5400 and VEV_5500 SELL-BIASED ===

3-day bot analysis confirmed Mark 22 systematically writes ALL OTM options.
VEV_5400 and VEV_5500 were NOT in _ALL_SYMBOLS — we were leaving money on the
table. Bid-vs-FV analysis from 3-day CSV data:

  VEV_5400: Day 1 overpriced +3.67 pts (100% of ticks), Day 2 +0.57 (68%),
            Day 3 −2.27 (only 0.05%). sell_margin=3.0 triggers only on
            clearly-overpriced days; at trigger, effective sell (bid−2) is
            above FV 100% of the time. buy_margin=10.0 → hold to settlement.
            VF never exceeded 5400 in 3 days → VEV_5400 expires worthless.
            Est. gain: ~4,180 (300 units × avg 13.93 sale price).

  VEV_5500: Day 1 overpriced +3.05 pts (97% of ticks), Day 2 +1.43 (42%),
            Day 3 −0.91. sell_margin=2.5 → triggers Days 1-2; at trigger,
            effective sell above FV 95% of the time. buy_margin=8.0.
            VF max was 5300, so VEV_5500 (K=5500) always expires worthless.
            Est. gain: ~1,400 (300 units × avg 4.67 sale price).

Refactored: 3c now loops over _VEV_SELL_BIASED dict (5300, 5400, 5500)
instead of a dedicated VEV_5300 block. Logic is identical.

=== v10: VEV_4000/4500 STATIC BANDS + VEV_5200 DRIFT FIX ===

Competitor analysis (495313.zip, score 65,017 vs our 62,326 — gap 2,692):
  VEV_4000: competitor +7,105 vs our +4,079 → gap +3,026
  VEV_4500: competitor +8,437 vs our +5,349 → gap +3,088
  VEV_5200: competitor +8,765 vs our +6,502 → gap +2,263

Root causes found and fixed:

1. VEV_4000/4500 (deep ITM): buy_margin=12/sell_margin=13 NEVER triggered.
   Premium range is only ±7pts from intrinsic; margins of 12-13 are 2× too wide.
   Fix: switch to static absolute bands [1246,1276] / [749,773] calibrated on
   3-day R4 data. These trigger on ~30-50% of ticks, capturing mean-reversion
   over the daily price range. Order prices: ba+4 / bb-4 (more aggressive).

2. VEV_5200 drift correction bug: old_smile_fv uses _T=0.016 (3-day constant).
   On Day 3 actual T≈0.004 (6× smaller). At VF=5295, drift_correction=+29.8pts,
   pushing sell_t to 129.8 — ABOVE market max of 122.5. Sell never triggers.
   buy_t rises to 109.8, so we BUY overpriced VEV_5200 aggressively (wrong!).
   Fix: add drift=False flag for VEV_5200 → static bands [89.3, 109.7].
   VEV_5000/5100 retain drift correction (they already outperform competitor).

=== v11: DYNAMIC T + R3 POST-MORTEM ===

Round 3 submission (486918.py) lesson: static absolute bands for VEV_5300/5400/5500
caused losses of −1,614 / −1,451 / −1,399. When VF moved outside calibration range,
fixed sell thresholds triggered BELOW fair value → sell loss. Our v9+ already fixes
this with FV-relative sell_margins via new_smile_fv.

Remaining issue: new_smile_fv used fixed _T=0.016 (calibrated for training Day 1).
By Day 3 (T=1/252≈0.004), FV_5300 falls from 37→9pts. With _T=0.016, FV stays at
37 → sell_t=39, triggering ~50% of ticks. With correct T=1/252, sell_t=11, triggering
~99% of ticks → much larger short position → much higher PnL at expiry (all OTM).

Fix: pass T_dyn (dynamically tracked per-day) into new_smile_fv instead of _T.
  T_dyn decreases by 1/252 per day, starting at _T_INIT = 2/252 ("1 day less"
  than training's Day-1 T=3/252, as the live competition expires 1 day sooner).
  Day detection: if state.timestamp < last saved timestamp → timestamp reset = new day.

Impact on sell thresholds at VF=5260:
  VEV_5300: FV(T=0.016)=37.0 → FV(T=2/252)=22.9 → FV(T=1/252)=9.2
  VEV_5400: FV(T=0.016)=7.5  → FV(T=2/252)=1.4  → FV(T=1/252)=0.1
  VEV_5500: FV(T=0.016)=1.6  → FV(T=2/252)=0.0  → FV(T=1/252)≈0
  → With correct T, sell triggers fire almost immediately; short positions
    build to -300 faster; expiry at 0 generates full premium profit.

=== v12: CMU-STYLE m_t = ln(K/S)/√T IV SMILE ===

CMU Physics top-7 team (Round 3) used standardized moneyness m_t = ln(K/S)/√TTE
in their quadratic IV fit. This is superior to K/S moneyness because m_t is
T-invariant: the same coefficients apply at all time-to-expiry values, whereas
K/S coefficients must be re-calibrated as T changes.

Refitted from R4 all-3-day data using m_t space:
  IV = A * m_t^2 + B * m_t + C   where m_t = ln(K/S) / sqrt(T)
  A = 0.1403,  B = 0.0052,  C = 0.3275   (RMSE = 0.0501)

vs old K/S refitted coefficients:
  A = 2.1530,  B = -4.2312, C = 2.4265   (RMSE = 0.0558)

The stale hardcoded K/S coefficients (A=14.535) are replaced with the m_t fit.
This produces correct FV at any T value without re-calibration, which is critical
for the dynamic T_dyn tracking across competition days.
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
HYDRO_BUY    = 9940   # p5 of 3-day distribution — only true extremes
HYDRO_SELL   = 10060  # was 10025 (hit 19% of ticks → dumped -200 into rising price)
               # 10060 hit only 1.3% of ticks — true extremes only
HYDRO_UNWIND = 9990
HYDRO_MM     = 30     # was 15 — larger passive orders to capture more Mark 38 flow

# ─────────────────────────────────────────────────────────────────────────────
# VELVETFRUIT_EXTRACT
# ─────────────────────────────────────────────────────────────────────────────
VELVET        = "VELVETFRUIT_EXTRACT"
VELVET_LIMIT  = 200
VELVET_BUY    = 5236
VELVET_SELL   = 5271
VELVET_MM     = 25
VELVET_MM_CUT = 100

MARK67         = "Mark 67"
MARK67_PREMIUM = 8

# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers  (r = 0, European call)
# ─────────────────────────────────────────────────────────────────────────────
_SQRT2  = 1.4142135623730951
_T      = 0.016          # legacy constant — kept for old_smile_fv drift correction only
_S_REF  = 5248.0

# Dynamic T tracking (v11)
# Training: Day1 T=3/252, Day2 T=2/252, Day3 T=1/252.
# Live competition: "1 day less expiration" → T_INIT = 2/252 (options expire 1 day sooner).
# T decreases linearly within each day and resets each new day via timestamp detection.
_T_INIT        = 2.0 / 252.0   # remaining T at start of live Day 0
_T_PER_DAY     = 1.0 / 252.0   # T consumed per day
_MAX_TIMESTAMP = 99900.0        # highest timestamp seen per training day

# Old IV smile — kept for ATM/OTM regime (bands were calibrated against it)
_IV_A =  7.867
_IV_B = -15.462
_IV_C =  7.922


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(T)
    if sq < 1e-9:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sq
    return S * _ncdf(d1) - K * _ncdf(d1 - sq)


def _old_smile_fv(K: float, S: float) -> float:
    """Old IV smile BS FV — used only for ATM/OTM drift correction reference."""
    m  = K / S
    iv = max(_IV_A * m * m + _IV_B * m + _IV_C, 0.05)
    return _bs_call(S, K, _T, iv)


# New IV smile: CMU-style m_t = ln(K/S)/sqrt(T) moneyness (v12)
# Refitted from R4 all-3-day data — T-invariant, RMSE=0.0501 (vs 0.0558 in K/S space)
_NEW_IV_A =  0.1403
_NEW_IV_B =  0.0052
_NEW_IV_C =  0.3275


def _new_smile_fv(K: float, S: float, T: float) -> float:
    """CMU-style IV smile BS FV — used for VEV_5300/5400/5500 directional trading.
    Uses standardized moneyness m_t = ln(K/S)/sqrt(T), which is T-invariant.
    T is passed dynamically (see v11/v12 day-tracking logic in run()).

    IMPORTANT: m_t is computed with T clamped to >= 1/252 to prevent divergence
    when T approaches 0 on Day 3 (OTM options expiring).  The actual T is still
    used inside _bs_call, so BS correctly returns ~0 for OTM options near expiry."""
    T_smile = max(T, 1.0 / 252.0)                            # clamp for stable m_t only
    m_t = math.log(K / S) / math.sqrt(T_smile)
    iv  = max(_NEW_IV_A * m_t * m_t + _NEW_IV_B * m_t + _NEW_IV_C, 0.05)
    return _bs_call(S, K, T, iv)                              # actual T for pricing


# Sell-biased OTM options: sell when overpriced, hold to expiry.
# Thresholds tuned from 3-day CSV analysis (see v9 notes in docstring).
_VEV_SELL_BIASED: Dict[int, Dict] = {
    5300: {"sell_margin":  2.0, "buy_margin": 12.0},  # avg +4.19 overpriced
    5400: {"sell_margin":  3.0, "buy_margin": 10.0},  # avg +3.67 Day1; conservative
    5500: {"sell_margin":  2.5, "buy_margin":  8.0},  # avg +3.05 Day1, +1.43 Day2
}

# ─────────────────────────────────────────────────────────────────────────────
# VEV options
# ─────────────────────────────────────────────────────────────────────────────
_VEV_LIMIT = 300
_VEV_MM    = 15

# Deep ITM: static absolute price bands (calibrated to 3-day R4 price ranges).
# v10: switched from intrinsic±margin (never triggered) to static thresholds.
# VEV_4000 mid range [1189,1302]; VEV_4500 mid range [690,800].
_VEV_DEEP_ITM: Dict[int, Dict] = {
    4000: {"buy": 1246.0, "sell": 1276.0},  # triggers on ~45% / ~30% of ticks
    4500: {"buy":  749.0, "sell":  773.0},  # triggers on ~50% / ~30% of ticks
}

# ATM/OTM: static p25/p75 bands + old-smile BS drift correction (v5 logic).
# v10: VEV_5200 band widened from [80,100]→[89.3,109.7] and drift correction
# disabled — old T=0.016 overcorrects Day 3 (actual T≈0.004), pushing sell_t
# to 129.8 (above market max 122.5), preventing all sells on the overpriced leg.
_VEV_ATM: Dict[int, Dict] = {
    5000: {"buy": 240.0, "sell": 264.0, "drift": True},
    5100: {"buy": 151.0, "sell": 173.0, "drift": True},
    5200: {"buy":  89.3, "sell": 109.7, "drift": False},  # static; no drift correction
}
_VEV_FV_REF_ATM: Dict[int, float] = {
    K: _old_smile_fv(float(K), _S_REF) for K in _VEV_ATM
}
_EMA_ALPHA       = 0.01
_DRIFT_THRESHOLD = 15.0

_VEV_MM_ONLY: Tuple[int, ...] = (6000, 6500)

_ALL_SYMBOLS: Tuple[str, ...] = (
    HYDROGEL,
    VELVET,
    *(f"VEV_{k}" for k in (
        list(_VEV_DEEP_ITM) + list(_VEV_ATM) +
        list(_VEV_SELL_BIASED) + list(_VEV_MM_ONLY)
    )),
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


def _mark67_active(state: TradingState) -> bool:
    for t in state.market_trades.get(VELVET, []):
        if getattr(t, "buyer", None) == MARK67:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Trader
# ─────────────────────────────────────────────────────────────────────────────

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

        mark67_on = _mark67_active(state)

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
                    if pos > 50:    bp -= 1; sp -= 1
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

                effective_sell = VELVET_SELL + (MARK67_PREMIUM if mark67_on else 0)

                if mid < VELVET_BUY:
                    want = VELVET_LIMIT - pos
                    if want > 0:
                        o = [Order(VELVET, ba + 4, want)]
                elif mid > effective_sell:
                    want = VELVET_LIMIT + pos
                    if want > 0:
                        o = [Order(VELVET, bb - 4, -want)]
                elif abs(pos) < VELVET_MM_CUT:
                    bp = bb + 1; sp = ba - 1
                    if pos > 50:    bp -= 1; sp -= 1
                    elif pos < -50: bp += 1; sp += 1
                    bp = min(bp, ba - 1); sp = max(sp, bb + 1)
                    if bp < sp:
                        bq = min(VELVET_MM, VELVET_LIMIT - pos)
                        sq = min(VELVET_MM, VELVET_LIMIT + pos)
                        if bq > 0: o.append(Order(VELVET, int(bp),  bq))
                        if sq > 0: o.append(Order(VELVET, int(sp), -sq))

                result[VELVET] = o

        ema_S: float = td.get("ema_S", _S_REF)
        if S_now is not None:
            ema_S = _EMA_ALPHA * S_now + (1.0 - _EMA_ALPHA) * ema_S
        td["ema_S"] = ema_S

        # ── Dynamic T (v11) ────────────────────────────────────────────────────
        # Detect new day when timestamp resets (state.timestamp < last seen ts).
        # T decreases by _T_PER_DAY each day and within-day proportionally.
        # _T_INIT = 2/252: "1 day less" than training's Day-1 starting T of 3/252.
        day_count: int   = td.get("day_count", 0)
        last_ts:   int   = td.get("last_ts",   state.timestamp)
        if state.timestamp < last_ts:          # timestamp reset → new day
            day_count += 1
        td["day_count"] = day_count
        td["last_ts"]   = state.timestamp
        # Linear interpolation within the day
        day_frac = state.timestamp / _MAX_TIMESTAMP   # 0→1 across the day
        T_dyn = max(
            _T_INIT - day_count * _T_PER_DAY - day_frac * _T_PER_DAY,
            1e-5,
        )

        # ── 3a. Deep ITM VEV (4000, 4500) — static absolute price bands ─────────
        # v10: switched from intrinsic±margin (never triggered, margin too wide
        # vs ±7pt premium range) to static bands calibrated on 3-day R4 data.
        # Band center tracks the typical day's mid range; buy at bottom, sell top.
        for K, cfg in _VEV_DEEP_ITM.items():
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
            o      = []

            buy_t  = cfg["buy"]
            sell_t = cfg["sell"]

            if mid < buy_t:
                want = _VEV_LIMIT - pos
                if want > 0:
                    o.append(Order(sym, ba + 4, want))
            elif mid > sell_t:
                want = _VEV_LIMIT + pos
                if want > 0:
                    o.append(Order(sym, bb - 4, -want))
            else:
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

        # ── 3b. ATM/OTM VEV (5000, 5100, 5200) — v5 static bands + threshold ─
        # v10: per-instrument drift flag; VEV_5200 uses static bands (drift=False)
        # because the fixed T=0.016 in old_smile_fv overcorrects on Day 3
        # (actual T≈0.004), pushing VEV_5200 sell_t to 129.8 > market max 122.5.
        drift    = ema_S - _S_REF
        apply_bs = abs(drift) >= _DRIFT_THRESHOLD

        for K, cfg in _VEV_ATM.items():
            sym = f"VEV_{K}"
            d   = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            mid = _mid(d)
            if mid is None:
                continue

            pos = int(state.position.get(sym, 0))
            bb  = max(d.buy_orders)
            ba  = min(d.sell_orders)
            o   = []

            use_drift = apply_bs and cfg.get("drift", True)
            fv_delta  = (_old_smile_fv(float(K), ema_S) - _VEV_FV_REF_ATM[K]) if use_drift else 0.0
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
            else:
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

        # ── 3c. Sell-biased OTM VEV (5300, 5400, 5500) — new IV smile ───────
        # FV tracks ema_S continuously so the sell zone naturally narrows when
        # VELVETFRUIT rises toward any strike.  Directional entry: bb−2 / ba+2.
        # Sell skew only when accidentally long (pos > LIMIT//4); no upward
        # skew when short — we want to maintain the short position.
        for K_sb, cfg_sb in _VEV_SELL_BIASED.items():
            sym_sb = f"VEV_{K_sb}"
            d_sb   = state.order_depths.get(sym_sb)
            if not d_sb or not d_sb.buy_orders or not d_sb.sell_orders:
                continue
            mid_sb = _mid(d_sb)
            if mid_sb is None:
                continue

            pos_sb = int(state.position.get(sym_sb, 0))
            bb_sb  = max(d_sb.buy_orders)
            ba_sb  = min(d_sb.sell_orders)
            fv_sb  = _new_smile_fv(float(K_sb), ema_S, T_dyn)   # v11: dynamic T
            sell_t = fv_sb + cfg_sb["sell_margin"]
            buy_t  = fv_sb - cfg_sb["buy_margin"]

            o_sb: List[Order] = []
            if mid_sb > sell_t and pos_sb > -_VEV_LIMIT:
                want = _VEV_LIMIT + pos_sb
                if want > 0:
                    o_sb.append(Order(sym_sb, bb_sb - 2, -want))
            elif mid_sb < buy_t and pos_sb < _VEV_LIMIT:
                want = _VEV_LIMIT - pos_sb
                if want > 0:
                    o_sb.append(Order(sym_sb, ba_sb + 2, want))
            else:
                bp_sb = bb_sb + 1; sp_sb = ba_sb - 1
                if pos_sb > _VEV_LIMIT // 4: bp_sb -= 1; sp_sb -= 1
                bp_sb = min(bp_sb, ba_sb - 1); sp_sb = max(sp_sb, bb_sb + 1)
                if bp_sb < sp_sb:
                    bq_sb = min(_VEV_MM, _VEV_LIMIT - pos_sb)
                    sq_sb = min(_VEV_MM, _VEV_LIMIT + pos_sb)
                    if bq_sb > 0: o_sb.append(Order(sym_sb, int(bp_sb),  bq_sb))
                    if sq_sb > 0: o_sb.append(Order(sym_sb, int(sp_sb), -sq_sb))

            result[sym_sb] = o_sb

        # ── 4. VEV MM_ONLY (6000, 6500) ──────────────────────────────────────
        for K in _VEV_MM_ONLY:
            sym = f"VEV_{K}"
            d   = state.order_depths.get(sym)
            if not d or not d.buy_orders or not d.sell_orders:
                continue
            pos = int(state.position.get(sym, 0))
            bb  = max(d.buy_orders)
            ba  = min(d.sell_orders)
            o   = []
            if (ba - bb) >= 3:
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