"""
Round 4 trader: HYDROGEL_PACK + VELVETFRUIT_EXTRACT + VEV options  [v7]

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
_T      = 0.016
_S_REF  = 5248.0

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


# ─────────────────────────────────────────────────────────────────────────────
# VEV options
# ─────────────────────────────────────────────────────────────────────────────
_VEV_LIMIT = 300
_VEV_MM    = 15

# Deep ITM: real-time intrinsic FV ± margin (always applied, no threshold).
# buy_margin / sell_margin in pts relative to (ema_S − K).
_VEV_DEEP_ITM: Dict[int, Dict] = {
    4000: {"buy_margin": 12.0, "sell_margin": 13.0},
    4500: {"buy_margin": 12.0, "sell_margin": 13.0},
}

# ATM/OTM: static p25/p75 bands + old-smile BS drift correction (v5 logic).
_VEV_ATM: Dict[int, Dict] = {
    5000: {"buy": 240.0, "sell": 264.0},
    5100: {"buy": 151.0, "sell": 173.0},
    5200: {"buy":  80.0, "sell": 100.0},
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
    *(f"VEV_{k}" for k in list(_VEV_DEEP_ITM) + list(_VEV_ATM) + list(_VEV_MM_ONLY)),
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

        # ── 3a. Deep ITM VEV (4000, 4500) — real-time intrinsic FV ───────────
        # buy_t / sell_t track ema_S continuously. No threshold gate.
        # Directional entry: ba+2 / bb−2 (not ba+4) — spread too wide to cross hard.
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

            intrinsic = max(ema_S - float(K), 0.0)
            buy_t  = intrinsic - cfg["buy_margin"]
            sell_t = intrinsic + cfg["sell_margin"]

            if mid < buy_t:
                want = _VEV_LIMIT - pos
                if want > 0:
                    o.append(Order(sym, ba + 2, want))
            elif mid > sell_t:
                want = _VEV_LIMIT + pos
                if want > 0:
                    o.append(Order(sym, bb - 2, -want))
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

            fv_delta = (_old_smile_fv(float(K), ema_S) - _VEV_FV_REF_ATM[K]) if apply_bs else 0.0
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