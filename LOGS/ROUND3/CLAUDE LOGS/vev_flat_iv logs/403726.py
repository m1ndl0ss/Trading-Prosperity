"""
VEV options — flat implied-vol BS scalp + passive MM.

Key findings from empirical analysis (30,000 ticks, 3 training days):

  Model accuracy (flat IV extracted from nearest ATM option):
    Strike  mean_residual  std   max_abs  → verdict
    5000       −0.13       0.63   2.41    near-perfect, no scalp signals at threshold=4
    5100       −0.65       1.35   3.36    near-perfect
    5200       −0.32       0.70   2.45    near-perfect
    5300        0.28       0.69   2.19    near-perfect
    5400       −2.62       1.31   4.85    ~13% ticks at |res|≥4 (see bias note below)
    5500        0.57       0.51   1.95    near-perfect

  Smile model (old tryout): biases of +3.9 on K=5100/5200, −5.5 on K=5400.
  Those systematic errors caused the tryout to trade AGAINST fair value.

  VEV_5400 bias: the market's actual IV for this strike is ~0.013 lower than
  the ATM reference, making the flat model overstate fair by ~2.6 units.
  A per-strike BIAS_CORRECTION table removes this systematic error.

What this strategy does:
  Layer 1 — BS SCALP (all 6 strikes):
    Extract IV each tick from the nearest ATM option (min extrinsic 3).
    Apply bias correction. Buy/sell when |market − adjusted_fair| ≥ SCALP_THRESH.
    Clip = 15 for OTM (K≥5300), 5 for ITM (K≤5100), 10 for K=5200.
    With flat IV, scalp signals are rare (only large 3-std outliers), so this
    layer mostly does nothing but catches genuine outliers safely.

  Layer 2 — PASSIVE MM (K=5200 only, spread=3):
    Post bid+1 / ask−1 inside the 3-unit spread.
    Capture 1 unit per round trip on market-flow fills.
    Capped at mm_size=10 to limit delta exposure (delta≈0.55 for K=5200).
    Position skew: when long, lean sells; when short, lean buys.

  No VELVETFRUIT trading — managed by velvet_like_hydro.py.
  No explicit delta hedging — kept small clip sizes to limit directional risk.
  Skip VEV_4000/4500 (zero extrinsic), VEV_6000/6500 (stuck at 0.5).

TTE formula (same as 6aPlot / 7aPlot):
  Day 0 → 5 days start, Day 1 → 4 days, Day 2 → 3 days.
  T_years = (start_days − timestamp/1_000_000) / 365
  Day boundary detected when timestamp resets (ts < prev_ts).
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── Products ──────────────────────────────────────────────────────────────────
_UNDERLYING = "VELVETFRUIT_EXTRACT"
_TRADE_STRIKES = (5000, 5100, 5200, 5300, 5400, 5500)
_ALL_PRODUCTS = (_UNDERLYING,
                 "VEV_4000", "VEV_4500",
                 "VEV_5000", "VEV_5100", "VEV_5200",
                 "VEV_5300", "VEV_5400", "VEV_5500",
                 "VEV_6000", "VEV_6500")

# ── Per-strike bias correction (market mid − flat-IV BS, mean over training) ──
# Subtract this from BS fair to get the bias-adjusted fair price.
_BIAS: Dict[int, float] = {
    5000: -0.13,
    5100: -0.65,
    5200: -0.32,
    5300:  0.28,
    5400: -2.62,   # largest; mainly an IV-slope effect (OTM IV < ATM IV by ~0.013)
    5500:  0.57,
}

# ── Clip sizes ─────────────────────────────────────────────────────────────────
# Kept small so unhedged delta exposure stays below ~10 VELVETFRUIT equivalents.
_CLIP: Dict[int, int] = {
    5000: 5,   # delta≈0.93 → 5×0.93 ≈ 4.7 VF equiv
    5100: 5,   # delta≈0.82 → 4.1 VF equiv
    5200: 10,  # delta≈0.55 → 5.5 VF equiv
    5300: 15,  # delta≈0.40 → 6.0 VF equiv
    5400: 15,  # delta≈0.20 → 3.0 VF equiv
    5500: 15,  # delta≈0.10 → 1.5 VF equiv
}

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PARAMS: Dict[str, Any] = {
    # Scalp trigger: |market_mid − adjusted_fair| must exceed this.
    # Residual std is 0.5–1.4 per strike; threshold=4 means ~3σ outlier.
    "scalp_thresh": 4.0,
    # Min extrinsic for IV extraction from a reference option (avoids near-zero).
    "min_extrinsic_ref": 3.0,
    # Min extrinsic to trade any option at all (skip deeply ITM/OTM at expiry).
    "min_extrinsic_trade": 0.8,
    # Passive MM: order size per side (K=5200 only).
    "mm_size": 10,
    # Position limit per option.
    "opt_limit": 200,
}

# ── TTE constants ─────────────────────────────────────────────────────────────
_TTE_START = {0: 5.0, 1: 4.0, 2: 3.0}   # days for competition sessions 0/1/2
_TK_D   = "d"    # session-day index in traderData
_TK_PTS = "pts"  # prev timestamp

# ── Math ──────────────────────────────────────────────────────────────────────
_N = NormalDist()


def _bs(S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
    """BS call price and delta (r=0, q=0)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        prc = max(0.0, S - K)
        dlt = 1.0 if S > K else 0.0
        return prc, dlt
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2), _N.cdf(d1)


def _iv(market_px: float, S: float, K: float, T: float,
        lo: float = 0.005, hi: float = 4.0) -> Optional[float]:
    """Bisection implied vol.  Returns None if market ≤ intrinsic or ill-conditioned."""
    intrinsic = max(0.0, S - K)
    if market_px <= intrinsic + 0.05 or T <= 0.0:
        return None
    for _ in range(50):
        mid = (lo + hi) * 0.5
        p = _bs(S, K, T, mid)[0]
        if p > market_px:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    result = (lo + hi) * 0.5
    return result if 0.05 <= result <= 3.0 else None


def _mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0


def _extract_iv(
    order_depths: Dict[str, OrderDepth],
    S: float,
    T: float,
    min_ext: float,
) -> Optional[float]:
    """
    Extract a single flat IV from the most ATM option with sufficient extrinsic.
    Tries candidates in order of |K − S| (nearest first).
    """
    candidates = sorted(_TRADE_STRIKES, key=lambda k: abs(k - S))
    for K in candidates:
        depth = order_depths.get(f"VEV_{K}")
        if not depth:
            continue
        c_mid = _mid(depth)
        if c_mid is None:
            continue
        intrinsic = max(0.0, S - K)
        extrinsic = c_mid - intrinsic
        if extrinsic < min_ext:
            continue
        iv = _iv(c_mid, S, K, T)
        if iv is not None:
            return iv
    return None


class Trader:
    def __init__(self) -> None:
        self.params = dict(PARAMS)
        self._prev_iv: float = 0.33   # warm-start; replaced after first extraction

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        p       = self.params
        scalp_t = float(p["scalp_thresh"])
        min_ext_ref   = float(p["min_extrinsic_ref"])
        min_ext_trade = float(p["min_extrinsic_trade"])
        mm_size = int(p["mm_size"])
        lim     = int(p["opt_limit"])

        # ── Placeholders ──────────────────────────────────────────────────────
        result: Dict[str, List[Order]] = {sym: [] for sym in _ALL_PRODUCTS}

        # ── Deserialise traderData ────────────────────────────────────────────
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Session-day and TTE ───────────────────────────────────────────────
        ts = int(state.timestamp)
        prev_ts = td.get(_TK_PTS)
        day = int(td.get(_TK_D, 0))
        if prev_ts is not None and ts < int(prev_ts):
            day = min(day + 1, 2)
        td[_TK_D]   = day
        td[_TK_PTS] = ts
        start_days  = _TTE_START[day]
        T = max((start_days - ts / 1_000_000.0) / 365.0, 1e-9)

        # ── Underlying price ──────────────────────────────────────────────────
        vf_depth = state.order_depths.get(_UNDERLYING)
        if not vf_depth or not vf_depth.buy_orders or not vf_depth.sell_orders:
            return result, 0, json.dumps(td, separators=(",", ":"))
        S = _mid(vf_depth)
        if S is None or S <= 0.0:
            return result, 0, json.dumps(td, separators=(",", ":"))

        # ── Extract flat IV (or carry previous) ───────────────────────────────
        iv = _extract_iv(state.order_depths, S, T, min_ext_ref)
        if iv is not None:
            self._prev_iv = iv
        else:
            iv = self._prev_iv

        # ── Option loop ───────────────────────────────────────────────────────
        for K in _TRADE_STRIKES:
            sym   = f"VEV_{K}"
            depth = state.order_depths.get(sym)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            c_mid = _mid(depth)
            if c_mid is None:
                continue

            intrinsic = max(0.0, S - K)
            extrinsic = c_mid - intrinsic
            if extrinsic < min_ext_trade:
                continue

            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            spread   = best_ask - best_bid
            pos      = int(state.position.get(sym, 0))
            clip     = _CLIP[K]
            bias     = _BIAS.get(K, 0.0)

            fair_raw, delta = _bs(S, K, T, iv)
            fair = fair_raw - bias   # bias-adjusted fair value

            orders: List[Order] = []

            # ── Layer 1: BS scalp on large deviations ─────────────────────────
            if best_ask <= fair - scalp_t:
                bq = min(clip, lim - pos)
                if bq > 0:
                    orders.append(Order(sym, best_ask, bq))
            elif best_bid >= fair + scalp_t:
                sq = min(clip, lim + pos)
                if sq > 0:
                    orders.append(Order(sym, best_bid, -sq))

            # ── Layer 2: Passive MM inside the spread (K=5200 only) ───────────
            elif K == 5200 and spread >= 3:
                buy_px  = best_bid + 1
                sell_px = best_ask - 1
                # Inventory skew
                if pos > 30:
                    buy_px  -= 1
                    sell_px -= 1
                elif pos < -30:
                    buy_px  += 1
                    sell_px += 1
                buy_px  = min(buy_px,  best_ask - 1)
                sell_px = max(sell_px, best_bid + 1)
                if buy_px < sell_px:
                    bq = min(mm_size, lim - pos)
                    sq = min(mm_size, lim + pos)
                    if bq > 0:
                        orders.append(Order(sym, int(buy_px),  bq))
                    if sq > 0:
                        orders.append(Order(sym, int(sell_px), -sq))

            result[sym] = orders

        return result, 0, json.dumps(td, separators=(",", ":"))