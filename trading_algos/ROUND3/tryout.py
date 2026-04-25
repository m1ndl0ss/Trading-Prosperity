"""
VELVETFRUIT_EXTRACT + VEV_* (Round 3 options on extract).

- Fair price: Black–Scholes call, r=0, with *implied vol* from the quadratic smile
  in moneyness m = K/S, fitted in ROUND3/6aPlot.py: v_hat = A*m^2 + B*m + C.
- Time: matches 6aPlot / 7aPlot — (start_tte_days − timestamp/1e6) / 365 years,
  with start_tte = 5, 4, 3 for each competition *session* (see traderData 'd').

**Local per-product replay** (backtest_round_pnl.py) only injects *one* symbol per run,
so VELVETFRUIT_EXTRACT is missing when replaying a VEV — we cannot see S. In that
case we skip VEV option logic; only VELVET-only replay will exercise the extract leg.
The official IMC engine passes the full order book: this strategy is intended for that.

**Post-mortem (LOGS/COMPOSER LOGS, ~−1466):** PnL was *positive* on VELVETFRUIT_EXTRACT and
small on VEV_5000, but *large negative* on near-ATM VEV_5100/5200/5300/5400. The
parabola+BS *fair* was systematically wrong live vs the book: aggressive 200-lot scalps on
"cheap" asks were picking off **bad** fair values, not the market. Default is now to run
**extract mean-reversion only**; re-enable VEV after recalibrating the smile (or with tiny
`OPTION_LIMIT` and high `SCALP_THRESHOLD` in paper mode).
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# --- 1. Smile: v_hat = A m^2 + B m + C (m = K/S). Refit on training tape: ---
#   python scripts/refit_vev_smile.py  →  same pipeline as ROUND3/6aPlot.py
A, B, C = 4.672405, -8.348788, 3.996275

# --- 2. Hyperparameters ---
# B–S + smile scalp on VEV: **off** by default (see module doc: official run ~−1.4k on VEVs).
TRADE_VEV_OPTIONS = False
SCALP_THRESHOLD = 6.0  # if TRADE_VEV: need large edge vs 2.0; small "mispricing" is noise
EMA_WINDOW = 20
EMA_THRESHOLD = 4.0
# Below this extrinsic, quotes are too noisy / dominated by one-tick discretization
MIN_EXTRINSIC = 0.5
# Per-strike cap if TRADE_VEV; 200 * many strikes = huge gross exposure. Start small.
OPTION_LIMIT = 40
EXTRACT_LIMIT = 200
# Implied vol from parabola: clip to a sane range before BS
SIGMA_MIN, SIGMA_MAX = 0.01, 2.0
# Local single-file backtest: set 0 / 1 / 2 to match `prices_round_3_day_{n}.csv` TTE.
# Live: keep 0; advance when `timestamp` resets (see traderData).
START_SESSION_DAY = 0

# Products this Trader touches (for empty placeholders)
PRODUCTS: Tuple[str, ...] = (
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
)

STRIKES: Tuple[int, ...] = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)

_N = NormalDist()
_TD = "t"
_PT = "p"
_D = "d"


def _empty_orders() -> Dict[str, List[Order]]:
    return {p: [] for p in PRODUCTS}


def _parse_td(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _mid(d: OrderDepth) -> Optional[float]:
    if not d.buy_orders or not d.sell_orders:
        return None
    return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2.0


def bs_call(S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
    """B–S call, r=0, q=0. Returns (price, delta)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        p = max(0.0, S - K)
        return p, 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N.cdf(d1) - K * _N.cdf(d2), _N.cdf(d1)


def _round_day_from_state(td: Dict[str, Any], ts: int) -> int:
    """
    Live multi-day: timestamp can reset (new day). When ts < previous ts, advance day.
    Cap at 2 (matches 3 session days: TTE 5,4,3 then stays at 3).
    """
    prev = td.get(_PT, None)
    d = int(td.get(_D, 0))
    if prev is not None and ts < int(prev):
        d = min(d + 1, 2)
    return d


def _tte_years(td: Dict[str, Any], ts: int) -> None:
    """Update td in place with pt, d; TTE uses same formula as 6aPlot (years)."""
    d = _round_day_from_state(td, ts)
    td[_D] = d
    start_days = 5.0 - float(d)  # 5, 4, 3
    t_days = start_days - (ts / 1_000_000.0)
    t_years = max(t_days, 1e-9) / 365.0
    td[_PT] = ts
    td[_TD] = t_years


class Trader:
    def __init__(self) -> None:
        self.ema_extract: Optional[float] = None
        self.alpha = 2.0 / (EMA_WINDOW + 1)
        self.start_session_day = START_SESSION_DAY

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td0 = _parse_td(state.traderData)
        if _D not in td0:
            td0[_D] = self.start_session_day
        out = _empty_orders()
        _tte_years(td0, int(state.timestamp))
        tte_years = float(td0[_TD])

        extract = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not extract or not extract.buy_orders or not extract.sell_orders:
            # No underlying book — cannot price VEV; still allow empty extract orders
            return out, 0, json.dumps(td0, separators=(",", ":"))

        s_mid = _mid(extract)
        if s_mid is None or s_mid <= 0:
            return out, 0, json.dumps(td0, separators=(",", ":"))

        if self.ema_extract is None:
            self.ema_extract = s_mid
        else:
            self.ema_extract = self.alpha * s_mid + (1.0 - self.alpha) * self.ema_extract

        for k in STRIKES:
            sym = f"VEV_{k}"
            if not TRADE_VEV_OPTIONS:
                out[sym] = []
                continue
            depth = state.order_depths.get(sym)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                out[sym] = []
                continue

            o_mid = _mid(depth)
            if o_mid is None:
                out[sym] = []
                continue

            intrinsic = max(0.0, s_mid - k)
            extrinsic = o_mid - intrinsic
            if extrinsic < MIN_EXTRINSIC:
                out[sym] = []
                continue

            m = k / s_mid
            v_hat = A * (m**2) + B * m + C
            v_hat = max(SIGMA_MIN, min(SIGMA_MAX, v_hat))

            fair_px, _ = bs_call(s_mid, float(k), tte_years, v_hat)
            pos = int(state.position.get(sym, 0))
            q_list: List[Order] = []
            best_ask = min(depth.sell_orders.keys())
            best_bid = max(depth.buy_orders.keys())

            if best_ask < fair_px - SCALP_THRESHOLD:
                bq = OPTION_LIMIT - pos
                if bq > 0:
                    q_list.append(Order(sym, int(best_ask), int(bq)))
            elif best_bid > fair_px + SCALP_THRESHOLD:
                sq = OPTION_LIMIT + pos
                if sq > 0:
                    q_list.append(Order(sym, int(best_bid), -int(sq)))
            out[sym] = q_list

        extract_pos = int(state.position.get("VELVETFRUIT_EXTRACT", 0))
        extract_orders: List[Order] = []
        s_best_bid = max(extract.buy_orders.keys())
        s_best_ask = min(extract.sell_orders.keys())

        if s_mid < self.ema_extract - EMA_THRESHOLD:
            bq = EXTRACT_LIMIT - extract_pos
            if bq > 0:
                extract_orders.append(Order("VELVETFRUIT_EXTRACT", int(s_best_ask), int(bq)))
        elif s_mid > self.ema_extract + EMA_THRESHOLD:
            sq = EXTRACT_LIMIT + extract_pos
            if sq > 0:
                extract_orders.append(Order("VELVETFRUIT_EXTRACT", int(s_best_bid), -int(sq)))
        out["VELVETFRUIT_EXTRACT"] = extract_orders

        return out, 0, json.dumps(td0, separators=(",", ":"))
