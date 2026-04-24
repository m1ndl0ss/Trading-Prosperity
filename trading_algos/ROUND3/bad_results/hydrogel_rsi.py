"""
HYDROGEL_PACK only — simple RSI regime quotes (low-frequency friendly).

- Track mid = (best_bid + best_ask) / 2, append one sample per `run` call.
- RSI(period) from the last (period) consecutive price changes (simple avg gain/loss).
- **Buy** limit orders: only while RSI < `oversold` (default 35). Pull buys when RSI
  is no longer below (>= oversold) — you only bid in the oversold zone.
- **Sell** limit orders: only while RSI > `overbought` (default 65). Pull sells when
  RSI is no longer above (<= overbought).

Default pricing uses **limit-at-touch** (buy @ best_bid, sell @ best_ask) as *offers*.

The repo's `replay_engine` only simulates **aggressive** fills (buy at best_ask, sell
at best_bid).  Set `aggressive: true` in `PARAMS` to match that and see non-zero
tape PnL locally; on the real exchange, passive offers can fill and you typically
set `aggressive: false`.  Position limit 200.

Tune `PARAMS` (period / thresholds) without touching the rest.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"
_TD_MIDS = "mids"
_MAX_MIDS = 500  # cap JSON size; we only need ~period+1

PARAMS: Dict[str, Any] = {
    "rsi_period": 24,
    "oversold": 35.0,
    "overbought": 65.0,
    # False: pass buy@best_bid, sell@best_ask (offers, live-like). True: lift/hit; needed
    # for this repo's tape replay to register fills (see module doc).
    "aggressive": True,
}


def _rolling_rsi(mids: List[float], period: int) -> Optional[float]:
    """
    Simple RSI from the last `period` one-step price changes
    (need len(mids) >= period + 1).
    """
    n = len(mids)
    if n < period + 1 or period < 1:
        return None
    window = mids[-(period + 1) :]
    diffs = [window[i + 1] - window[i] for i in range(period)]
    if not diffs:
        return None
    gains = [max(0.0, d) for d in diffs]
    losses = [max(0.0, -d) for d in diffs]
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l < 1e-12:
        if avg_g < 1e-12:
            return 50.0
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _mid_price(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders.keys())
    ba = min(depth.sell_orders.keys())
    return (bb + ba) / 2.0


class Trader:
    """RSI-based quoting on hydrogel only; other products: []."""

    LIMIT: Dict[str, int] = {HYDROGEL_PACK: 200}

    def __init__(self) -> None:
        self.params = dict(PARAMS)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        mids: List[float] = []
        for x in td.get(_TD_MIDS, []):
            try:
                mids.append(float(x))
            except (TypeError, ValueError):
                continue

        period = int(self.params.get("rsi_period", 24))
        overs = float(self.params.get("oversold", 35.0))
        ob = float(self.params.get("overbought", 65.0))
        aggr = bool(self.params.get("aggressive", False))

        result: Dict[str, List[Order]] = {}
        for product in state.order_depths:
            if product != HYDROGEL_PACK:
                result[product] = []
                continue

            depth = state.order_depths[product]
            m = _mid_price(depth)
            if m is None:
                result[product] = []
                continue

            mids.append(m)
            if len(mids) > _MAX_MIDS:
                mids = mids[-_MAX_MIDS:]

            rsi = _rolling_rsi(mids, period)
            pos = int(state.position.get(HYDROGEL_PACK, 0))
            lim = self.LIMIT[HYDROGEL_PACK]
            orders: List[Order] = []

            if rsi is not None and depth.buy_orders and depth.sell_orders:
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                # Passive: post at touch. Aggressive: buy lifts ask, sell hits bid
                # (see module doc: replay only matches aggressive in this repo).
                buy_px, sell_px = (best_ask, best_bid) if aggr else (best_bid, best_ask)

                if rsi < overs:
                    bq = lim - pos
                    if bq > 0:
                        orders.append(Order(HYDROGEL_PACK, int(buy_px), bq))
                if rsi > ob:
                    sq = lim + pos
                    if sq > 0:
                        orders.append(Order(HYDROGEL_PACK, int(sell_px), -sq))

            result[product] = orders

        td[_TD_MIDS] = mids
        return result, conversions, json.dumps(td, separators=(",", ":"))
