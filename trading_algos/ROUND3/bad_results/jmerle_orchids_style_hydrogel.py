"""
jmerle IMC Prosperity 2 `OrchidsStrategy` (Round 2), adapted to this repo.

Reference:
  https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round2.py

Each tick (original):
  1. Request conversion: ``convert(-1 * position)`` — flatten the current position
     over the cross-market / conversion channel.
  2. Read *foreign* (South archipelago) bid/ask and fees from
     ``state.observations.conversionObservations[symbol]``.
  3. ``buy_price = askPrice + transportFees + importTariff`` (acquisition cost
     to bring inventory in).
  4. Post a single **sell** (short) for ``limit`` at
     ``max(int(bidPrice - 0.5), int(buy_price + 1))`` so any fill is worth
     reconverting the short back at the next ask.

**Prosperity 3** `datamodel` uses ``observations: Dict[Product, int]`` — no
conversion struct in the stub. This implementation:

  - Tries a **P2-style** object on ``state.observations`` (attribute
    ``conversionObservations`` mapping product → object with
    ``bidPrice``, ``askPrice``, ``transportFees``, ``importTariff``) when present.
  - Otherwise, if ``use_book_as_conversion_proxy`` is True, uses the **local**
    best bid/ask as bid/ask stand-ins and ``PARAMS`` for fees (tunable for tape;
    does not model two-island reality).

Local tape replay in ``replay_engine`` must apply a non-zero ``conversions`` return
(see ``replay_to_dataframe``): we flatten the pre-tick position at ``mid`` before
matching orders.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

HYDROGEL_PACK = "HYDROGEL_PACK"

PARAMS: Dict[str, Any] = {
    # When true and no P2-style conversion obs: use L1 of this product’s book
    "use_book_as_conversion_proxy": True,
    "fee_transport": 0.0,
    "fee_import": 0.0,
}


def _as_float(x: Any) -> float:
    if x is None:
        return 0.0
    return float(x)


def _orchid_fundamentals(
    state: TradingState, product: str, depth: OrderDepth, p: Dict[str, Any]
) -> Optional[Dict[str, float]]:
    """Return bidPrice, askPrice, transportFees, importTariff or None if unusable."""
    obs_root = state.observations
    conv_map = getattr(obs_root, "conversionObservations", None)
    if conv_map is not None and product in conv_map:
        o = conv_map[product]

        def _f(name: str) -> float:
            v = o[name] if isinstance(o, dict) else getattr(o, name, None)
            return _as_float(v) if v is not None else 0.0

        return {
            "bidPrice": _f("bidPrice"),
            "askPrice": _f("askPrice"),
            "transportFees": _f("transportFees"),
            "importTariff": _f("importTariff"),
        }
    if p.get("use_book_as_conversion_proxy") and depth.buy_orders and depth.sell_orders:
        bb = float(max(depth.buy_orders.keys()))
        ba = float(min(depth.sell_orders.keys()))
        return {
            "bidPrice": bb,
            "askPrice": ba,
            "transportFees": float(p.get("fee_transport", 0.0)),
            "importTariff": float(p.get("fee_import", 0.0)),
        }
    return None


class Trader:
    """Orchids-style: convert to flat, then one aggressive short offer to limit — HYDROGEL only."""

    LIMIT: Dict[str, int] = {HYDROGEL_PACK: 200}

    def __init__(self) -> None:
        self.params = dict(PARAMS)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        p = self.params
        result: Dict[str, List[Order]] = {sym: [] for sym in state.order_depths}
        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except (json.JSONDecodeError, TypeError):
                td = {}

        product = HYDROGEL_PACK
        if product not in state.order_depths:
            return result, 0, json.dumps(td, separators=(",", ":"))

        pos = int(state.position.get(product, 0))
        # jmerle: convert(-1 * position) — return value to engine is the requested amount
        conversions = -pos

        depth = state.order_depths[product]
        f = _orchid_fundamentals(state, product, depth, p)
        if f is None:
            return result, conversions, json.dumps(td, separators=(",", ":"))

        buy_price = f["askPrice"] + f["transportFees"] + f["importTariff"]
        sell_px = max(int(f["bidPrice"] - 0.5), int(buy_price + 1.0))
        lim = self.LIMIT[product]
        result[product] = [Order(product, int(sell_px), -lim)]
        return result, conversions, json.dumps(td, separators=(",", ":"))
