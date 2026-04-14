from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np

from datamodel import Order, TradingState

TOMATOES = "TOMATOES"
EMERALDS = "EMERALDS"

TD_TOMATOES = "tomatoes"

# Emeralds ~ static fair-value-based market making
POSITION_LIMIT_EMERALDS = 80
FAIR = 10_000
SPREAD_HALF = 7

# Tomatoes ~ rolling linear forecast + regression-style quoting
LAG = 4
FIT_MIN_ROWS = 12
FIT_TAIL = 80
BAND = 1
POSITION_LIMIT_TOMATOES = 20
ORDER_SIZE_TOMATOES = 20
ALLOW_SHORT_SELL = True


def _predict_next_mid(mids: List[float]) -> float | None:
    if len(mids) <= LAG:
        return None
    start = max(LAG, len(mids) - FIT_TAIL)
    rows_x: List[List[float]] = []
    rows_y: List[float] = []
    for i in range(start, len(mids)):
        rows_x.append([1.0, *mids[i - LAG : i]])
        rows_y.append(mids[i])
    if len(rows_y) < FIT_MIN_ROWS:
        return None
    x = np.asarray(rows_x, dtype=float)
    y = np.asarray(rows_y, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    x_next = np.asarray([1.0, *mids[-LAG:]], dtype=float)
    return float(x_next @ beta)


def _compute_orders_regression(
    symbol: str,
    buy_orders: Dict[int, int],
    sell_orders: Dict[int, int],
    acc_bid: int,
    acc_ask: int,
    pos: int,
    limit: int,
    quote_cap: int,
    allow_short: bool,
) -> List[Order]:
    orders: List[Order] = []
    if not buy_orders or not sell_orders:
        return orders

    osell = dict(sorted(sell_orders.items()))
    obuy = dict(sorted(buy_orders.items(), reverse=True))

    best_sell = min(sell_orders)
    best_buy = max(buy_orders)

    cpos = pos

    for ask, vol in osell.items():
        if (
            (ask <= acc_bid or (cpos < 0 and ask == acc_bid + 1))
            and cpos < limit
        ):
            take = min(vol, limit - cpos)
            if take <= 0:
                continue
            cpos += take
            orders.append(Order(symbol, ask, take))

    undercut_buy = best_buy + 1
    undercut_sell = best_sell - 1
    bid_pr = min(undercut_buy, acc_bid)
    sell_pr = max(undercut_sell, acc_ask)

    q_room = min(quote_cap, limit - cpos)
    if cpos < limit and q_room > 0:
        orders.append(Order(symbol, bid_pr, q_room))
        cpos += q_room

    cpos = pos
    for bid, vol in obuy.items():
        if (
            (bid >= acc_ask or (cpos > 0 and bid + 1 == acc_ask))
            and cpos > -limit
            and (allow_short or cpos > 0)
        ):
            cap_out = (cpos + limit) if allow_short else max(cpos, 0)
            give = min(vol, cap_out)
            if give <= 0:
                continue
            cpos -= give
            orders.append(Order(symbol, bid, -give))

    if allow_short:
        q_room = min(quote_cap, cpos + limit)
        if cpos > -limit and q_room > 0:
            orders.append(Order(symbol, sell_pr, -q_room))
    else:
        q_room = min(quote_cap, max(cpos, 0))
        if cpos > 0 and q_room > 0:
            orders.append(Order(symbol, sell_pr, -q_room))

    return orders


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        conversions = 0
        try:
            trader_data_dict: Dict[str, Any] = (
                json.loads(state.traderData) if state.traderData else {}
            )
        except (json.JSONDecodeError, TypeError):
            trader_data_dict = {}

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []

            # Emeralds
            # Assumes a fixed fair of 10,000 and posts a single bid at 9,993 and ask at 10,007, 
            # scaling each order so inventory stays within ±80; no extra state is stored in traderData.
            if product == EMERALDS:
                if order_depth.buy_orders and order_depth.sell_orders:
                    bid_px = FAIR - SPREAD_HALF
                    ask_px = FAIR + SPREAD_HALF
                    pos = state.position.get(EMERALDS, 0)
                    bid_vol = POSITION_LIMIT_EMERALDS - pos
                    ask_vol = -POSITION_LIMIT_EMERALDS - pos
                    if bid_vol > 0:
                        orders.append(Order(EMERALDS, bid_px, bid_vol))
                    if ask_vol < 0:
                        orders.append(Order(EMERALDS, ask_px, ask_vol))

            # Tomatoes
            # Appends each mid to traderData["tomatoes"], refits a short AR(4) forecast with
            # rolling least squares, rounds it to a one-tick fair band (pred ± 1), then lifts cheap asks, hits rich
            # bids, and rests improving quotes at the touch—optionally including shorts—capped at 20 lots.
            elif product == TOMATOES:
                td_t: Dict[str, Any] = dict(trader_data_dict.get(TD_TOMATOES, {}))
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders)
                    best_ask = min(order_depth.sell_orders.keys())
                    mid = (best_bid + best_ask) / 2.0

                    mids: List[float] = list(td_t.get("mids", []))
                    mids.append(mid)
                    mids = mids[-800:]

                    pred = _predict_next_mid(mids)
                    pos = state.position.get(TOMATOES, 0)

                    if pred is None:
                        td_t = {"mids": mids}
                    else:
                        pred_i = int(round(pred))
                        acc_bid = pred_i - BAND
                        acc_ask = pred_i + BAND
                        orders = _compute_orders_regression(
                            TOMATOES,
                            order_depth.buy_orders,
                            order_depth.sell_orders,
                            acc_bid,
                            acc_ask,
                            pos,
                            POSITION_LIMIT_TOMATOES,
                            ORDER_SIZE_TOMATOES,
                            ALLOW_SHORT_SELL,
                        )
                        td_t = {
                            "mids": mids,
                            "pred": pred_i,
                            "acc_bid": acc_bid,
                            "acc_ask": acc_ask,
                        }
                    trader_data_dict[TD_TOMATOES] = td_t

            result[product] = orders

        new_trader_data = json.dumps(trader_data_dict, separators=(",", ":"))
        return result, conversions, new_trader_data
