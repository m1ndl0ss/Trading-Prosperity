"""
Offline CSV replay and market time series for the local dashboard only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from datamodel import OrderDepth, TradingState


def read_prices_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def read_trades_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def row_to_depth(row: pd.Series) -> OrderDepth:
    """
    Build an OrderDepth from a prices CSV row.

    On the real exchange, **ask-side quantities are negative**; algos use
    expressions like ``min(-order_depth.sell_orders[best_ask], ...)`` to get a
    positive lift size. CSV columns store unsigned volumes, so we negate ask
    sizes here to match competition semantics (bid side stays positive).
    """
    d = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and int(bv) > 0:
            d.buy_orders[int(bp)] = int(bv)
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av) and int(av) > 0:
            d.sell_orders[int(ap)] = -int(av)
    return d


def best_bid_ask(depth: OrderDepth) -> Tuple[int, int]:
    if not depth.buy_orders or not depth.sell_orders:
        raise ValueError("book missing bid or ask level")
    return max(depth.buy_orders), min(depth.sell_orders)


def _load_td(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def fair_and_ema_from_traderdata(td: str) -> Tuple[Optional[float], Optional[float]]:
    snap = _load_td(td)
    if not snap:
        return None, None
    fair = float(snap["fair"]) if "fair" in snap else None
    ema = float(snap["ema"]) if "ema" in snap else None
    return fair, ema


def trades_by_timestamp(trades: pd.DataFrame, product: str) -> Dict[int, List[Tuple[int, int]]]:
    em = trades[trades["symbol"] == product].copy()
    if em.empty:
        return {}
    em["price"] = em["price"].astype(float)
    em["quantity"] = em["quantity"].astype(int)
    out: Dict[int, List[Tuple[int, int]]] = {}
    for _, r in em.iterrows():
        t = int(r["timestamp"])
        out.setdefault(t, []).append((int(r["price"]), int(r["quantity"])))
    return out


def market_frame(prices: pd.DataFrame, product: str) -> pd.DataFrame:
    sub = prices[prices["product"] == product].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values("timestamp")
    rows: List[Dict[str, Any]] = []
    for ts in sub["timestamp"].unique():
        row = sub.loc[sub["timestamp"] == ts].iloc[0]
        depth = row_to_depth(row)
        try:
            best_bid, best_ask = best_bid_ask(depth)
        except ValueError:
            continue
        mid = float(row["mid_price"]) if pd.notna(row.get("mid_price")) else (best_bid + best_ask) / 2.0
        bv = row.get("bid_volume_1")
        av = row.get("ask_volume_1")
        rows.append(
            {
                "timestamp": int(ts),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "bid_vol_1": int(bv) if pd.notna(bv) else None,
                "ask_vol_1": int(av) if pd.notna(av) else None,
            }
        )
    return pd.DataFrame(rows)


def replay_to_dataframe(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    product: str,
    trader_cls: Type,
    initial_cash: float = 0.0,
) -> pd.DataFrame:
    book = prices[prices["product"] == product].copy()
    if book.empty:
        return pd.DataFrame()
    book = book.sort_values("timestamp").set_index("timestamp")
    by_t = trades_by_timestamp(trades, product)

    trader = trader_cls()
    td = ""
    pos = 0
    cash = float(initial_cash)

    records: List[Dict[str, Any]] = []

    for ts in sorted(book.index.unique()):
        row = book.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        depth = row_to_depth(row)
        try:
            best_bid, best_ask = best_bid_ask(depth)
        except ValueError:
            continue

        mid = float(row["mid_price"]) if pd.notna(row.get("mid_price")) else (best_bid + best_ask) / 2.0
        bv = row.get("bid_volume_1")
        av = row.get("ask_volume_1")
        bid_vol_1 = int(bv) if pd.notna(bv) else None
        ask_vol_1 = int(av) if pd.notna(av) else None

        state = TradingState(
            td,
            int(ts),
            {},
            {product: depth},
            {},
            {},
            {product: pos},
            {},
        )
        orders_dict, _, td = trader.run(state)
        out = orders_dict.get(product, [])
        bid_o = next((o for o in out if o.quantity > 0), None)
        ask_o = next((o for o in out if o.quantity < 0), None)

        bid_left = bid_o.quantity if bid_o else 0
        ask_left = -ask_o.quantity if ask_o else 0
        buy_step = sell_step = 0

        for trade_price, trade_qty in by_t.get(int(ts), []):
            if bid_o and trade_price == best_bid and bid_o.price > best_bid:
                take = min(trade_qty, bid_left)
                if take:
                    pos += take
                    cash -= take * bid_o.price
                    bid_left -= take
                    buy_step += take
            elif bid_o and trade_price == best_ask and bid_o.price == best_ask:
                take = min(trade_qty, bid_left)
                if take:
                    pos += take
                    cash -= take * bid_o.price
                    bid_left -= take
                    buy_step += take
            elif ask_o and trade_price == best_ask and ask_o.price < best_ask:
                take = min(trade_qty, ask_left)
                if take:
                    pos -= take
                    cash += take * ask_o.price
                    ask_left -= take
                    sell_step += take
            elif ask_o and trade_price == best_bid and ask_o.price == best_bid:
                take = min(trade_qty, ask_left)
                if take:
                    pos -= take
                    cash += take * ask_o.price
                    ask_left -= take
                    sell_step += take

        fair, ema = fair_and_ema_from_traderdata(td)
        equity = cash + pos * mid
        sim_buy_px = float(bid_o.price) if bid_o and buy_step > 0 else float("nan")
        sim_sell_px = float(ask_o.price) if ask_o and sell_step > 0 else float("nan")

        records.append(
            {
                "timestamp": int(ts),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "bid_vol_1": bid_vol_1,
                "ask_vol_1": ask_vol_1,
                "fair": fair,
                "ema": ema,
                "bid_px": bid_o.price if bid_o else None,
                "ask_px": ask_o.price if ask_o else None,
                "bid_size": bid_o.quantity if bid_o else None,
                "ask_size": -ask_o.quantity if ask_o else None,
                "pos": pos,
                "cash": cash,
                "equity": equity,
                "buy_fills_step": buy_step,
                "sell_fills_step": sell_step,
                "sim_buy_px": sim_buy_px,
                "sim_sell_px": sim_sell_px,
                "sim_buy_vol": buy_step,
                "sim_sell_vol": sell_step,
            }
        )

    return pd.DataFrame(records)


def replay_summary(df: pd.DataFrame, initial_cash: float) -> Dict[str, Any]:
    if df.empty:
        return {
            "ending_pos": 0,
            "cash": float(initial_cash),
            "mark": 0,
            "equity": float(initial_cash),
            "pnl": 0.0,
            "buy_fills": 0,
            "sell_fills": 0,
        }
    last = df.iloc[-1]
    mark = int(last["mid"])
    equity = float(last["equity"])
    return {
        "ending_pos": int(last["pos"]),
        "cash": float(last["cash"]),
        "mark": mark,
        "equity": equity,
        "pnl": equity - float(initial_cash),
        "buy_fills": int(df["buy_fills_step"].sum()),
        "sell_fills": int(df["sell_fills_step"].sum()),
    }
