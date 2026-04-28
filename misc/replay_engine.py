"""
Offline CSV replay and market time series for the local dashboard only.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Type

import pandas as pd

from datamodel import OrderDepth, Trade, TradingState


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


def book_levels_from_row(row: pd.Series) -> Dict[str, Any]:
    """
    Extract L1–L3 bid/ask prices and unsigned volumes from a prices CSV row
    (for dashboard / replay DataFrame columns).
    """
    out: Dict[str, Any] = {}
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        out[f"bid_px_{i}"] = int(bp) if pd.notna(bp) else None
        out[f"bid_vol_{i}"] = int(bv) if pd.notna(bv) else None
        out[f"ask_px_{i}"] = int(ap) if pd.notna(ap) else None
        out[f"ask_vol_{i}"] = int(av) if pd.notna(av) else None
    return out


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


def _uid(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and not x.strip()):
        return None
    return str(x).strip()


def market_trades_by_timestamp(trades: pd.DataFrame, product: str) -> Dict[int, List[Trade]]:
    """Build ``Trade`` list per timestamp for a symbol (e.g. HYDROGEL) for counterparty-aware algos."""
    em = trades[trades["symbol"] == product].copy()
    if em.empty:
        return {}
    out: Dict[int, List[Trade]] = {}
    for _, r in em.iterrows():
        ts = int(r["timestamp"])
        pr = int(float(r["price"]))
        qty = int(r["quantity"])
        bu = _uid(r["buyer"]) if "buyer" in r.index else None
        se = _uid(r["seller"]) if "seller" in r.index else None
        tr = Trade(product, pr, qty, buyer=bu, seller=se, timestamp=ts)
        out.setdefault(ts, []).append(tr)
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
        row_dict: Dict[str, Any] = {
            "timestamp": int(ts),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "bid_vol_1": int(bv) if pd.notna(bv) else None,
            "ask_vol_1": int(av) if pd.notna(av) else None,
        }
        row_dict.update(book_levels_from_row(row))
        rows.append(row_dict)
    return pd.DataFrame(rows)


def replay_to_dataframe(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    product: str,
    trader_cls: Type,
    initial_cash: float = 0.0,
    inject_market_trades: bool = False,
) -> pd.DataFrame:
    book = prices[prices["product"] == product].copy()
    if book.empty:
        return pd.DataFrame()
    book = book.sort_values("timestamp").set_index("timestamp")
    by_t = trades_by_timestamp(trades, product)
    mkt_t: Optional[Dict[int, List[Trade]]] = None
    if inject_market_trades:
        mkt_t = market_trades_by_timestamp(trades, product)

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

        if mkt_t is not None:
            mt_list = mkt_t.get(int(ts), [])
            market_trades: Dict[str, List[Trade]] = {product: mt_list}
        else:
            market_trades = {}

        state = TradingState(
            td,
            int(ts),
            {},
            {product: depth},
            {},
            market_trades,
            {product: pos},
            {},
        )
        orders_dict, conv, td = trader.run(state)
        # jmerle Orchids-style: non-zero `conversions` flattens pre-tick position
        # before orders (e.g. convert(-pos)); mark at mid in replay when fees unknown.
        if conv != 0:
            cash += pos * mid
            pos = 0
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

        rec: Dict[str, Any] = {
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
        rec.update(book_levels_from_row(row))
        records.append(rec)

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


def replay_path_metrics(df: pd.DataFrame, initial_cash: float) -> Dict[str, Any]:
    """PnL, fills, max drawdown of equity, max |inventory|, total execution volume (tape replay)."""
    s = replay_summary(df, initial_cash)
    if df.empty:
        return {**s, "max_drawdown": 0.0, "max_abs_pos": 0, "turnover_fills": 0}
    eq = df["equity"].astype(float)
    peak = eq.cummax()
    mdd = float((peak - eq).max())
    turn = int(df["buy_fills_step"].sum() + df["sell_fills_step"].sum())
    return {
        **s,
        "max_drawdown": mdd,
        "max_abs_pos": int(df["pos"].abs().max()),
        "turnover_fills": turn,
    }


def _passive_fills_one_product(
    best_bid: int,
    best_ask: int,
    tape_at_ts: List[Tuple[int, int]],
    out: List,
    pos: int,
) -> Tuple[int, float, int, int]:
    """
    Orchids-style passive match vs tape (same rules as ``replay_to_dataframe``).
    Returns new position, cash delta, buy fill count, sell fill count.
    """
    bid_o = next((o for o in out if o.quantity > 0), None)
    ask_o = next((o for o in out if o.quantity < 0), None)
    bid_left = bid_o.quantity if bid_o else 0
    ask_left = -ask_o.quantity if ask_o else 0
    cash_d = 0.0
    buy_step = sell_step = 0
    for trade_price, trade_qty in tape_at_ts:
        if (
            bid_o
            and trade_price == best_bid
            and bid_o.price > best_bid
        ):
            take = min(trade_qty, bid_left)
            if take:
                pos += take
                cash_d -= take * bid_o.price
                bid_left -= take
                buy_step += take
        elif (
            bid_o
            and trade_price == best_ask
            and bid_o.price == best_ask
        ):
            take = min(trade_qty, bid_left)
            if take:
                pos += take
                cash_d -= take * bid_o.price
                bid_left -= take
                buy_step += take
        elif (
            ask_o
            and trade_price == best_ask
            and ask_o.price < best_ask
        ):
            take = min(trade_qty, ask_left)
            if take:
                pos -= take
                cash_d += take * ask_o.price
                ask_left -= take
                sell_step += take
        elif (
            ask_o
            and trade_price == best_bid
            and ask_o.price == best_bid
        ):
            take = min(trade_qty, ask_left)
            if take:
                pos -= take
                cash_d += take * ask_o.price
                ask_left -= take
                sell_step += take
    return pos, cash_d, buy_step, sell_step


def replay_portfolio_symbol_pnls(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    trader_cls: Type,
    initial_cash: float = 0.0,
    inject_market_trades: bool = True,
) -> Tuple[
    Dict[str, float],
    Dict[str, int],
    float,
    int,
    int,
    Dict[str, Tuple[int, int]],
    Dict[str, int],
]:
    """
    One trader instance, all products each tick (shared ``traderData`` / EMA).

    Returns
    -------
    pnl_by_symbol, ending_pos, total_equity, total_buy_fills, total_sell_fills, fills_by_symbol
        Fills: ``symbol -> (buy_lots, sell_lots)`` (lots like single-product replay).
    order_non_empty_ticks
        Count of timestamps where the trader returned a non-empty order list for that symbol
        (intent to quote / trade, independent of whether passive fill model matched tape).
    """
    if prices.empty or "product" not in prices.columns:
        return {}, {}, float(initial_cash), 0, 0, {}, {}, {}
    products = sorted(prices["product"].dropna().astype(str).unique().tolist())
    books: Dict[str, Any] = {}
    for p in products:
        b = prices[prices["product"] == p].copy().sort_values("timestamp").set_index("timestamp")
        books[p] = b
    by_t: Dict[str, Dict[int, List[Tuple[int, int]]]] = {
        p: trades_by_timestamp(trades, p) for p in products
    }
    mkt_by_p: Optional[Dict[str, Dict[int, List[Trade]]]] = None
    if inject_market_trades:
        mkt_by_p = {p: market_trades_by_timestamp(trades, p) for p in products}
    positions: Dict[str, int] = {p: 0 for p in products}
    cf: Dict[str, float] = {p: 0.0 for p in products}
    cash = float(initial_cash)
    trader = trader_cls()
    td = ""
    all_ts = sorted(prices["timestamp"].unique().tolist())
    total_buy = total_sell = 0
    fills_bu: Dict[str, int] = {p: 0 for p in products}
    fills_su: Dict[str, int] = {p: 0 for p in products}
    order_tick_n: DefaultDict[str, int] = defaultdict(int)

    for ts in all_ts:
        order_depths: Dict[str, OrderDepth] = {}
        meta: Dict[str, Tuple[int, int, float]] = {}
        for p in products:
            if ts not in books[p].index:
                continue
            row = books[p].loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            try:
                depth = row_to_depth(row)
                best_bid, best_ask = best_bid_ask(depth)
            except ValueError:
                continue
            mid = float(row["mid_price"]) if pd.notna(row.get("mid_price")) else (
                (best_bid + best_ask) / 2.0
            )
            order_depths[p] = depth
            meta[p] = (best_bid, best_ask, mid)
        if not order_depths:
            continue
        market_trades: Dict[str, List[Trade]] = {}
        if mkt_by_p is not None:
            for p in order_depths:
                market_trades[p] = mkt_by_p[p].get(int(ts), [])

        state = TradingState(
            td,
            int(ts),
            {},
            order_depths,
            {},
            market_trades,
            {sym: positions.get(sym, 0) for sym in products},
            {},
        )
        orders_dict, conv, td = trader.run(state)
        if conv != 0:
            for p in list(order_depths.keys()):
                if p not in meta:
                    continue
                _bb, _ba, mmu = meta[p]
                cash += positions[p] * mmu
                cf[p] += positions[p] * mmu
                positions[p] = 0

        for p in order_depths:
            if list(orders_dict.get(p) or []):
                order_tick_n[p] += 1

        for p in order_depths:
            best_bid, best_ask, _mid = meta[p]
            tape_list = by_t[p].get(int(ts), [])
            out = list(orders_dict.get(p) or [])
            pos0 = positions[p]
            pos1, d_cash, bst, sst = _passive_fills_one_product(
                best_bid, best_ask, tape_list, out, pos0
            )
            positions[p] = pos1
            cash += d_cash
            cf[p] += d_cash
            total_buy += bst
            total_sell += sst
            fills_bu[p] += bst
            fills_su[p] += sst

    pnl_by_symbol: Dict[str, float] = {}
    ending: Dict[str, int] = {}
    for p in products:
        ending[p] = int(positions[p])
        # mark at last available mid in book
        b = books[p]
        if b.empty:
            pnl_by_symbol[p] = float(cf[p])
            continue
        last_ts = int(b.index.max())
        row = b.loc[last_ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        try:
            dlast = row_to_depth(row)
            bb, ba = best_bid_ask(dlast)
            mk = float(row["mid_price"]) if pd.notna(row.get("mid_price")) else (bb + ba) / 2.0
        except ValueError:
            mk = 0.0
        pnl_by_symbol[p] = float(cf[p] + positions[p] * mk)

    total_eq = float(cash)
    for p in products:
        b = books[p]
        if b.empty:
            continue
        last_ts = int(b.index.max())
        row = b.loc[last_ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        try:
            dlast = row_to_depth(row)
            bb, ba = best_bid_ask(dlast)
            mk = float(row["mid_price"]) if pd.notna(row.get("mid_price")) else (bb + ba) / 2.0
        except ValueError:
            mk = 0.0
        total_eq += positions[p] * mk
    fmap = {p: (fills_bu[p], fills_su[p]) for p in products}
    on = {p: int(order_tick_n[p]) for p in products}
    return pnl_by_symbol, ending, total_eq, total_buy, total_sell, fmap, on
