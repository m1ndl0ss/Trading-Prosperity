#!/usr/bin/env python3
"""
HYDROGEL_PACK alpha research report.

The script keeps research separate from production traders:

* signal tables from the raw ROUND3 tape
* Sharpe-like replay metrics from existing equity curves
* baseline and candidate strategy comparisons
* leave-one-day-out validation for simple threshold families
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Type

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datamodel import Order, OrderDepth, TradingState
from replay_engine import read_prices_csv, read_trades_csv, replay_to_dataframe
from tape_benchmark import discover_days, load_trader

PRODUCT = "HYDROGEL_PACK"
LIMIT = 200


def _best_bid_ask(depth: OrderDepth) -> Tuple[int, int] | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return max(depth.buy_orders), min(depth.sell_orders)


def _mid(depth: OrderDepth) -> float | None:
    bba = _best_bid_ask(depth)
    if bba is None:
        return None
    bb, ba = bba
    return (bb + ba) / 2.0


class FixedThresholdTrader:
    """Aggressive threshold baseline: buy low, sell high."""

    buy_below = 9940.0
    sell_above = 10050.0
    clip = 25

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        depth = state.order_depths.get(PRODUCT)
        if depth is None:
            return {}, 0, state.traderData
        bba = _best_bid_ask(depth)
        m = _mid(depth)
        if bba is None or m is None:
            return {PRODUCT: []}, 0, state.traderData
        best_bid, best_ask = bba
        pos = int(state.position.get(PRODUCT, 0))
        orders: List[Order] = []
        if m <= self.buy_below and pos < LIMIT:
            orders.append(Order(PRODUCT, best_ask, min(self.clip, LIMIT - pos)))
        if m >= self.sell_above and pos > -LIMIT:
            orders.append(Order(PRODUCT, best_bid, -min(self.clip, LIMIT + pos)))
        return {PRODUCT: orders}, 0, state.traderData


class InsideSpreadTrader:
    """Local-replay passive proxy: improve one tick inside both sides."""

    clip = 12

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        depth = state.order_depths.get(PRODUCT)
        bba = _best_bid_ask(depth) if depth is not None else None
        if bba is None:
            return {PRODUCT: []}, 0, state.traderData
        best_bid, best_ask = bba
        pos = int(state.position.get(PRODUCT, 0))
        orders: List[Order] = []
        if best_bid + 1 < best_ask and pos < LIMIT:
            orders.append(Order(PRODUCT, best_bid + 1, min(self.clip, LIMIT - pos)))
        if best_ask - 1 > best_bid and pos > -LIMIT:
            orders.append(Order(PRODUCT, best_ask - 1, -min(self.clip, LIMIT + pos)))
        return {PRODUCT: orders}, 0, state.traderData


class SlowReversionTrader:
    """Inventory-aware EMA deviation candidate used for walk-forward search."""

    alpha = 0.002
    band = 35.0
    exit_band = 5.0
    clip = 25
    soft_limit = 120

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        depth = state.order_depths.get(PRODUCT)
        bba = _best_bid_ask(depth) if depth is not None else None
        m = _mid(depth) if depth is not None else None
        if bba is None or m is None:
            return {PRODUCT: []}, 0, state.traderData

        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                raw = json.loads(state.traderData)
                if isinstance(raw, dict):
                    td = raw
            except json.JSONDecodeError:
                td = {}
        ema = float(td.get("ema", m))
        ema = (1.0 - self.alpha) * ema + self.alpha * m
        td["ema"] = ema

        best_bid, best_ask = bba
        dev = m - ema
        pos = int(state.position.get(PRODUCT, 0))
        orders: List[Order] = []

        if dev <= -self.band and pos < self.soft_limit:
            orders.append(Order(PRODUCT, best_ask, min(self.clip, LIMIT - pos)))
        elif dev >= self.band and pos > -self.soft_limit:
            orders.append(Order(PRODUCT, best_bid, -min(self.clip, LIMIT + pos)))

        if pos > 0 and dev >= -self.exit_band:
            orders.append(Order(PRODUCT, best_bid, -min(abs(pos), self.clip)))
        elif pos < 0 and dev <= self.exit_band:
            orders.append(Order(PRODUCT, best_ask, min(abs(pos), self.clip)))

        return {PRODUCT: orders}, 0, json.dumps(td, separators=(",", ":"))


def configured_trader(base: Type, **params: Any) -> Type:
    name = base.__name__ + "_" + "_".join(f"{k}{v}" for k, v in params.items())
    return type(name, (base,), params)


def load_product_tape(data_dir: Path, product: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for day_key, prices_path, _trades_path in discover_days(data_dir):
        df = read_prices_csv(prices_path)
        sub = df[df["product"] == product].copy()
        if sub.empty:
            continue
        sub["day_key"] = str(day_key)
        frames.append(sub)
    if not frames:
        raise SystemExit(f"No {product} rows found under {data_dir}")
    return pd.concat(frames, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)


def add_trade_flow_features(tape: pd.DataFrame, data_dir: Path, product: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for day_key, _prices_path, trades_path in discover_days(data_dir):
        trades = read_trades_csv(trades_path)
        sub = trades[trades["symbol"] == product].copy()
        if sub.empty:
            continue
        sub["day_key"] = str(day_key)
        grp = sub.groupby(["day_key", "timestamp"], as_index=False).agg(
            trade_qty=("quantity", "sum"),
            trade_count=("quantity", "size"),
            trade_vwap=("price", lambda s: float(np.average(s, weights=sub.loc[s.index, "quantity"]))),
        )
        frames.append(grp)

    out = tape.copy()
    if frames:
        flow = pd.concat(frames, ignore_index=True)
        out = out.merge(flow, on=["day_key", "timestamp"], how="left")
    else:
        out["trade_qty"] = np.nan
        out["trade_count"] = np.nan
        out["trade_vwap"] = np.nan

    out["trade_qty"] = out["trade_qty"].fillna(0).astype(float)
    out["trade_count"] = out["trade_count"].fillna(0).astype(float)
    out["trade_vwap"] = out["trade_vwap"].fillna(out["mid_price"]).astype(float)
    out["trade_px_dist"] = out["trade_vwap"] - out["mid_price"].astype(float)
    out["trade_qty_20"] = out.groupby("day_key")["trade_qty"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    out["trade_px_dist_20"] = out.groupby("day_key")["trade_px_dist"].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )
    return out


def enrich_features(tape: pd.DataFrame) -> pd.DataFrame:
    out = tape.copy()
    out["mid"] = out["mid_price"].astype(float)
    out["spread"] = (out["ask_price_1"] - out["bid_price_1"]).astype(float)
    bid_vol = out["bid_volume_1"].astype(float)
    ask_vol = out["ask_volume_1"].astype(float)
    out["imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol).replace(0, np.nan)
    for alpha in (0.002, 0.005, 0.01, 0.03):
        col = f"ema_{alpha:g}"
        out[col] = out.groupby("day_key")["mid"].transform(lambda s, a=alpha: s.ewm(alpha=a, adjust=False).mean())
        out[f"dev_{alpha:g}"] = out["mid"] - out[col]
    lo = out.groupby("day_key")["mid"].cummin()
    hi = out.groupby("day_key")["mid"].cummax()
    out["range_pos"] = (out["mid"] - lo) / (hi - lo).replace(0, np.nan)
    out["ret_1"] = out.groupby("day_key")["mid"].diff()
    for horizon in (10, 25, 50, 100):
        out[f"fwd_{horizon}"] = out.groupby("day_key")["mid"].shift(-horizon) - out["mid"]
    return out


def print_signal_table(df: pd.DataFrame, feature: str, horizon: int, bins: int = 8) -> None:
    use = df[[feature, f"fwd_{horizon}", "spread"]].dropna().copy()
    if use.empty or use[feature].nunique() < 2:
        return
    use["bin"] = pd.qcut(use[feature], bins, duplicates="drop")
    grouped = use.groupby("bin", observed=True).agg(
        n=(feature, "size"),
        feature_mid=(feature, "mean"),
        fwd_mean=(f"fwd_{horizon}", "mean"),
        fwd_std=(f"fwd_{horizon}", "std"),
        up_rate=(f"fwd_{horizon}", lambda s: float((s > 0).mean())),
        half_spread=("spread", lambda s: float(s.mean() / 2.0)),
    )
    grouped["mean_abs_after_half_spread"] = grouped["fwd_mean"].abs() - grouped["half_spread"]
    print(f"\n=== {feature} -> {horizon}-tick forward mid move ===")
    print(grouped.round(3).to_string())


def equity_metrics(df: pd.DataFrame, initial_cash: float = 0.0) -> Dict[str, float]:
    if df.empty:
        return {
            "pnl": 0.0,
            "sharpe_tick": 0.0,
            "max_drawdown": 0.0,
            "fills": 0.0,
            "pnl_per_fill": 0.0,
            "limit_pct": 0.0,
            "max_abs_pos": 0.0,
        }
    equity = df["equity"].astype(float)
    pnl = float(equity.iloc[-1] - initial_cash)
    diff = equity.diff().fillna(0.0)
    vol = float(diff.std(ddof=0))
    sharpe = 0.0 if vol < 1e-12 else float(diff.mean() / vol * math.sqrt(len(diff)))
    drawdown = equity.cummax() - equity
    fills = float(df["buy_fills_step"].sum() + df["sell_fills_step"].sum())
    max_abs_pos = float(df["pos"].abs().max())
    return {
        "pnl": pnl,
        "sharpe_tick": sharpe,
        "max_drawdown": float(drawdown.max()),
        "fills": fills,
        "pnl_per_fill": 0.0 if fills <= 0 else pnl / fills,
        "limit_pct": float((df["pos"].abs() >= LIMIT).mean()),
        "max_abs_pos": max_abs_pos,
    }


def replay_day(trader_cls: Type, prices_path: Path, trades_path: Path, product: str) -> Dict[str, float]:
    prices = read_prices_csv(prices_path)
    trades = read_trades_csv(trades_path)
    df = replay_to_dataframe(prices, trades, product, trader_cls, initial_cash=0.0)
    return equity_metrics(df)


def replay_many(data_dir: Path, product: str, trader_cls: Type) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for day_key, prices_path, trades_path in discover_days(data_dir):
        metrics = replay_day(trader_cls, prices_path, trades_path, product)
        metrics["day"] = str(day_key)
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_replay(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "name": name,
        "total_pnl": float(df["pnl"].sum()),
        "mean_day_pnl": float(df["pnl"].mean()),
        "worst_day_pnl": float(df["pnl"].min()),
        "mean_sharpe_tick": float(df["sharpe_tick"].mean()),
        "max_drawdown": float(df["max_drawdown"].max()),
        "fills": float(df["fills"].sum()),
        "pnl_per_fill": float(df["pnl"].sum() / df["fills"].sum()) if df["fills"].sum() > 0 else 0.0,
        "limit_pct": float(df["limit_pct"].mean()),
        "max_abs_pos": float(df["max_abs_pos"].max()),
    }
    return out


def baseline_results(data_dir: Path, product: str, trader_paths: Sequence[Path]) -> pd.DataFrame:
    candidates: List[Tuple[str, Type]] = [
        ("fixed_9940_10050", FixedThresholdTrader),
        ("inside_spread_proxy", InsideSpreadTrader),
    ]
    for path in trader_paths:
        if path.is_file():
            candidates.append((path.stem, load_trader(path)))

    rows: List[Dict[str, Any]] = []
    for name, cls in candidates:
        day_df = replay_many(data_dir, product, cls)
        rows.append(summarize_replay(name, day_df))
    return pd.DataFrame(rows).sort_values(["total_pnl", "worst_day_pnl"], ascending=False)


def walk_forward_search(data_dir: Path, product: str) -> pd.DataFrame:
    # Keep this intentionally small: full replay is Python-loop heavy, and this
    # search is for robustness checks after the signal report, not max fitting.
    alphas = (0.001, 0.002)
    bands = (28.0, 36.0, 44.0)
    clips = (25,)
    soft_limits = (120,)
    days = discover_days(data_dir)
    rows: List[Dict[str, Any]] = []

    for holdout_idx, (holdout_day, _p, _t) in enumerate(days):
        train_scores: List[Tuple[float, float, float, Type, Dict[str, Any]]] = []
        for alpha in alphas:
            for band in bands:
                for clip in clips:
                    for soft_limit in soft_limits:
                        cls = configured_trader(
                            SlowReversionTrader,
                            alpha=alpha,
                            band=band,
                            exit_band=max(4.0, band * 0.15),
                            clip=clip,
                            soft_limit=soft_limit,
                        )
                        train_day_rows = []
                        for idx, (_day_key, prices_path, trades_path) in enumerate(days):
                            if idx == holdout_idx:
                                continue
                            train_day_rows.append(replay_day(cls, prices_path, trades_path, product))
                        train = pd.DataFrame(train_day_rows)
                        score = float(train["pnl"].sum())
                        worst = float(train["pnl"].min())
                        dd = float(train["max_drawdown"].max())
                        train_scores.append((score, worst, -dd, cls, cls.__dict__.copy()))
        train_scores.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_cls = train_scores[0][3]
        test_metrics = replay_day(best_cls, days[holdout_idx][1], days[holdout_idx][2], product)
        rows.append(
            {
                "holdout_day": str(holdout_day),
                "train_pnl": train_scores[0][0],
                "train_worst_day": train_scores[0][1],
                "test_pnl": test_metrics["pnl"],
                "test_sharpe_tick": test_metrics["sharpe_tick"],
                "test_max_drawdown": test_metrics["max_drawdown"],
                "alpha": best_cls.alpha,
                "band": best_cls.band,
                "exit_band": best_cls.exit_band,
                "clip": best_cls.clip,
                "soft_limit": best_cls.soft_limit,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="HYDROGEL_PACK signal, replay, and walk-forward report")
    ap.add_argument("--data", type=Path, default=ROOT / "Data" / "ROUND3")
    ap.add_argument("--product", default=PRODUCT)
    ap.add_argument("--skip-search", action="store_true", help="Skip leave-one-day-out grid search")
    args = ap.parse_args()

    tape = enrich_features(add_trade_flow_features(load_product_tape(args.data, args.product), args.data, args.product))

    print(f"Product: {args.product}")
    print(f"Data:    {args.data}")
    print("\n=== Mid summary by day ===")
    print(tape.groupby("day_key")["mid"].agg(["count", "min", "max", "mean", "std"]).round(3).to_string())
    print("\n=== Spread counts ===")
    print(tape["spread"].value_counts().sort_index().to_string())

    for feature in ("mid", "dev_0.002", "dev_0.005", "range_pos", "imbalance", "trade_qty_20", "trade_px_dist_20"):
        print_signal_table(tape, feature, horizon=50)

    trader_paths = [
        ROOT / "trading_algos" / "ROUND3" / "hydrogel_buy_low_sell_high.py",
        ROOT / "trading_algos" / "ROUND3" / "hydrogel_only.py",
        ROOT / "trading_algos" / "ROUND3" / "hydrogel_rsi.py",
    ]
    print("\n=== Replay baseline comparison ===")
    baselines = baseline_results(args.data, args.product, trader_paths)
    print(baselines.round(3).to_string(index=False))

    if not args.skip_search:
        print("\n=== Leave-one-day-out SlowReversion search ===")
        wf = walk_forward_search(args.data, args.product)
        print(wf.round(3).to_string(index=False))
        print("\n=== Candidate aggregate from selected holdout params ===")
        print(wf[["train_pnl", "test_pnl", "test_max_drawdown"]].sum().round(3).to_string())


if __name__ == "__main__":
    main()
