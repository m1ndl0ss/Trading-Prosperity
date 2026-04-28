#!/usr/bin/env python3
"""
HYDROGEL_PACK only — R4 day_1, day_2, day_3 tape replay:
r4_m14_67_tilt_minseoks vs 497103.

Inject market_trades so Mark 14/38 tilts apply for the minseoks trader.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd

# misc/hydro_research/compare/this -> repo root = 4 levels up
_MISC = Path(__file__).resolve().parent.parent.parent
_ROOT = _MISC.parent
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))

from replay_engine import (  # noqa: E402
    read_prices_csv,
    read_trades_csv,
    replay_path_metrics,
    replay_to_dataframe,
)

HYDRO = "HYDROGEL_PACK"

DAY_RUNS: Tuple[Tuple[str, Path, Path], ...] = (
    ("R4_day1", Path("ROUND4") / "prices_round_4_day_1.csv", Path("ROUND4") / "trades_round_4_day_1.csv"),
    ("R4_day2", Path("ROUND4") / "prices_round_4_day_2.csv", Path("ROUND4") / "trades_round_4_day_2.csv"),
    ("R4_day3", Path("ROUND4") / "prices_round_4_day_3.csv", Path("ROUND4") / "trades_round_4_day_3.csv"),
)


def _load_trader(name: str, path: Path) -> Type:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.Trader


def sharpe_from_equity_diff(equity: pd.Series) -> float:
    """
    Intraday-style Sharpe: sqrt(N) * mean(Δequity) / std(Δequity).
    N = number of per-tick equity changes (common scaling for i.i.d. increments;
    tick changes are not independent — interpret as a relative risk-adjusted
    path metric, not a literal annual return Sharpe).
    """
    d = equity.astype(float).diff().dropna()
    if len(d) < 2:
        return float("nan")
    s = float(d.std())
    if s < 1e-15:
        return float("nan")
    return float(np.sqrt(len(d)) * d.mean() / s)


def annualized_sharpe_from_daily_pnls(pnls: List[float]) -> float:
    """Classic daily PnL Sharpe: mean/ std * sqrt(252). Unreliable for n<5."""
    a = np.asarray(pnls, dtype=float)
    if len(a) < 2:
        return float("nan")
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1))
    if s < 1e-15:
        return float("nan")
    return m / s * np.sqrt(252.0)


def run(
    data: Path,
    out_csv: Path,
) -> pd.DataFrame:
    path_min = _ROOT / "trading_algos" / "ROUND4" / "r4_m14_67_tilt_minseoks_option_strat.py"
    path_497 = _ROOT / "LOGS" / "ROUND4" / "Potential improvement of friend's algo" / "497103.py"
    for p in (path_min, path_497):
        if not p.is_file():
            raise SystemExit(f"Missing file: {p}")

    TraderMin = _load_trader("hydro_m14_67", path_min)
    TraderFr = _load_trader("hydro_497103", path_497)

    rows: List[Dict[str, Any]] = []
    by_algo: Dict[str, List[float]] = {"m14_67_tilt": [], "v12_497103": []}

    for day_id, prel, trel in DAY_RUNS:
        pp = data / prel
        tt = data / trel
        if not pp.is_file() or not tt.is_file():
            raise SystemExit(f"Missing {pp} or {tt}")
        prices = read_prices_csv(pp)
        trades = read_trades_csv(tt)
        n_ticks = int(prices[prices["product"] == HYDRO]["timestamp"].nunique())

        for label, tcls, key in (
            ("r4_m14_67_tilt_minseoks", TraderMin, "m14_67_tilt"),
            ("v12_497103_PotentialFriend", TraderFr, "v12_497103"),
        ):
            df = replay_to_dataframe(
                prices,
                trades,
                HYDRO,
                tcls,
                initial_cash=0.0,
                inject_market_trades=True,
            )
            m = replay_path_metrics(df, 0.0)
            sh = sharpe_from_equity_diff(df["equity"])
            rows.append(
                {
                    "day_id": day_id,
                    "algo": label,
                    "algo_key": key,
                    "n_ticks": n_ticks,
                    "pnl": m["pnl"],
                    "ending_pos": m["ending_pos"],
                    "max_drawdown": m["max_drawdown"],
                    "max_abs_pos": m["max_abs_pos"],
                    "turnover_fills": m["turnover_fills"],
                    "buy_fills": m["buy_fills"],
                    "sell_fills": m["sell_fills"],
                    "sharpe_intraday_eq_diff": sh,
                }
            )
            by_algo[key].append(float(m["pnl"]))

    n_days = len(DAY_RUNS)
    for key, name in (("m14_67_tilt", "r4_m14_67_tilt_minseoks"), ("v12_497103", "v12_497103_PotentialFriend")):
        pnl_s = by_algo[key]
        rows.append(
            {
                "day_id": f"COMBINED_{n_days}d_annualized_from_DAILY_pnl",
                "algo": name,
                "algo_key": key,
                "n_ticks": 0,
                "pnl": float(np.sum(pnl_s)),
                "ending_pos": 0,
                "max_drawdown": 0,
                "max_abs_pos": 0,
                "turnover_fills": 0,
                "buy_fills": 0,
                "sell_fills": 0,
                "sharpe_intraday_eq_diff": annualized_sharpe_from_daily_pnls(pnl_s),
            }
        )
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=_ROOT / "Data", help="Data root (ROUND4 CSVs)")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=_MISC / "hydro_research" / "output" / "compare" / "hydro_m14_vs_497103_r4d1d2d3_hydro_only.csv",
    )
    args = ap.parse_args()
    df = run(args.data, args.out_csv)
    print(df.to_string(index=False))
    print()
    print(f"Wrote {args.out_csv}")
    print(
        "Notes: sharpe_intraday_eq_diff = sqrt(N)*mean(diff equity)/std(diff equity) per day. "
        "COMBINED row's Sharpe = mean(daily pnl)/std(daily pnl)*sqrt(252) over all run days (still noisy with few days). "
        "Replay is single-product, passive fill; not official IMC PnL."
    )


if __name__ == "__main__":
    main()
