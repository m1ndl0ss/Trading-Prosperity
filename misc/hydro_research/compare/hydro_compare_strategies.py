#!/usr/bin/env python3
"""
HYDRO research matrix: replay all strategy variants x R3d0, R4d1, R4d2.
Uses market_trades injection on tape (R4 has Mark IDs; R3 has empty counterparty fields).

Run after oscillation/hydro_oscillation_stats.py (for Donchian JSON) or pass --donchian-json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import pandas as pd

# .../misc/hydro_research/compare/thisfile -> misc
_MISC = Path(__file__).resolve().parent.parent.parent
_HYDRO_RESEARCH = Path(__file__).resolve().parent.parent
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))
if str(_HYDRO_RESEARCH / "strategies") not in sys.path:
    sys.path.insert(0, str(_HYDRO_RESEARCH / "strategies"))

from hydro_strategy_ablations import (  # noqa: E402
    RESEARCH_TRADERS,
    HydroDonchianR3CalibratedTrader,
    load_full_file_trader,
)
from replay_engine import (  # noqa: E402
    read_prices_csv,
    read_trades_csv,
    replay_path_metrics,
    replay_to_dataframe,
)

HYDRO = "HYDROGEL_PACK"
_OUT = _HYDRO_RESEARCH / "output" / "compare"
_DONCH_DEFAULT = _HYDRO_RESEARCH / "output" / "oscillation" / "hydro_donchian_r3d0.json"

DAY_RUNS: Tuple[Tuple[str, Path, Path, bool], ...] = (
    ("R3_day0", Path("ROUND3") / "prices_round_3_day_0.csv", Path("ROUND3") / "trades_round_3_day_0.csv", True),
    ("R4_day1", Path("ROUND4") / "prices_round_4_day_1.csv", Path("ROUND4") / "trades_round_4_day_1.csv", True),
    ("R4_day2", Path("ROUND4") / "prices_round_4_day_2.csv", Path("ROUND4") / "trades_round_4_day_2.csv", True),
)


def _as_type(tr: Union[Type, Any]) -> Type:
    if isinstance(tr, type):
        return tr
    inst = tr

    class _Wrap:
        def run(self, state):
            return inst.run(state)

    return _Wrap


def run_one(
    trader_cls: Type,
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    inject_mkt: bool,
) -> Dict[str, Any]:
    df = replay_to_dataframe(
        prices,
        trades,
        HYDRO,
        trader_cls,
        initial_cash=0.0,
        inject_market_trades=inject_mkt,
    )
    return replay_path_metrics(df, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=_MISC.parent / "Data",
        help="Data root containing ROUND3, ROUND4",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=_OUT / "hydro_strategy_matrix.csv",
    )
    ap.add_argument(
        "--out-notes",
        type=Path,
        default=_OUT / "hydro_research_notes.txt",
    )
    ap.add_argument(
        "--donchian-json",
        type=Path,
        default=_DONCH_DEFAULT,
        help="From hydro_oscillation_stats.py (R3 p10/p90 mids)",
    )
    ap.add_argument("--skip-full-file", action="store_true", help="Omit prod_file (slow import)")
    args = ap.parse_args()

    data = args.data
    if not data.is_dir():
        raise SystemExit(f"Data directory not found: {data}")

    donch: Dict[str, Any] = {}
    if args.donchian_json.is_file():
        donch = json.loads(args.donchian_json.read_text(encoding="utf-8"))
    d_buy = int(donch.get("buy_thresh", 9935)) if donch else 9935
    d_sell = int(donch.get("sell_thresh", 10030)) if donch else 10030

    specs: List[Tuple[str, str, Union[Type, Any]]] = []
    for k, label, fac in RESEARCH_TRADERS:
        if k == "prod_file" and args.skip_full_file:
            continue
        if k == "prod_file":
            specs.append((k, label, load_full_file_trader()))
        else:
            specs.append((k, label, fac()))
    specs.append(
        (
            "donchian_r3",
            f"Donchian_R3_p10p90_{d_buy}_{d_sell}",
            HydroDonchianR3CalibratedTrader(d_buy, d_sell),
        )
    )

    rows: List[Dict[str, Any]] = []
    for day_id, prel, trel, inject_mkt in DAY_RUNS:
        pp = data / prel
        tt = data / trel
        if not pp.is_file() or not tt.is_file():
            raise SystemExit(f"Missing {pp} or {tt}")
        prices = read_prices_csv(pp)
        trades = read_trades_csv(tt)

        for sid, label, tr in specs:
            cls = _as_type(tr)
            m = run_one(cls, prices, trades, inject_mkt=inject_mkt)
            rows.append(
                {
                    "day_id": day_id,
                    "strategy_id": sid,
                    "strategy_label": label,
                    "inject_market_trades": inject_mkt,
                    "pnl": m["pnl"],
                    "ending_pos": m["ending_pos"],
                    "max_drawdown": m["max_drawdown"],
                    "max_abs_pos": m["max_abs_pos"],
                    "turnover_fills": m["turnover_fills"],
                    "buy_fills": m["buy_fills"],
                    "sell_fills": m["sell_fills"],
                }
            )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out)} rows)")

    has_donch = bool(donch)
    notes = _build_notes(out, d_buy, d_sell, has_donch)
    args.out_notes.write_text(notes, encoding="utf-8")
    print(f"Wrote {args.out_notes}")


def _build_notes(df: pd.DataFrame, d_buy: int, d_sell: int, has_donch: bool) -> str:
    lines = [
        "HYDROGEL_PACK research (local tape replay)",
        "=============================================",
        "",
        "Data: R3 day_0, R4 day_1, day_2 only. R4 day_3 is NOT used.",
        "Replay: single-product, passive fill model; not official IMC PnL.",
        "market_trades: injected from each day's trades CSV so Mark 14/38 sides fire on R4.",
        "R3 tape has empty buyer/seller -> no Mark tilt (tilt branch never triggers).",
        "",
        f"Donchian ablation: buy_thresh={d_buy} sell_thresh={d_sell} (from R3 day0 mid p10/p90){' — JSON' if has_donch else ' — defaults'}.",
        "",
        "Per-strategy median PnL across 3 days:",
    ]
    g = df.groupby("strategy_id")["pnl"].median().sort_values(ascending=False)
    for sid, pnl in g.items():
        lines.append(f"  {sid}: {pnl:,.3f}")
    lines.append("")
    lines.append("R4 days only (median):")
    d4 = df[df["day_id"].str.startswith("R4")]
    if not d4.empty:
        g4 = d4.groupby("strategy_id")["pnl"].median().sort_values(ascending=False)
        for sid, pnl in g4.items():
            lines.append(f"  {sid}: {pnl:,.3f}")
    lines.extend(
        [
            "",
            "R4_day1 vs R4_day2: check hydro_strategy_matrix.csv for rank stability; n=2 is not significant.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
