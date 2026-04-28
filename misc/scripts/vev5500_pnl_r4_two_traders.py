#!/usr/bin/env python3
"""
Full-book tape replay: per-symbol PnL for VEV_5500 (and summary) for two traders, R4 days 1-3.
Uses ``replay_portfolio_symbol_pnls`` so VELVET / shared traderData (EMA) match a single running algo.

Usage (repo root):

  python misc/scripts/vev5500_pnl_r4_two_traders.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MISC = Path(__file__).resolve().parent.parent
_REPO = _MISC.parent
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))

from replay_engine import (  # noqa: E402
    read_prices_csv,
    read_trades_csv,
    replay_portfolio_symbol_pnls,
)

VEV = "VEV_5500"

DAY_PATHS: tuple[tuple[str, Path, Path], ...] = (
    ("1", Path("ROUND4") / "prices_round_4_day_1.csv", Path("ROUND4") / "trades_round_4_day_1.csv"),
    ("2", Path("ROUND4") / "prices_round_4_day_2.csv", Path("ROUND4") / "trades_round_4_day_2.csv"),
    ("3", Path("ROUND4") / "prices_round_4_day_3.csv", Path("ROUND4") / "trades_round_4_day_3.csv"),
)


def load_trader(path: Path) -> type:
    spec = importlib.util.spec_from_file_location("tr", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"load failed: {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Trader


def main() -> None:
    data = _REPO / "Data"
    p497 = _REPO / "LOGS" / "ROUND4" / "Potential improvement of friend's algo" / "497103.py"
    pmin = _REPO / "trading_algos" / "ROUND4" / "r4_m14_67_tilt_minseoks_option_strat.py"
    for f in (p497, pmin):
        if not f.is_file():
            raise SystemExit(f"Missing {f}")
    t497 = load_trader(p497)
    tmin = load_trader(pmin)

    print("Full-book replay, inject_market_trades=True, initial_cash=0")
    print(f"Data: {data}")
    print(
        "Note: passive fill rules (Orchids-style) often do not match option aggressive "
        "quotes vs tape; pnl may be 0 even when ticks_with_orders is large. "
        "Compare to IMC logs for realized VEV_5500 PnL."
    )
    print()

    for name, tcls, label in (("497103 (friend v12)", t497, "497103"), ("r4_m14_67_tilt_minseoks", tmin, "minseoks")):
        print("=" * 70)
        print(label)
        print("=" * 70)
        tot_vev = 0.0
        for dkey, pcsv, tcsv in DAY_PATHS:
            pp, tt = data / pcsv, data / tcsv
            if not pp.is_file() or not tt.is_file():
                print(f"  day {dkey}: missing csv")
                continue
            prices = read_prices_csv(pp)
            trades = read_trades_csv(tt)
            tape_n = int((trades["symbol"] == VEV).sum())
            pnl_s, end_pos, _total_eq, _tb, _ts, fmap, onticks = replay_portfolio_symbol_pnls(
                prices, trades, tcls, initial_cash=0.0, inject_market_trades=True
            )
            vev = float(pnl_s.get(VEV, 0.0))
            pos5500 = int(end_pos.get(VEV, 0))
            b5500, s5500 = fmap.get(VEV, (0, 0))
            ot = int(onticks.get(VEV, 0))
            tot_vev += vev
            traded = (b5500 + s5500) > 0
            print(
                f"  R4 day_{dkey}  {VEV}  tape_rows={tape_n}  pnl={vev:,.2f}  "
                f"ending_pos={pos5500:>4}  fills buy={b5500} sell={s5500}  "
                f"ticks_with_orders={ot}  passive_fills={traded}"
            )
        print(f"  --- {VEV} 3-day sum pnl: {tot_vev:,.2f}")
        print()


if __name__ == "__main__":
    main()
