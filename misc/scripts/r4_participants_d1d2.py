"""
Round 4, Days 1-3: all market participants in the trade tape.
Per participant per day: trade counts, lot flow, EOD M2M PnL, product focus, inventory.

Run:  python misc/r4_participants_d1d2.py
Writes: misc/r4_participants_d1d2d3.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd

# Import EOD from sibling module
sys.path.insert(0, os.path.dirname(__file__))
from counterparty_eod_pnl import counterparty_eod  # noqa: E402

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "ROUND4")
# (day, filename)
DAY_FILES: tuple[tuple[int, str], ...] = (
    (1, "trades_round_4_day_1.csv"),
    (2, "trades_round_4_day_2.csv"),
    (3, "trades_round_4_day_3.csv"),
)
OUT_CSV = os.path.join(os.path.dirname(__file__), "r4_participants_d1d2d3.csv")

KEYS = (
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_5000",
    "VEV_5200",
)


def all_participants(dfs: list[pd.DataFrame]) -> list[str]:
    s: set[str] = set()
    for d in dfs:
        s |= {str(x) for x in d["buyer"].dropna().unique()}
        s |= {str(x) for x in d["seller"].dropna().unique()}
    s.discard("")
    return sorted(s)


def one_day(
    name: str,
    day: int,
    df: pd.DataFrame,
) -> dict | None:
    sub = df[(df["buyer"] == name) | (df["seller"] == name)]
    n_trades = int(len(sub))
    if n_trades == 0:
        return None

    n_as_buyer = int((sub["buyer"] == name).sum())
    n_as_seller = int((sub["seller"] == name).sum())
    lot_buy = int(sub.loc[sub["buyer"] == name, "quantity"].sum())
    lot_sell = int(sub.loc[sub["seller"] == name, "quantity"].sum())
    den = lot_buy + lot_sell
    lot_buy_share = (lot_buy / den) if den else 0.5

    sym_c = sub["symbol"].value_counts()
    top = sym_c.index[0] if len(sym_c) else ""
    top2 = sym_c.index[1] if len(sym_c) > 1 else ""
    top_share = float(sym_c.iloc[0] / n_trades) if n_trades else 0.0
    top2_share = float(sym_c.iloc[1] / n_trades) if len(sym_c) > 1 else 0.0
    n_symbols = int(sym_c.nunique())

    cash, m2m, _last, pos, _m2m_by = counterparty_eod(df, name)
    eod_pnl = float(cash + m2m)

    row = {
        "name": name,
        "day": day,
        "n_trades": n_trades,
        "n_as_buyer": n_as_buyer,
        "n_as_seller": n_as_seller,
        "lot_buy": lot_buy,
        "lot_sell": lot_sell,
        "lot_buy_share": round(lot_buy_share, 4),
        "eod_pnl": round(eod_pnl, 2),
        "cash": round(cash, 2),
        "eod_m2m": round(m2m, 2),
        "n_symbols_touched": n_symbols,
        "top1_symbol": top,
        "top1_trade_share": round(top_share, 4),
        "top2_symbol": top2,
        "top2_trade_share": round(top2_share, 4),
    }
    for k in KEYS:
        row[f"pos_eod_{k}"] = int(pos.get(k, 0))
    return row


def main() -> int:
    frames: list[pd.DataFrame] = []
    day_frames: list[tuple[int, pd.DataFrame]] = []
    for dnum, fn in DAY_FILES:
        p = os.path.join(_BASE, fn)
        if not os.path.isfile(p):
            print(f"Skip missing: {p}", file=sys.stderr)
            continue
        dframe = pd.read_csv(p, sep=";")
        frames.append(dframe)
        day_frames.append((dnum, dframe))

    if not frames:
        print("No trade files found", file=sys.stderr)
        return 1

    part = all_participants(frames)

    rows: list[dict] = []
    for dnum, dframe in day_frames:
        for n in part:
            r = one_day(n, dnum, dframe)
            if r:
                rows.append(r)

    out = pd.DataFrame(rows)
    out = out.sort_values(["day", "eod_pnl"], ascending=[True, False])
    out.to_csv(OUT_CSV, index=False)

    days = sorted(out["day"].unique().tolist())
    d_str = "+".join(f"D{x}" for x in days)

    # ---- Console report -------------------------------------------------
    print(f"Round 4 - participants, days {min(days)}-{max(days)} (trade tape only)\n")
    print(f"Unique names across {d_str}: {len(part)}")
    print(f"Wrote: {os.path.normpath(OUT_CSV)}\n")

    for d in days:
        s = out[out["day"] == d]
        if s.empty:
            continue
        print("=" * 78)
        print(
            f"  DAY {d}  -  {len(s)} active participants  |  "
            f"sorted by EOD M2M PnL (desc)"
        )
        print("=" * 78)
        print(
            f"  {'name':<14} {'tr':>4} {'buy%':>6} {'EOD PnL':>12} "
            f"{'top focus':<18} {'HY pos':>7} {'VT pos':>7}"
        )
        for _, r in s.iterrows():
            print(
                f"  {r['name']:<14} {r['n_trades']:>4} {r['lot_buy_share']*100:>5.1f}% "
                f"{r['eod_pnl']:>12,.0f}  "
                f"{str(r['top1_symbol']):<18} {r.get('pos_eod_HYDROGEL_PACK',0):>+7.0f} "
                f"{r.get('pos_eod_VELVETFRUIT_EXTRACT',0):>+7.0f}"
            )
        print()

    print("=" * 78)
    print(
        f"  {len(days)}-DAY PnL (sum of EOD if each day is an "
        f"independent session from flat)"
    )
    print("=" * 78)
    agg = out.groupby("name", as_index=False).agg(
        sum_pnl=("eod_pnl", "sum"),
        trades=("n_trades", "sum"),
    )
    agg = agg.sort_values("sum_pnl", ascending=False)
    tlab = f"trades {d_str}" if days else "trades"
    print(f"  {'name':<14} {'sum PnL':>14}  {tlab:>18}")
    for _, r in agg.head(20).iterrows():
        print(
            f"  {r['name']:<14} {r['sum_pnl']:>14,.0f}  {r['trades']!s:>18}"
        )
    if len(agg) > 20:
        print(f"  ... ({len(agg) - 20} more; see CSV)\n")
    else:
        print()

    print("=" * 78)
    print("  BIAS SNAPSHOTS (heuristic)")
    print("=" * 78)
    for d in days:
        b = out[out["day"] == d].set_index("name")
        if len(b) == 0:
            continue
        hi = b["lot_buy_share"].nlargest(3)
        lo = b["lot_buy_share"].nsmallest(3)
        print(f"  Day {d} - most buy-heavy (by lot):")
        for n, v in hi.items():
            print(f"    {n}: {100*v:.1f}% of lots on buy side")
        print(f"  Day {d} - most sell-heavy (by lot):")
        for n, v in lo.items():
            print(
                f"    {n}: {100*v:.1f}% of lots on buy side (sell-heavy)"
            )
        print()

    print(
        f"  MM-style proxy: see top1_trade_share in CSV  "
        f"(>0.4 often one-product specialist on that day)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
