"""
Summarize Mark 38 counterparty activity for all ROUND4 trade CSVs
(Data/trades_round_4_day_*.csv).  Run: python misc/mark38_r4_all_days.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import pandas as pd

DATA_GLOB = os.path.join(
    os.path.dirname(__file__),
    "..",
    "Data",
    "ROUND4",
    "trades_round_4_day_*.csv",
)
CRASH_LO, CRASH_HI = 380_000, 520_000


def day_from_path(path: str) -> int:
    m = re.search(r"day_(\d+)\.csv$", path, re.I)
    return int(m.group(1)) if m else -1


def m38_frame(df: pd.DataFrame) -> pd.DataFrame:
    b = df["buyer"] == "Mark 38"
    s = df["seller"] == "Mark 38"
    out = df[b | s].copy()
    out["side"] = "buy"
    out.loc[s & ~b, "side"] = "sell"
    out.loc[b & s, "side"] = "both"  # should not happen
    return out


def print_day(name: str, m38: pd.DataFrame, full: pd.DataFrame) -> None:
    n = len(m38)
    print(f"\n{'='*72}")
    print(f"  {name}  -  all trades: {len(full):,}  |  Mark 38 touches: {n:,}")
    print("=" * 72)
    if n == 0:
        return

    as_buyer = (m38["buyer"] == "Mark 38").sum()
    as_sell = (m38["seller"] == "Mark 38").sum()
    q_buy = m38.loc[m38["buyer"] == "Mark 38", "quantity"].sum()
    q_sell = m38.loc[m38["seller"] == "Mark 38", "quantity"].sum()
    ts0, ts1 = m38["timestamp"].min(), m38["timestamp"].max()
    print(f"  as buyer: {as_buyer:4d}  ({q_buy:6.0f} lots)  |  as seller: {as_sell:4d}  ({q_sell:6.0f} lots)")
    print(f"  timestamp range: {ts0:,}  ->  {ts1:,}")

    print(f"\n  By symbol (count / lots):")
    g = m38.groupby("symbol", sort=False)
    rows = []
    for sym, sub in g:
        rows.append(
            (len(sub), (sub["quantity"] * 1.0).sum(), sym)
        )
    for cnt, vol, sym in sorted(rows, key=lambda x: -x[0]):
        sub = m38[m38["symbol"] == sym]
        nb = (sub["buyer"] == "Mark 38").sum()
        ns = (sub["seller"] == "Mark 38").sum()
        print(f"    {sym:24s}  n={cnt:4d}  vol={vol:6.0f}   (buy {nb:3d} / sell {ns:3d})")

    # HYDROGEL: sample (fingerprint) + 50k timestamp buckets
    h = m38[m38["symbol"] == "HYDROGEL_PACK"].sort_values("timestamp")
    if h.empty:
        print("\n  HYDROGEL_PACK: (no Mark 38 trades this day)")
    else:
        print(f"\n  HYDROGEL_PACK: {len(h)} trades - first/last 5 when long:")
        h5 = h.head(5)
        for r in h5.itertuples(index=False):
            side = "BUY " if r.buyer == "Mark 38" else "SELL"
            print(
                f"    ts={r.timestamp:>7,}  {side}  {r.price!s:>10} x {r.quantity!s:>3}"
            )
        if len(h) > 10:
            print("    ...")
            for r in h.tail(5).itertuples(index=False):
                side = "BUY " if r.buyer == "Mark 38" else "SELL"
                print(
                    f"    ts={r.timestamp:>7,}  {side}  {r.price!s:>10} x {r.quantity!s:>3}"
                )
        elif len(h) > 5:
            for r in h.iloc[5:].itertuples(index=False):
                side = "BUY " if r.buyer == "Mark 38" else "SELL"
                print(
                    f"    ts={r.timestamp:>7,}  {side}  {r.price!s:>10} x {r.quantity!s:>3}"
                )

        h = h.copy()
        h["bucket"] = (h["timestamp"] // 50_000).astype(int) * 50_000

        def net_flow(sub: pd.DataFrame) -> float:
            bq = sub.loc[sub["buyer"] == "Mark 38", "quantity"].sum()
            sq = sub.loc[sub["seller"] == "Mark 38", "quantity"].sum()
            return bq - sq

        agg = h.groupby("bucket", sort=True).apply(net_flow, include_groups=False)
        print("  HYDRO net lot flow (buy - sell) per 50k ts bucket [Mark 38]:")
        for bkt, v in agg.items():
            print(f"    [{bkt:>7,} - {bkt + 49_999:>7,}):  {v:>+6.0f} lots")

    v = m38[m38["symbol"] == "VELVETFRUIT_EXTRACT"]
    if v.empty:
        print("\n  VELVETFRUIT_EXTRACT: (no Mark 38 trades this day)")
    else:
        print(f"\n  VELVETFRUIT_EXTRACT: {len(v)} trades (full list):")
        for r in v.sort_values("timestamp").itertuples(index=False):
            side = "BUY " if r.buyer == "Mark 38" else "SELL"
            print(
                f"    ts={r.timestamp:>7,}  {side}  {r.price!s:>8} x {r.quantity!s:>3}"
            )


def print_crash_window(name: str, m38: pd.DataFrame) -> None:
    w = m38[(m38["timestamp"] >= CRASH_LO) & (m38["timestamp"] <= CRASH_HI)]
    n = len(w)
    print(f"\n  Crash band [{CRASH_LO:,}-{CRASH_HI:,}]:  Mark 38 rows = {n}")
    if n == 0:
        return
    print(f"    HYDRO: {(w['symbol'] == 'HYDROGEL_PACK').sum():3d}  |  VELVET: {(w['symbol'] == 'VELVETFRUIT_EXTRACT').sum():3d}  |  VEV: {(w['symbol'].astype(str).str.startswith('VEV_')).sum():3d}")


def main() -> int:
    paths = sorted(glob.glob(DATA_GLOB), key=day_from_path)
    if not paths:
        print("No files matched:", os.path.normpath(DATA_GLOB), file=sys.stderr)
        return 1

    print("Mark 38 fingerprint - Round 4 (all available days)\n")
    print(f"Data files: {len(paths)}")

    for path in paths:
        day = day_from_path(path)
        df = pd.read_csv(path, sep=";")
        m38 = m38_frame(df)
        print_day(f"Day {day}", m38, df)
        print_crash_window(f"Day {day}", m38)

    # Totals across days
    all_m = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        all_m.append(m38_frame(df))
    big = pd.concat(all_m, ignore_index=True) if all_m else pd.DataFrame()
    print(f"\n{'='*72}")
    print(f"  COMBINED (all {len(paths)} days)")
    print("=" * 72)
    print(f"  Total Mark 38 touches: {len(big):,}")
    if not big.empty:
        vc = big["symbol"].value_counts()
        print("  By symbol (top 15):")
        for sym, c in vc.head(15).items():
            print(f"    {sym:24s}  {c:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
