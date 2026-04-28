"""
For Round 4 days 1-3 in Data/ROUND4:

1) Per (day, product, Mark): EOD M2M PnL on that product only (from flat;
   counterparty_eod with symbol filter). List rows with EOD PnL > 0 as
   "positive PnL generators" for that slice.

2) Inter-Mark trades: rows where both buyer and seller are in the Mark set
   for that file. Each CSV row is listed exactly once (no double-counting).

3) Optional pair summary: unordered {MarkA, MarkB} trade count and lot sum
   per (day, product) — each underlying print still appears once in the
   detailed list.

Run:  python misc/r4_positive_pnl_intermark_tape.py

Writes:
  misc/r4_positive_pnl_by_mark_product_day.csv
  misc/r4_intermark_trades_dedup.csv
  misc/r4_intermark_pair_counts.csv
  misc/r4_positive_pnl_intermark_report.txt
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from counterparty_eod_pnl import counterparty_eod  # noqa: E402

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "ROUND4")
DAYS: tuple[tuple[int, str], ...] = (
    (1, "trades_round_4_day_1.csv"),
    (2, "trades_round_4_day_2.csv"),
    (3, "trades_round_4_day_3.csv"),
)
_DIR = os.path.dirname(__file__)
OUT_PNL = os.path.join(_DIR, "r4_positive_pnl_by_mark_product_day.csv")
OUT_TRD = os.path.join(_DIR, "r4_intermark_trades_dedup.csv")
OUT_PAIR = os.path.join(_DIR, "r4_intermark_pair_counts.csv")
OUT_TXT = os.path.join(_DIR, "r4_positive_pnl_intermark_report.txt")


def mark_set(df: pd.DataFrame) -> set[str]:
    s = {str(x) for x in df["buyer"].dropna().unique()} | {
        str(x) for x in df["seller"].dropna().unique()
    }
    s.discard("")
    return s


def main() -> int:
    pnl_rows: list[dict] = []
    trade_rows: list[dict] = []
    pair_rows: list[dict] = []

    # ---- (1) per-mark per-product EOD, flag positive --------------------
    for day, fn in DAYS:
        path = os.path.join(_BASE, fn)
        df = pd.read_csv(path, sep=";")
        marks = sorted(mark_set(df))
        mset = set(marks)
        products = sorted(df["symbol"].dropna().astype(str).unique().tolist())

        for product in products:
            flt = frozenset({product})
            for m in marks:
                cash, m2m, _last, pos, _mb = counterparty_eod(df, m, flt)
                pnl = float(cash + m2m)
                pnl_rows.append(
                    {
                        "day": day,
                        "product": product,
                        "mark": m,
                        "eod_pnl": round(pnl, 2),
                        "eod_pos": int(pos.get(product, 0)),
                        "positive_pnl": pnl > 0.0001,
                    }
                )

        # ---- (2) & (3) inter-mark: each file row at most once -----------
        d = df.reset_index().rename(columns={"index": "row_idx"})
        for r in d.itertuples(index=False):
            b, s, sym = r.buyer, r.seller, r.symbol
            if pd.isna(b) or pd.isna(s) or str(b) not in mset or str(s) not in mset:
                continue
            b, s, sym = str(b), str(s), str(sym)
            if b == s:
                continue
            ts = int(r.timestamp) if not pd.isna(r.timestamp) else -1
            p = float(r.price)
            q = int(r.quantity)
            rid = int(getattr(r, "row_idx", -1))
            # One row per file line (timestamp can repeat across distinct prints)
            key = f"{day}|{rid}"
            trade_rows.append(
                {
                    "day": day,
                    "dedup_key": key,
                    "row_in_file": rid,
                    "timestamp": ts,
                    "product": sym,
                    "buyer": b,
                    "seller": s,
                    "price": p,
                    "quantity": q,
                }
            )
            a, c = (b, s) if b < s else (s, b)
            pair_key = (day, sym, a, c)
            # aggregate pairs after loop: use dict
            pr = next(
                (
                    x
                    for x in pair_rows
                    if x["day"] == day
                    and x["product"] == sym
                    and x["mark_a"] == a
                    and x["mark_b"] == c
                ),
                None,
            )
            if pr is None:
                pair_rows.append(
                    {
                        "day": day,
                        "product": sym,
                        "mark_a": a,
                        "mark_b": c,
                        "n_trades": 1,
                        "lot_sum": q,
                    }
                )
            else:
                pr["n_trades"] += 1
                pr["lot_sum"] += q

    df_p = pd.DataFrame(pnl_rows)
    df_t = pd.DataFrame(trade_rows)
    if not df_t.empty and df_t["dedup_key"].duplicated().any():
        raise SystemExit("duplicate trade keys within build - check input")

    # ---- write CSVs -----------------
    pos_only = df_p[df_p["positive_pnl"]].copy()
    pos_only = pos_only.sort_values(
        ["day", "product", "eod_pnl"], ascending=[True, True, False]
    )
    pos_only.drop(columns=["positive_pnl"], errors="ignore").to_csv(OUT_PNL, index=False)
    df_t.drop(columns=["dedup_key"], errors="ignore").to_csv(
        OUT_TRD, index=False
    )
    pd.DataFrame(pair_rows).to_csv(OUT_PAIR, index=False)

    # ---- text report for humans -----
    lines: list[str] = []
    lines.append("R4 - positive EOD (single-product) generators & inter-Mark tape\n")
    lines.append(
        "EOD: cash + M2M at last print for that product (same rules as other scripts).\n"
    )
    lines.append(
        "Each inter-Mark line is one exchange on the file (not duplicated A/B and B/A).\n\n"
    )

    lines.append("=" * 78)
    lines.append("PART A - Mark / product / day  with  EOD PnL  >  0  (product slice)")
    lines.append("=" * 78)
    for _, row in pos_only.iterrows():
        lines.append(
            f"  Day {int(row['day'])}  {str(row['product']):<22}  {row['mark']:<10}  "
            f"PnL={row['eod_pnl']:+12,.2f}  pos={int(row['eod_pos']):+5d}"
        )
    if pos_only.empty:
        lines.append("  (none)")

    lines.append("\n" + "=" * 78)
    lines.append("PART B - inter-Mark prints (one row per trade, all products)")
    lines.append("=" * 78)
    n_im = 0 if df_t.empty else int(len(df_t))
    lines.append(
        f"  ({n_im} inter-Mark file rows; every row = one print. Full list: see *_intermark_trades_dedup.csv and this .txt file.)"
    )
    if not df_t.empty:
        day_prev = None
        for _, row in df_t.sort_values(
            ["day", "product", "timestamp", "buyer"]
        ).iterrows():
            d = int(row["day"])
            if d != day_prev:
                lines.append(f"\n  --- Day {d} ---")
                day_prev = d
            lines.append(
                f"  ts={int(row['timestamp']):>7}  {str(row['product']):<24}  "
                f"buyer={row['buyer']:<9}  seller={row['seller']:<9}  "
                f"px={row['price']!s:>10}  x {int(row['quantity']):>3}"
            )
    else:
        lines.append("  (none)")

    lines.append("\n" + "=" * 78)
    lines.append("PART C - Unordered Mark pairs, trade counts & lot sum (per day, product)")
    lines.append("=" * 78)
    pp = pd.DataFrame(pair_rows)
    if not pp.empty:
        pp = pp.sort_values(["day", "product", "n_trades"], ascending=[True, True, False])
        for _, row in pp.iterrows():
            lines.append(
                f"  Day {int(row['day'])}  {str(row['product']):<20}  "
                f"{row['mark_a']:<8}  <->  {row['mark_b']:<8}  "
                f"n={int(row['n_trades']):>4}  lots={int(row['lot_sum']):>6}"
            )
    else:
        lines.append("  (none)")

    lines.append(f"\n\nFiles: {os.path.normpath(OUT_PNL)}")
    lines.append(f"       {os.path.normpath(OUT_TRD)}")
    lines.append(f"       {os.path.normpath(OUT_PAIR)}")
    with open(OUT_TXT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    # stdout: skip individual ts= lines in PART B (list is in CSV and full .txt)
    brief_lines: list[str] = []
    in_b = False
    for x in lines:
        if x.startswith("PART B"):
            in_b = True
        if x.startswith("PART C"):
            in_b = False
        if in_b and x.startswith("  ts="):
            continue
        brief_lines.append(x)
    print("\n".join(brief_lines))
    ntot = len(lines)
    n_ts = sum(1 for x in lines if x.startswith("  ts="))
    if n_ts:
        print(
            f"\n({n_ts} inter-Mark print lines; full list in {os.path.normpath(OUT_TRD)} and {os.path.normpath(OUT_TXT)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
