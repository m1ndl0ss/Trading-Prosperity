"""
1) Mark 67 - EOD M2M PnL on VELVETFRUIT_EXTRACT only (per day)

2) Synthetic "trading against Mark 38 on HYDRO": one merged book of all
   counterparties on the other side of his HYDROGEL_PACK prints (in tape order;
   M2M at that day's last HYDRO print).

   This is the aggregate of everyone who sat across from M38 on hydrogel
   (not a single named trader). Mark 38 on both sides: skipped.

Run: python misc/mark67_velvet_m38contra_hydro.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from counterparty_eod_pnl import counterparty_eod  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "ROUND4")
HYDRO = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
DAYS = (
    (1, "trades_round_4_day_1.csv"),
    (2, "trades_round_4_day_2.csv"),
    (3, "trades_round_4_day_3.csv"),
)


def last_price(df: pd.DataFrame, symbol: str) -> float:
    d = df[df["symbol"] == symbol]
    if d.empty:
        return 0.0
    return float(d["price"].iloc[-1])


def aggregate_contra_m38_hydro_eod(df: pd.DataFrame) -> tuple[float, float, int, float]:
    """
    One wallet = sum of all agents on the non-Mark-38 side of HYDRO trades
    that involve Mark 38.
    """
    d = df.reset_index().sort_values(["timestamp", "index"], kind="mergesort")
    lhyd = last_price(d, HYDRO)
    cash = 0.0
    pos = 0

    for r in d.itertuples(index=False):
        if str(r.symbol) != HYDRO:
            continue
        b, s = r.buyer, r.seller
        p = float(r.price)
        q = int(r.quantity)
        if b == "Mark 38" and s == "Mark 38":
            continue
        if b == "Mark 38" and s != "Mark 38":
            # Contra = seller: receives cash, reduces long / adds short
            cash += p * q
            pos -= q
        elif s == "Mark 38" and b != "Mark 38":
            # Contra = buyer: pays cash, takes inventory
            cash -= p * q
            pos += q

    pnl = cash + pos * lhyd
    return pnl, cash, pos, lhyd


def main() -> int:
    vflt = frozenset({VELVET})

    print("Mark 67 - VELVET only (EOD M2M)\n")
    print(f"  {'day':>3}  {'EOD PnL':>14}  {'cash':>14}  {'M2M':>14}  {'pos VELV':>10}  last VELV")
    for day, fn in DAYS:
        path = os.path.join(BASE, fn)
        df = pd.read_csv(path, sep=";")
        cash, m2m, last, pos, _ = counterparty_eod(df, "Mark 67", vflt)
        pnl = cash + m2m
        qv = int(pos.get(VELVET, 0))
        lv = last.get(VELVET, 0.0)
        print(
            f"  {day:>3}  {pnl:>14,.2f}  {cash:>14,.2f}  {m2m:>14,.2f}  {qv:>+10d}  {lv:>10.2f}"
        )

    print("\n" + "=" * 72)
    print(
        "Contra Mark 38 on HYDRO (merged wallet of the counterparty in each M38 HYDRO row)\n"
    )
    print(
        f"  {'day':>3}  {'EOD PnL':>14}  {'cash':>14}  {'M2M':>14}  {'pos HYDRO':>10}  last HYD"
    )
    for day, fn in DAYS:
        path = os.path.join(BASE, fn)
        df = pd.read_csv(path, sep=";")
        pnl, cash, pos, lh = aggregate_contra_m38_hydro_eod(df)
        m2m = pos * lh
        print(
            f"  {day:>3}  {pnl:>14,.2f}  {cash:>14,.2f}  {m2m:>14,.2f}  {pos:>+10d}  {lh:>10.2f}"
        )

    # Bilateral identity: merged-contra HYDRO EOD = - Mark 38 HYDRO-only EOD (same last px).
    hflt = frozenset({HYDRO})
    print("\n  Sanity (M38 HYDRO-only + merged contra = 0):")
    for day, fn in DAYS:
        df = pd.read_csv(os.path.join(BASE, fn), sep=";")
        c, m, _, _, _ = counterparty_eod(df, "Mark 38", hflt)
        m38 = c + m
        contra, _, _, _ = aggregate_contra_m38_hydro_eod(df)
        print(f"  Day {day}:  {m38 + contra:>8.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
