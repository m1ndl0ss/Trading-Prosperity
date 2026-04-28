"""
Mark counterparty research (Round 4): multi-iteration pipeline from the research plan.

Outputs under misc/mark_research/<category>/ (default --out-dir). Run:

  python misc/mark_behavior_research.py --all
  python misc/mark_behavior_research.py --iter0
  python misc/mark_behavior_research.py --iter1
  ... --iter2 --iter3 --iter4 --iter5

Data: Data/ROUND4/trades_round_4_day_*.csv and prices_round_4_day_*.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------------

# This file lives in misc/scripts/; repo root is two levels up.
_SCRIPTS = Path(__file__).resolve().parent
MISC_DIR = _SCRIPTS.parent
REPO_ROOT = MISC_DIR.parent
DATA_R4 = REPO_ROOT / "Data" / "ROUND4"
# Default report root: misc/mark_research/<phase>/
DEFAULT_MARK_RESEARCH_OUT = MISC_DIR / "mark_research"

DAY_FILES: tuple[tuple[int, str, str], ...] = (
    (1, "trades_round_4_day_1.csv", "prices_round_4_day_1.csv"),
    (2, "trades_round_4_day_2.csv", "prices_round_4_day_2.csv"),
    (3, "trades_round_4_day_3.csv", "prices_round_4_day_3.csv"),
)

MARK_RE = re.compile(r"^Mark \d{1,2}$")

# Subdirs under the output base separate artifacts by research phase.
def output_paths(out_base: Path) -> dict[str, Path]:
    b = out_base
    return {
        "iter0_coverage": b / "iter0_inventory" / "coverage.csv",
        "iter0_partners": b / "iter0_inventory" / "partner_concentration.csv",
        "iter1_flow": b / "iter1_flow" / "flow_by_mark_symbol_day.csv",
        "iter2_regime": b / "iter2_regime" / "flow_by_regime.csv",
        "iter3_event": b / "iter3_event_study" / "event_study.csv",
        "iter4_stability": b / "iter4_stability" / "stability_trusted.csv",
        "iter5_exec": b / "iter5_execution" / "execution_notes.txt",
    }

# Forward-return horizons (match tape timestamp steps; 100 = common in logs)
FORWARD_DELTAS: tuple[int, ...] = (1, 100, 1000, 5000)

# Min trades for iter4 "signal" to count
MIN_TRADES_REGIME: int = 5


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _is_mark(x: Any) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip()
    return bool(MARK_RE.match(s))


def _mark_set(df: pd.DataFrame) -> set[str]:
    s: set[str] = set()
    for c in ("buyer", "seller"):
        s |= {str(x) for x in df[c].dropna().unique() if _is_mark(x)}
    return s


def load_trades_day(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep=";")
    d["day"] = int(re.search(r"day_(\d+)", path.name, re.I).group(1)) if re.search(
        r"day_(\d+)", path.name, re.I
    ) else -1
    return d


def load_prices_day(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep=";")
    if "product" not in d.columns and "symbol" in d.columns:
        d = d.rename(columns={"symbol": "product"})
    return d


def _partner_for_row(buyer: str, seller: str, me: str) -> str:
    if buyer == me and seller != me:
        return str(seller)
    if seller == me and buyer != me:
        return str(buyer)
    return ""


# =============================================================================
# Iter 0: coverage + partner concentration
# =============================================================================


def run_iter0(out_base: Path, data_dir: Path) -> None:
    paths = output_paths(out_base)
    rows_cov: list[dict[str, Any]] = []
    rows_p: list[dict[str, Any]] = []

    for dnum, tfile, _ in DAY_FILES:
        tp = data_dir / tfile
        if not tp.is_file():
            print(f"skip missing {tp}", file=sys.stderr)
            continue
        tr = load_trades_day(tp)
        for name in sorted(_mark_set(tr)):
            sub = tr[(tr["buyer"] == name) | (tr["seller"] == name)]
            for sym, g in sub.groupby("symbol", sort=True):
                n = len(g)
                rows_cov.append(
                    {
                        "day": dnum,
                        "mark": name,
                        "symbol": str(sym),
                        "n_trades": n,
                    }
                )
                # partner concentration: other side
                other = g.apply(
                    lambda r: _partner_for_row(
                        r["buyer"], r["seller"], name
                    ),
                    axis=1,
                )
                other = other[other != ""]
                if len(other) == 0:
                    continue
                vc = other.value_counts()
                top = vc.index[0]
                top_share = float(vc.iloc[0] / len(other))
                rows_p.append(
                    {
                        "day": dnum,
                        "mark": name,
                        "symbol": str(sym),
                        "n_trades": n,
                        "top_counterparty": str(top),
                        "top_counterparty_share": round(top_share, 4),
                        "n_distinct_counterparties": int(vc.nunique()),
                    }
                )

    df_c = pd.DataFrame(rows_cov).sort_values(
        ["mark", "symbol", "day"]
    )
    df_p = pd.DataFrame(rows_p).sort_values(
        ["mark", "symbol", "day"]
    )
    out_base.mkdir(parents=True, exist_ok=True)
    pc, pp = paths["iter0_coverage"], paths["iter0_partners"]
    pc.parent.mkdir(parents=True, exist_ok=True)
    df_c.to_csv(pc, index=False)
    df_p.to_csv(pp, index=False)
    print(f"Wrote {pc} ({len(df_c)} rows)")
    print(f"Wrote {pp} ({len(df_p)} rows)")


# =============================================================================
# Iter 1: flow stats
# =============================================================================


def run_iter1(out_base: Path, data_dir: Path) -> None:
    paths = output_paths(out_base)
    rows: list[dict[str, Any]] = []

    for dnum, tfile, _ in DAY_FILES:
        tp = data_dir / tfile
        if not tp.is_file():
            continue
        tr = load_trades_day(tp)
        for name in sorted(_mark_set(tr)):
            sub = tr[(tr["buyer"] == name) | (tr["seller"] == name)]
            for sym, g in sub.groupby("symbol", sort=True):
                gb = g[g["buyer"] == name]
                gs = g[g["seller"] == name]
                lot_buy = int(gb["quantity"].sum()) if len(gb) else 0
                lot_sell = int(gs["quantity"].sum()) if len(gs) else 0
                n_buy = int(len(gb))
                n_sell = int(len(gs))
                n_tr = int(len(g))
                tot = lot_buy + lot_sell
                lot_buy_share = (lot_buy / tot) if tot else 0.5
                q_all = g["quantity"].astype(float)
                max_q = int(q_all.max()) if len(q_all) else 0
                cv = float(q_all.std() / q_all.mean()) if q_all.mean() and len(q_all) > 1 else 0.0
                net = lot_buy - lot_sell
                rows.append(
                    {
                        "day": dnum,
                        "mark": name,
                        "symbol": str(sym),
                        "n_trades": n_tr,
                        "n_as_buyer": n_buy,
                        "n_as_seller": n_sell,
                        "lot_buy": lot_buy,
                        "lot_sell": lot_sell,
                        "net_qty": net,
                        "lot_buy_share": round(lot_buy_share, 4),
                        "avg_buy_size": round(lot_buy / n_buy, 4) if n_buy else 0.0,
                        "avg_sell_size": round(lot_sell / n_sell, 4) if n_sell else 0.0,
                        "max_single_qty": max_q,
                        "qty_cv": round(cv, 4),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["mark", "symbol", "day"])
    p = paths["iter1_flow"]
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Wrote {p} ({len(df)} rows)")


# =============================================================================
# Iter 2: mid/spread regime stratification
# =============================================================================


def _build_mid_series(prices: pd.DataFrame, product: str) -> pd.DataFrame:
    p = prices[prices["product"].astype(str) == product].copy()
    if p.empty:
        return p
    p = p.sort_values("timestamp")
    p["bb"] = p["bid_price_1"]
    p["ba"] = p["ask_price_1"]
    p["spread"] = p["ba"] - p["bb"]
    p["mid"] = p["mid_price"]
    return p[["timestamp", "mid", "spread", "bb", "ba"]]


def run_iter2(out_base: Path, data_dir: Path) -> None:
    paths = output_paths(out_base)
    rows: list[dict[str, Any]] = []

    for dnum, tfile, pfile in DAY_FILES:
        tp = data_dir / tfile
        pp = data_dir / pfile
        if not tp.is_file() or not pp.is_file():
            continue
        tr = load_trades_day(tp)
        pr = load_prices_day(pp)
        for name in sorted(_mark_set(tr)):
            sub = tr[(tr["buyer"] == name) | (tr["seller"] == name)]
            for sym in sub["symbol"].dropna().unique():
                ps = _build_mid_series(pr, str(sym))
                if ps.empty or "mid" not in ps.columns:
                    continue
                mids = ps["mid"].astype(float)
                tss = ps["timestamp"].to_numpy()
                tercile_edges = (
                    float(mids.quantile(1 / 3)),
                    float(mids.quantile(2 / 3)),
                )

                g = sub[sub["symbol"] == sym]
                for reg in (0, 1, 2):
                    gr = []
                    for _, r in g.iterrows():
                        ts = int(r["timestamp"])
                        j = int(np.searchsorted(tss, ts, side="right") - 1)
                        if j < 0:
                            j = 0
                        mid_now = float(mids.iloc[j] if j < len(mids) else mids.iloc[-1])
                        if mid_now < tercile_edges[0]:
                            b = 0
                        elif mid_now < tercile_edges[1]:
                            b = 1
                        else:
                            b = 2
                        if b == reg:
                            gr.append(r)
                    if not gr:
                        gr_df = g.iloc[0:0]
                    else:
                        gr_df = pd.DataFrame(gr)
                    if gr_df.empty:
                        lot_b = lot_s = ntr = 0
                        lbs = 0.5
                    else:
                        ntr = len(gr_df)
                        gbb = gr_df[gr_df["buyer"] == name]
                        gss = gr_df[gr_df["seller"] == name]
                        lot_b = int(gbb["quantity"].sum())
                        lot_s = int(gss["quantity"].sum())
                        tot = lot_b + lot_s
                        lbs = (lot_b / tot) if tot else 0.5
                    rows.append(
                        {
                            "day": dnum,
                            "mark": name,
                            "symbol": str(sym),
                            "mid_tercile": reg,
                            "tercile_label": ["low", "mid", "high"][reg],
                            "n_trades": ntr,
                            "lot_buy": lot_b,
                            "lot_sell": lot_s,
                            "net_qty": lot_b - lot_s,
                            "lot_buy_share": round(lbs, 4),
                        }
                    )
                # spread regime: wide if spread at trade >= day median
                med_s = float(ps["spread"].median()) if not ps.empty else 0.0
                for wname, want_wide in (("tight", False), ("wide", True)):
                    g2: list[pd.Series] = []
                    for _, r in g.iterrows():
                        ts = int(r["timestamp"])
                        j = int(np.searchsorted(tss, ts, side="right") - 1)
                        if j < 0 or j >= len(ps):
                            continue
                        sp = float(ps["spread"].iloc[min(j, len(ps) - 1)])
                        if (sp >= med_s) == want_wide:
                            g2.append(r)
                    if not g2:
                        gdf2 = g.iloc[0:0]
                    else:
                        gdf2 = pd.DataFrame(g2)
                    ntr2 = int(len(gdf2))
                    if gdf2.empty:
                        lot_b2 = lot_s2 = 0
                        lbs2 = 0.5
                    else:
                        gbb = gdf2[gdf2["buyer"] == name]
                        gss = gdf2[gdf2["seller"] == name]
                        lot_b2 = int(gbb["quantity"].sum())
                        lot_s2 = int(gss["quantity"].sum())
                        t2 = lot_b2 + lot_s2
                        lbs2 = (lot_b2 / t2) if t2 else 0.5
                    rows.append(
                        {
                            "day": dnum,
                            "mark": name,
                            "symbol": str(sym),
                            "mid_tercile": -1,
                            "tercile_label": wname,
                            "n_trades": ntr2,
                            "lot_buy": lot_b2,
                            "lot_sell": lot_s2,
                            "net_qty": lot_b2 - lot_s2,
                            "lot_buy_share": round(lbs2, 4),
                        }
                    )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(
            ["mark", "symbol", "day", "mid_tercile", "tercile_label"]
        )
    p = paths["iter2_regime"]
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Wrote {p} ({len(df)} rows)")


# =============================================================================
# Iter 3: event study
# =============================================================================


def _mid_asof(
    p_sym: pd.DataFrame, ts: int, col: str = "mid"
) -> float | None:
    if p_sym.empty or col not in p_sym:
        return None
    tss = p_sym["timestamp"].to_numpy()
    j = int(np.searchsorted(tss, ts, side="right") - 1)
    if j < 0:
        j = 0
    if j >= len(p_sym):
        j = len(p_sym) - 1
    return float(p_sym[col].iloc[j])


def run_iter3(out_base: Path, data_dir: Path) -> None:
    """Event study: streaming aggregate (day, mark, symbol, side) -> mean/median forward returns."""
    from collections import defaultdict

    paths = output_paths(out_base)

    # (day, mark, symbol, side) -> dlt -> list of returns
    bins: dict[tuple[int, str, str, str], dict[int, list[float]]] = defaultdict(
        lambda: {dlt: [] for dlt in FORWARD_DELTAS}
    )

    for dnum, tfile, pfile in DAY_FILES:
        tp = data_dir / tfile
        pp = data_dir / pfile
        if not tp.is_file() or not pp.is_file():
            continue
        tr = load_trades_day(tp)
        pr = load_prices_day(pp)

        for name in sorted(_mark_set(tr)):
            sub = tr[(tr["buyer"] == name) | (tr["seller"] == name)]
            for sym in sub["symbol"].dropna().unique():
                s = str(sym)
                p_sym = pr[pr["product"].astype(str) == s].sort_values("timestamp")
                if p_sym.empty or "mid_price" not in p_sym:
                    continue
                p_sym = p_sym.rename(columns={"mid_price": "mid"})

                for _, r in sub[sub["symbol"] == s].iterrows():
                    ts = int(r["timestamp"])
                    is_buy = r["buyer"] == name
                    side = "buy" if is_buy else "sell"
                    m0 = _mid_asof(p_sym, ts, "mid")
                    if m0 is None or m0 == 0:
                        continue
                    key = (dnum, name, s, side)
                    for dlt in FORWARD_DELTAS:
                        m1 = _mid_asof(p_sym, ts + dlt, "mid")
                        if m1 is None:
                            continue
                        bins[key][dlt].append((m1 - m0) / m0)

    agg_rows: list[dict[str, Any]] = []
    for (dnum, name, s, side), dct in sorted(bins.items()):
        row: dict[str, Any] = {
            "day": dnum,
            "mark": name,
            "symbol": s,
            "side": side,
        }
        for dlt in FORWARD_DELTAS:
            arr = dct[dlt]
            vals = np.array(arr, dtype=float) if arr else np.array([])
            row[f"mean_ret_{dlt}"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"median_ret_{dlt}"] = float(np.median(vals)) if len(vals) else np.nan
        ref = 100 if 100 in FORWARD_DELTAS else FORWARD_DELTAS[0]
        row["n_trades"] = len(dct[ref])
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)
    if len(agg):
        agg = agg.sort_values(["mark", "symbol", "day", "side"])
    p3 = paths["iter3_event"]
    p3.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(p3, index=False)
    print(f"Wrote {p3} ({len(agg)} aggregate rows)")


# =============================================================================
# Iter 4: cross-day stability
# =============================================================================


def run_iter4(out_base: Path) -> None:
    paths = output_paths(out_base)
    path = paths["iter3_event"]
    if not path.is_file():
        print("iter4: run iter3 first", file=sys.stderr)
        return
    e = pd.read_csv(path)
    if e.empty or "mark" not in e.columns:
        p4 = paths["iter4_stability"]
        p4.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "mark": [],
                "symbol": [],
                "verdict": [],
                "notes": [],
            }
        ).to_csv(p4, index=False)
        return

    c = "mean_ret_100" if "mean_ret_100" in e.columns else None
    if c is None:
        rc = [x for x in e.columns if x.startswith("mean_ret_")]
        c = rc[0] if rc else "mean_ret_1"

    out_rows: list[dict[str, Any]] = []
    for (mark, sym), g in e.groupby(["mark", "symbol"]):
        days = int(g["day"].nunique())
        gbuy = g[g["side"] == "buy"].sort_values("day")
        gsell = g[g["side"] == "sell"].sort_values("day")
        nbt = int(gbuy["n_trades"].sum()) if "n_trades" in gbuy.columns else 0
        nst = int(gsell["n_trades"].sum()) if "n_trades" in gsell.columns else 0

        bvals = gbuy[c].dropna()
        svals = gsell[c].dropna()
        buy_ok = bool(
            len(bvals) >= 2 and ((bvals > 0).all() or (bvals < 0).all())
        )
        sell_ok = bool(
            len(svals) >= 2 and ((svals > 0).all() or (svals < 0).all())
        )

        blockers: list[str] = []
        if days < 2:
            blockers.append("less_than_2_days")
        if nbt + nst < MIN_TRADES_REGIME * 2:
            blockers.append("low_n")

        verdict = "unstable"
        reason: list[str] = list(blockers)

        if not blockers:
            if buy_ok:
                verdict = "trusted_buy_fwd_sign_stable"
                reason.append("buy_mean_ret_100_same_sign_all_days")
            elif sell_ok:
                verdict = "trusted_sell_fwd_sign_stable"
                reason.append("sell_mean_ret_100_same_sign_all_days")
            else:
                reason.append("buy_or_sell_sign_mixed_or_single_day")
        else:
            if buy_ok or sell_ok:
                verdict = "candidate_pattern_blocked_by_n_or_days"
                if buy_ok:
                    reason.append("buy_would_trust_if_more_days")
                if sell_ok:
                    reason.append("sell_would_trust_if_more_days")

        out_rows.append(
            {
                "mark": mark,
                "symbol": str(sym),
                "days_with_data": days,
                "n_trades_buy": nbt,
                "n_trades_sell": nst,
                "mean_fwd_buy_all_days": float(bvals.mean()) if len(bvals) else np.nan,
                "mean_fwd_sell_all_days": float(svals.mean()) if len(svals) else np.nan,
                "verdict": verdict,
                "notes": ";".join(reason) if reason else "review",
            }
        )

    res = pd.DataFrame(out_rows).sort_values(["verdict", "mark", "symbol"])
    p4 = paths["iter4_stability"]
    p4.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(p4, index=False)
    print(f"Wrote {p4} ({len(res)} rows)")


# =============================================================================
# Iter 5: execution notes (maps to r4 M14/67/38/55)
# =============================================================================


def run_iter5(out_base: Path) -> None:
    paths = output_paths(out_base)
    p4 = paths["iter4_stability"]
    p1 = paths["iter1_flow"]
    lines: list[str] = []
    lines.append("Mark research -> execution (Round 4, read-only notes)")
    lines.append("Algor: r4_m14_67_tilt_* style (HYDRO: follow 14, oppose 38; VELVET: follow 67, oppose 55).")
    lines.append("")
    lines.append("How to use this file:")
    lines.append(
        "- See iter4_stability/stability_trusted.csv: trusted_* rows -> consider tilts; "
        "unstable / candidate_* -> do not add rules without more data."
    )
    lines.append("- 'unstable' + low_n: do not add new rules; use default MM only.")
    lines.append("- PnL oscillation in backtests = inventory + MTM, not proof that a Mark is random.")
    lines.append("")

    if p1.is_file():
        f1 = pd.read_csv(p1)
        h = f1[(f1["symbol"] == "HYDROGEL_PACK") | (f1["symbol"] == "VELVETFRUIT_EXTRACT")]
        h = h.sort_values(["day", "mark"])
        lines.append("--- Snapshot: HYDRO / VELVET flow (iter1) ---")
        for _, r in h.iterrows():
            lines.append(
                f"  d{r['day']:.0f} {r['mark']:<8} {r['symbol'][:6]:<8} net={r['net_qty']:.0f} "
                f"buy_share={r['lot_buy_share']:.2f} n={r['n_trades']:.0f}"
            )
        lines.append("")

    if p4.is_file():
        t = pd.read_csv(p4)
        if len(t) > 0:
            lines.append("--- Iter4 verdicts (excerpt) ---")
            for v in (
                "trusted_buy_fwd_sign_stable",
                "trusted_sell_fwd_sign_stable",
                "unstable",
            ):
                sub = t[t["verdict"] == v]
                for _, r in sub.head(15).iterrows():
                    lines.append(
                        f"  {r['verdict']:<20} {r['mark']:<8} {r['symbol']:<20} {r.get('notes', '')}"
                    )
            lines.append("")

    lines.append("--- Suggested parameter discipline ---")
    lines.append("1. Cap |quote_tilt| at 1 tick per layer (as now); do not scale by Mark without replay validation.")
    lines.append("2. If iter3 mean_ret_100 for 'buy' is negative on 2+ days: fading aggressive buys is *consistent*; still check inventory risk.")
    lines.append("3. MMs (balanced buy/sell on iter1) but unstable iter4: use neutral skew; avoid dedicated fade.")
    lines.append("")

    outp = paths["iter5_exec"]
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {outp} ({len(lines)} lines)")


# =============================================================================
# main
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mark counterparty research pipeline (Round 4)"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_MARK_RESEARCH_OUT,
        help="Base output directory (default: misc/mark_research/; subfolders per phase)",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=DATA_R4,
        help="Round 4 data directory (default: Data/ROUND4)",
    )
    ap.add_argument("--iter0", action="store_true", help="Coverage + partners")
    ap.add_argument("--iter1", action="store_true", help="Flow stats")
    ap.add_argument("--iter2", action="store_true", help="Regime split")
    ap.add_argument("--iter3", action="store_true", help="Event study")
    ap.add_argument("--iter4", action="store_true", help="Stability table")
    ap.add_argument("--iter5", action="store_true", help="Execution notes")
    ap.add_argument(
        "--all", action="store_true", help="Run iter0..iter5 in order"
    )
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()

    run_all = args.all
    if not (run_all or any(
        (args.iter0, args.iter1, args.iter2, args.iter3, args.iter4, args.iter5)
    )):
        run_all = True

    data_r4 = args.data.resolve()

    if run_all or args.iter0:
        run_iter0(out_dir, data_r4)
    if run_all or args.iter1:
        run_iter1(out_dir, data_r4)
    if run_all or args.iter2:
        run_iter2(out_dir, data_r4)
    if run_all or args.iter3:
        run_iter3(out_dir, data_r4)
    if run_all or args.iter4:
        run_iter4(out_dir)
    if run_all or args.iter5:
        run_iter5(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
