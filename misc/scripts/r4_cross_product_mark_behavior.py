"""Ad-hoc: cross-product timing for Round 4 marks (tape + mids). Run from repo root."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

TROOT = Path(__file__).resolve().parents[2] / "Data" / "ROUND4"
HY, VE = "HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"


def load_trades() -> pd.DataFrame:
    dfs = []
    for d in (1, 2, 3):
        df = pd.read_csv(TROOT / f"trades_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_mids_day(day: int) -> pd.DataFrame:
    pr = pd.read_csv(TROOT / f"prices_round_4_day_{day}.csv", sep=";")
    pr = pr[pr["product"].isin([HY, VE])][["timestamp", "product", "mid_price"]].copy()
    h = pr[pr["product"] == HY].rename(columns={"mid_price": "hyd_mid"}).drop(columns=["product"])
    v = pr[pr["product"] == VE].rename(columns={"mid_price": "vel_mid"}).drop(columns=["product"])
    m = h.merge(v, on="timestamp", how="outer").sort_values("timestamp")
    m["hyd_d"] = m["hyd_mid"].diff()
    m["vel_d"] = m["vel_mid"].diff()
    m["day"] = day
    return m


def main() -> None:
    tr = load_trades()
    flat: list[tuple] = []
    for _, r in tr.iterrows():
        t, d, sym = int(r["timestamp"]), int(r["day"]), r["symbol"]
        q = int(r["quantity"])
        for m, sgn in ((r["buyer"], 1), (r["seller"], -1)):
            cat = "HYDRO" if sym == HY else "VELVET" if sym == VE else "VEV"
            flat.append((t, d, sym, m, sgn, q, cat))
    F = pd.DataFrame(
        flat,
        columns=["t", "day", "sym", "mark", "side", "qty", "cat"],
    )

    # 1) Same-timestamp multi-category
    co = (
        F.groupby(["day", "t", "mark"])[["cat"]]
        .apply(lambda g: frozenset(g["cat"].unique()))
        .reset_index(name="cats")
    )
    multi = co[co["cats"].apply(len) >= 2]
    print("=== Same-timestamp: mark has 2+ categories (HYDRO/VELVET/VEV) at one timestamp ===")
    print(multi.groupby("mark").size().sort_values(ascending=False).to_string())
    print()

    def cond_side(mark: str, fromcat: str, tocat: str, win: int) -> pd.DataFrame:
        rows: list[tuple] = []
        for day in (1, 2, 3):
            sub = F[(F.day == day) & (F.mark == mark)].sort_values("t")
            cats = sub["cat"].values
            sides = sub["side"].values
            times = sub["t"].values
            for i in range(len(sub) - 1):
                if cats[i] != fromcat:
                    continue
                s0 = sides[i]
                for j in range(i + 1, len(sub)):
                    if times[j] > times[i] + win:
                        break
                    if cats[j] == tocat:
                        rows.append((s0, sides[j]))
                        break
        return pd.DataFrame(rows, columns=["from_side", "to_side"])

    print("=== Mark 14: first VELVET within 400t after HYDRO (side +1 buy -1 sell) ===")
    c = cond_side("Mark 14", "HYDRO", "VELVET", 400)
    print("n transitions", len(c))
    if len(c):
        print(pd.crosstab(c.from_side, c.to_side, margins=True))
        b = c[c.from_side == 1]
        s = c[c.from_side == -1]
        if len(b):
            print("P(vel buy | hyd buy) =", (b.to_side == 1).mean(), f"n={len(b)}")
        if len(s):
            print("P(vel buy | hyd sell) =", (s.to_side == 1).mean(), f"n={len(s)}")
    print()

    print("=== Mark 38: first VEV within 400t after HYDRO ===")
    c38 = cond_side("Mark 38", "HYDRO", "VEV", 400)
    print("n", len(c38))
    if len(c38):
        print(pd.crosstab(c38.from_side, c38.to_side, margins=True))
    print()

    print("=== Mark 01: first VEV within 200t after VELVET (shell stacking) ===")
    c01 = cond_side("Mark 01", "VELVET", "VEV", 200)
    print("n", len(c01))
    if len(c01):
        b = c01[c01.from_side == 1]
        s = c01[c01.from_side == -1]
        print("P(VEV buy after Velvet buy) =", (b.to_side == 1).mean() if len(b) else np.nan, f"n={len(b)}")
        print(
            "P(VEV buy after Velvet sell) =",
            (s.to_side == 1).mean() if len(s) else np.nan,
            f"n={len(s)}",
        )
    print()

    # Hydro mid shock -> net velvet qty for mark in (t0, t0+H]
    def shock_velvet(mark: str, thresh: float, horizon: int) -> pd.DataFrame:
        rows: list[tuple] = []
        for day in (1, 2, 3):
            mids = load_mids_day(day).dropna(subset=["hyd_mid"])
            mids["absd"] = mids["hyd_d"].abs()
            for _, row in mids[mids.absd >= thresh].iterrows():
                t0 = int(row["timestamp"])
                trd = tr[
                    (tr.day == day)
                    & (tr.timestamp > t0)
                    & (tr.timestamp <= t0 + horizon)
                    & (tr.symbol == VE)
                ]
                net = 0
                for _, r in trd.iterrows():
                    q = int(r["quantity"])
                    if r["buyer"] == mark:
                        net += q
                    if r["seller"] == mark:
                        net -= q
                rows.append((day, t0, float(row["hyd_d"]), net))
        return pd.DataFrame(rows, columns=["day", "t", "hyd_move", "vel_net_qty"])

    print(f"=== Hydro mid move |d|>={8} (coarser shock) -> net VELVET qty in next {400}t ===")
    for mk in ("Mark 14", "Mark 67", "Mark 55"):
        sh = shock_velvet(mk, 8.0, 400)
        if len(sh) == 0:
            continue
        up = sh[sh.hyd_move > 0]
        dn = sh[sh.hyd_move < 0]
        print(
            mk,
            "events",
            len(sh),
            "mean vel_net after UP",
            up.vel_net_qty.mean() if len(up) else np.nan,
            "after DN",
            dn.vel_net_qty.mean() if len(dn) else np.nan,
        )
    print()

    print(f"=== Velvet mid |d|>={5} -> net HYDRO qty next {400}t (14/38) ===")
    rows = []
    for day in (1, 2, 3):
        mids = load_mids_day(day).dropna(subset=["vel_mid"])
        for _, row in mids[mids.vel_d.abs() >= 5.0].iterrows():
            t0 = int(row["timestamp"])
            for mark in ("Mark 14", "Mark 38"):
                trd = tr[
                    (tr.day == day)
                    & (tr.timestamp > t0)
                    & (tr.timestamp <= t0 + 400)
                    & (tr.symbol == HY)
                ]
                net = 0
                for _, r in trd.iterrows():
                    q = int(r["quantity"])
                    if r["buyer"] == mark:
                        net += q
                    if r["seller"] == mark:
                        net -= q
                rows.append((mark, float(row["vel_d"]), net))
    RV = pd.DataFrame(rows, columns=["mark", "vel_move", "hyd_net"])
    for mark in ("Mark 14", "Mark 38"):
        sub = RV[RV.mark == mark]
        up = sub[sub.vel_move > 0]
        dn = sub[sub.vel_move < 0]
        print(
            mark,
            "events",
            len(sub),
            "hyd net after vel UP",
            up.hyd_net.mean() if len(up) else np.nan,
            "after vel DN",
            dn.hyd_net.mean() if len(dn) else np.nan,
        )


if __name__ == "__main__":
    main()
