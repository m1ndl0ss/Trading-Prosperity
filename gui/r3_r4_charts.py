"""
Side-by-side price + tape viewer for Prosperity CSVs (any two rounds/days/products).

- **Left / right charts** independently: round folder under ``Data/{ROUND}``, day,
  and product symbol.
- **Participant filter** appears per side only when that side's trade CSV has
  ``buyer`` / ``seller`` columns (typically ROUND4). Empty selection = show all trades.

Run from ``Trading-Prosperity`` root:

    .venv\\Scripts\\python.exe -m streamlit run gui/r3_r4_charts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "misc", _ROOT.parent):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from replay_engine import read_prices_csv, read_trades_csv
except ImportError:
    read_prices_csv = lambda p: pd.read_csv(p, sep=";")  # type: ignore
    read_trades_csv = lambda p: pd.read_csv(p, sep=";")  # type: ignore

ROOT = _ROOT
_DAY_RE = re.compile(r"day_(-?\d+)\.csv$", re.IGNORECASE)
def _scan_data_rounds() -> list[str]:
    """ROUND* folders under Data/ that have at least one paired prices+trades pair."""
    data = ROOT / "Data"
    if not data.is_dir():
        return []
    names: list[str] = []
    for child in sorted(data.iterdir()):
        if child.is_dir() and _day_map(child.name):
            names.append(child.name)
    return names


def trade_csv_has_buyer_seller(round_name: str) -> bool:
    dm = _day_map(round_name)
    if not dm:
        return False
    _pp, trades_path = next(iter(dm.values()))
    try:
        hdr = pd.read_csv(trades_path, sep=";", nrows=0).columns.tolist()
    except OSError:
        return False
    return "buyer" in hdr and "seller" in hdr


def participant_names_for_round(round_name: str) -> list[str]:
    names: set[str] = set()
    folder = ROOT / "Data" / round_name
    if not folder.is_dir():
        return []
    if not trade_csv_has_buyer_seller(round_name):
        return []
    for tf in folder.glob("trades_*.csv"):
        t = pd.read_csv(tf, sep=";", usecols=["buyer", "seller"])
        for col in ("buyer", "seller"):
            s = t[col].dropna().astype(str).str.strip()
            s = s[s != ""]
            names.update(s.unique())
    return sorted(names)


_PLOTLY_CONFIG: dict = {
    "scrollZoom": True,
    "displayModeBar": True,
    "doubleClick": "reset",
}


def _day_map(round_name: str) -> dict[int, tuple[Path, Path]]:
    folder = ROOT / "Data" / round_name
    out: dict[int, tuple[Path, Path]] = {}
    if not folder.is_dir():
        return out
    for pf in folder.glob("prices_*.csv"):
        m = _DAY_RE.search(pf.name)
        if not m:
            continue
        d = int(m.group(1))
        tf = folder / pf.name.replace("prices_", "trades_", 1)
        if tf.is_file():
            out[d] = (pf, tf)
    return dict(sorted(out.items()))


def _products_for_day(prices_path: Path) -> list[str]:
    df = pd.read_csv(prices_path, sep=";", usecols=["product"])
    return sorted(df["product"].unique().tolist())


def _mid_series(prices_path: Path, product: str) -> pd.DataFrame:
    df = read_prices_csv(prices_path)
    sub = df[df["product"] == product].sort_values("timestamp")
    return sub[["timestamp", "mid_price"]].rename(columns={"mid_price": "mid"})


def filter_trades_by_participants(
    trades: pd.DataFrame,
    product: str,
    participants: list[str],
    role: str,
) -> pd.DataFrame:
    """Keep trades for *product* where any *participant* matches buyer/seller per *role*."""
    t = trades[trades["symbol"] == product].copy()
    if not participants:
        return t
    b = t["buyer"].fillna("").astype(str).str.strip()
    s = t["seller"].fillna("").astype(str).str.strip()
    mask = pd.Series(False, index=t.index)
    for p in participants:
        if role in ("Either side", "Buyer only"):
            mask |= b == p
        if role in ("Either side", "Seller only"):
            mask |= s == p
    return t.loc[mask]


def finalize_trades_for_side(
    trades_full: pd.DataFrame,
    product: str,
    round_name: str,
    participants: list[str],
    role: str,
) -> pd.DataFrame:
    """Symbol slice; optionally filter by counterparties when the tape has buyer/seller columns."""
    sym = trades_full[trades_full["symbol"] == product].copy()
    if not participants:
        return sym
    if not trade_csv_has_buyer_seller(round_name):
        return sym
    return filter_trades_by_participants(trades_full, product, participants, role)


def hover_names_useful(trades_subset: pd.DataFrame) -> bool:
    """Use buyer/seller hover text only when columns exist and at least one side is non-empty."""
    if trades_subset.empty:
        return False
    if "buyer" not in trades_subset.columns or "seller" not in trades_subset.columns:
        return False
    buyer = trades_subset["buyer"].fillna("").astype(str).str.strip()
    seller = trades_subset["seller"].fillna("").astype(str).str.strip()
    return bool(((buyer != "") | (seller != "")).any())


def default_product_choice(products: list[str]) -> tuple[str, int]:
    pref = ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT")
    for p in pref:
        if p in products:
            return p, products.index(p)
    return products[0], 0


def _trade_scatter(trades: pd.DataFrame, title_suffix: str, show_names: bool) -> go.Figure:
    fig = go.Figure()
    if trades.empty:
        fig.update_layout(
            title=f"Trades {title_suffix} (none)",
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    hover = []
    for _, r in trades.iterrows():
        buyer = str(r.get("buyer", "") or "").strip()
        seller = str(r.get("seller", "") or "").strip()
        qty = int(r.get("quantity", 0) or 0)
        if show_names and (buyer or seller):
            hover.append(f"buyer={buyer or '—'}<br>seller={seller or '—'}<br>qty={qty}")
        else:
            hover.append(f"qty={qty}")

    fig.add_trace(
        go.Scatter(
            x=trades["timestamp"],
            y=trades["price"],
            mode="markers",
            marker=dict(
                size=6 + trades["quantity"].clip(upper=20) * 0.35,
                color="#EF553B",
                opacity=0.85,
                line=dict(width=0.5, color="white"),
            ),
            text=hover,
            hoverinfo="text+x+y",
            name="Trades",
        )
    )
    fig.update_layout(
        title=f"Trades {title_suffix} (n={len(trades)})",
        xaxis_title="timestamp",
        yaxis_title="price",
        height=400,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def _combined_figure(
    mid: pd.DataFrame,
    trades: pd.DataFrame,
    panel_title: str,
    show_trade_hover_names: bool,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mid["timestamp"],
            y=mid["mid"],
            mode="lines",
            name="Mid",
            line=dict(color="#636EFA", width=1.2),
        )
    )
    if not trades.empty:
        hover = []
        for _, r in trades.iterrows():
            buyer = str(r.get("buyer", "") or "").strip()
            seller = str(r.get("seller", "") or "").strip()
            qty = int(r.get("quantity", 0) or 0)
            if show_trade_hover_names and (buyer or seller):
                hover.append(f"buyer={buyer or '—'}<br>seller={seller or '—'}<br>qty={qty}")
            else:
                hover.append(f"qty={qty}")
        fig.add_trace(
            go.Scatter(
                x=trades["timestamp"],
                y=trades["price"],
                mode="markers",
                name="Trades",
                marker=dict(
                    size=5 + trades["quantity"].clip(upper=24) * 0.3,
                    color="#EF553B",
                    opacity=0.75,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover,
                hoverinfo="text+x+y",
            )
        )
    fig.update_layout(
        title=panel_title,
        xaxis_title="timestamp",
        yaxis_title="price",
        height=480,
        margin=dict(l=50, r=20, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Prosperity — compare rounds", layout="wide")
    st.title("Prices & trades — side-by-side (two independent panels)")

    avail = _scan_data_rounds()
    if len(avail) == 0:
        st.error("No folders under **Data/** with paired `prices_*.csv` / `trades_*.csv`.")
        st.stop()

    rl_default = avail.index("ROUND3") if "ROUND3" in avail else 0
    rr_default = avail.index("ROUND4") if "ROUND4" in avail else min(1, len(avail) - 1)
    if rl_default == rr_default and len(avail) > 1:
        rr_default = (rr_default + 1) % len(avail)

    with st.sidebar:
        st.header("Left chart")
        round_l = st.selectbox(
            "Round",
            options=avail,
            index=rl_default,
            key="r4cmp_round_l",
        )
        dm_l = _day_map(round_l)
        day_keys_l = list(dm_l.keys())
        day_l = st.selectbox(
            "Day",
            options=day_keys_l,
            format_func=lambda d: f"Day {d}",
            key="r4cmp_day_l",
            index=0,
        )
        plist_l = _products_for_day(dm_l[day_l][0])
        _dp_l, ix_l = default_product_choice(plist_l)
        prod_l = st.selectbox(
            "Product / symbol",
            plist_l,
            index=min(ix_l, len(plist_l) - 1),
            key="r4cmp_prod_l",
        )

        st.divider()
        st.header("Right chart")
        round_r = st.selectbox(
            "Round",
            options=avail,
            index=min(rr_default, len(avail) - 1),
            key="r4cmp_round_r",
        )
        dm_r = _day_map(round_r)
        day_keys_r = list(dm_r.keys())
        day_r = st.selectbox(
            "Day",
            options=day_keys_r,
            format_func=lambda d: f"Day {d}",
            key="r4cmp_day_r",
            index=0,
        )
        plist_r = _products_for_day(dm_r[day_r][0])
        _dp_r, ix_r = default_product_choice(plist_r)
        prod_r = st.selectbox(
            "Product / symbol",
            plist_r,
            index=min(ix_r, len(plist_r) - 1),
            key="r4cmp_prod_r",
        )

        def _participant_block(side: str, rn: str) -> tuple[list[str], str]:
            if not trade_csv_has_buyer_seller(rn):
                return [], "Either side"
            st.divider()
            st.subheader(f'{side}: tape filter ({rn})')
            opts = participant_names_for_round(rn)
            pk = st.multiselect(
                "Participant(s)",
                options=opts,
                default=[],
                key=f"r4cmp_part_{side}_{rn}",
                help="Empty = all trades for this symbol.",
            )
            rl = st.radio(
                "Match role",
                ("Either side", "Buyer only", "Seller only"),
                key=f"r4cmp_role_{side}_{rn}",
                horizontal=False,
            )
            return pk, rl

        part_l, role_l = _participant_block("Left", round_l)
        part_r, role_r = _participant_block("Right", round_r)

        st.divider()
        show_split_trades = st.checkbox("Extra row: trades-only panels", value=False)

        st.caption(
            "Tape filters apply only when that round's trades CSV includes **buyer/seller** "
            "(typically ROUND4). Empty participant list = all prints on that side."
        )

    paths_l = dm_l[day_l]
    paths_r = dm_r[day_r]
    prices_l_path, trades_l_path = paths_l
    prices_r_path, trades_r_path = paths_r

    mid_l = _mid_series(prices_l_path, prod_l)
    mid_r = _mid_series(prices_r_path, prod_r)

    raw_l = read_trades_csv(trades_l_path)
    raw_r = read_trades_csv(trades_r_path)
    t_left = finalize_trades_for_side(raw_l, prod_l, round_l, part_l, role_l)
    t_right = finalize_trades_for_side(raw_r, prod_r, round_r, part_r, role_r)

    names_l = hover_names_useful(t_left)
    names_r = hover_names_useful(t_right)

    title_l = f"{round_l} · Day {day_l} · {prod_l} (tape: {len(t_left)})"
    title_r = f"{round_r} · Day {day_r} · {prod_r} (tape: {len(t_right)})"

    fig_l = _combined_figure(mid_l, t_left, title_l, show_trade_hover_names=names_l)
    fig_r = _combined_figure(mid_r, t_right, title_r, show_trade_hover_names=names_r)

    chart_l, chart_r = st.columns(2, gap="large")
    with chart_l:
        st.plotly_chart(fig_l, use_container_width=True, config=_PLOTLY_CONFIG)
    with chart_r:
        st.plotly_chart(fig_r, use_container_width=True, config=_PLOTLY_CONFIG)

    if show_split_trades:
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _trade_scatter(t_left, f"{round_l} Day {day_l} — {prod_l}", names_l),
                use_container_width=True,
                config=_PLOTLY_CONFIG,
            )
        with c2:
            st.plotly_chart(
                _trade_scatter(t_right, f"{round_r} Day {day_r} — {prod_r}", names_r),
                use_container_width=True,
                config=_PLOTLY_CONFIG,
            )

    with st.expander("Trade tables"):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"**Left — {round_l}**")
            st.dataframe(t_left, use_container_width=True, height=260)
        with tc2:
            st.markdown(f"**Right — {round_r}**")
            st.dataframe(t_right, use_container_width=True, height=260)


if __name__ == "__main__":
    main()
