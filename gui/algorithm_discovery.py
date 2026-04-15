"""
Discover `Trader` classes from `trading_algos/*.py` (flat) or
`trading_algos/{subdir}/*.py` (round-scoped, e.g. TUTORIAL or ROUND1).

Each module may trade one or more products (EMERALDS, TOMATOES, …):
  - `PRODUCT = "…"` / `SYMBOL = "…"` / `PRODUCTS = ("A", "B")`, or
  - module-level string constants whose values match CSV `product` columns.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, FrozenSet, List, Type

_PRODUCT_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

ALGO_PKG = "trading_algos"


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def trading_algorithms_fingerprint(subdir: str | None = None) -> tuple[int, ...]:
    """Mtimes so Streamlit cache invalidates when any algo file changes."""
    base = _root() / ALGO_PKG
    if subdir:
        d = base / subdir
    else:
        d = base
    if not d.is_dir():
        return ()
    return tuple(
        p.stat().st_mtime_ns
        for p in sorted(d.glob("*.py"))
        if not p.name.startswith("_")
    )


def _infer_products(mod: ModuleType, known_products: FrozenSet[str] | None) -> frozenset[str]:
    explicit: set[str] = set()
    products_attr = getattr(mod, "PRODUCTS", None)
    if isinstance(products_attr, (list, tuple, set, frozenset)):
        for x in products_attr:
            if isinstance(x, str) and x.strip():
                p = x.strip()
                if known_products is None or p in known_products:
                    explicit.add(p)
    for key in ("PRODUCT", "SYMBOL"):
        v = getattr(mod, key, None)
        if isinstance(v, str) and v.strip():
            p = v.strip()
            if known_products is None or p in known_products:
                explicit.add(p)
    if explicit:
        return frozenset(explicit)

    candidates: set[str] = set()
    for name, val in vars(mod).items():
        if name.startswith("_"):
            continue
        if not isinstance(val, str) or not val:
            continue
        if _PRODUCT_RE.match(val):
            candidates.add(val)

    if known_products is not None:
        hit = candidates & known_products
        if hit:
            return frozenset(hit)
        return frozenset()

    if 1 <= len(candidates) <= 4:
        return frozenset(candidates)
    return frozenset()


@dataclass(frozen=True)
class DiscoveredStrategy:
    strategy_id: str
    label: str
    products: frozenset[str]
    trader_cls: Type[Any]


def discover_strategies(
    known_products: FrozenSet[str] | None,
    subdir: str | None = None,
) -> List[DiscoveredStrategy]:
    """
    Scan for Trader classes.

    If *subdir* is given (e.g. "TUTORIAL"), only `trading_algos/{subdir}/*.py`
    is scanned and module names are `trading_algos.{subdir}.{stem}`.
    Otherwise the flat `trading_algos/*.py` root is used.
    """
    base = _root() / ALGO_PKG
    if subdir:
        pkg_dir = base / subdir
        mod_prefix = f"{ALGO_PKG}.{subdir}"
    else:
        pkg_dir = base
        mod_prefix = ALGO_PKG

    out: List[DiscoveredStrategy] = []
    if not pkg_dir.is_dir():
        return out

    for path in sorted(pkg_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod_name = f"{mod_prefix}.{path.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        t_cls = getattr(mod, "Trader", None)
        if not isinstance(t_cls, type):
            continue
        prods = _infer_products(mod, known_products)
        if not prods:
            continue
        if known_products is not None and not (prods & known_products):
            continue
        nice = path.stem.replace("_", " ")
        prod_str = ", ".join(sorted(prods))
        out.append(
            DiscoveredStrategy(
                strategy_id=path.stem,
                label=f"{nice} · {prod_str}",
                products=prods,
                trader_cls=t_cls,
            )
        )
    return out


def find_strategy(
    known_products: FrozenSet[str] | None,
    strategy_id: str,
    product: str | None = None,
    subdir: str | None = None,
) -> DiscoveredStrategy | None:
    for d in discover_strategies(known_products, subdir=subdir):
        if d.strategy_id != strategy_id:
            continue
        if product is None or product in d.products:
            return d
    return None


def strategy_goal_hint(strategy_id: str, subdir: str | None = None) -> str | None:
    if subdir:
        mod_name = f"{ALGO_PKG}.{subdir}.{strategy_id}"
    else:
        mod_name = f"{ALGO_PKG}.{strategy_id}"
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return None
    hint = getattr(mod, "GOAL_HINT", None)
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    return None
