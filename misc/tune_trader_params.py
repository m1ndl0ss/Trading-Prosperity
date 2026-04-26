#!/usr/bin/env python3
"""
Hyperparameter search on tape replay: maximize PnL on Round 1 (all days) plus one
Round 2 day, then report validation PnL on every Round 2 day.

Requires ``Trader`` subclasses that copy module ``PARAMS`` into ``self.params`` in
``__init__`` (e.g. osmium / starfruit-style algos in this repo).

Usage (from Trading-Prosperity repo root):

  python tune_trader_params.py \\
    --trader trading_algos/ROUND2/224853_osmium_only.py \\
    --r2-train-day -1 \\
    --objective-product ASH_COATED_OSMIUM \\
    --space tuning_spaces/osmium_starfruit.json \\
    --trials 80

With Optuna (``pip install optuna``) the study uses TPE; otherwise a reproducible
random search runs for the same number of trials.

  python tune_trader_params.py --trader ... --space tuning_spaces/osmium_starfruit.json \\
    --r2-train-day 0 --trials 50 --val-exclude-r2-train-day
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tape_benchmark import (
    discover_days,
    discover_products_union,
    load_trader,
    pnl_for_day,
)

try:
    import optuna

    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


def _load_space(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        alt = ROOT / path
        if alt.is_file():
            path = alt
        else:
            raise SystemExit(f"Space file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise SystemExit("Space JSON must be a non-empty object: {param: spec, ...}")
    for name, spec in data.items():
        if not isinstance(spec, dict):
            raise SystemExit(f"Invalid spec for {name!r}")
        t = spec.get("type")
        if t not in ("float", "int", "categorical"):
            raise SystemExit(f"{name}: type must be float, int, or categorical")
        if t == "categorical":
            ch = spec.get("choices")
            if not isinstance(ch, list) or len(ch) < 1:
                raise SystemExit(f"{name}: categorical needs non-empty choices")
        else:
            if "low" not in spec or "high" not in spec:
                raise SystemExit(f"{name}: {t} needs low and high")
    return data


def _suggest_optuna(trial: Any, space: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "float":
            log = bool(spec.get("log", False))
            lo, hi = float(spec["low"]), float(spec["high"])
            out[name] = trial.suggest_float(name, lo, hi, log=log)
        elif t == "int":
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            out[name] = trial.suggest_categorical(name, list(spec["choices"]))
    return out


def _sample_random(rng: random.Random, space: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "float":
            lo, hi = float(spec["low"]), float(spec["high"])
            if spec.get("log") and lo > 0 and hi > 0:
                out[name] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            else:
                out[name] = rng.uniform(lo, hi)
        elif t == "int":
            out[name] = rng.randint(int(spec["low"]), int(spec["high"]))
        else:
            choices = list(spec["choices"])
            out[name] = rng.choice(choices)
    return out


def make_tuning_trader_class(base_cls: Type, param_updates: Dict[str, Any]) -> Type:
    """Subclass that merges ``param_updates`` into ``self.params`` after ``super().__init__``."""

    class TuningTrader(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            if not hasattr(self, "params") or not isinstance(self.params, dict):
                raise TypeError(
                    f"{base_cls.__name__} must set self.params to a dict in __init__ for tuning."
                )
            merged = dict(self.params)
            merged.update(param_updates)
            self.params = merged

    TuningTrader.__name__ = f"Tuning_{base_cls.__name__}"
    return TuningTrader


def _train_pairs(
    data_root: Path,
    r2_train_day: str,
) -> Tuple[List[Tuple[str, Path, Path]], List[Tuple[str, Path, Path]]]:
    r1_dir = data_root / "ROUND1"
    r2_dir = data_root / "ROUND2"
    if not r1_dir.is_dir():
        raise SystemExit(f"Missing {r1_dir}")
    if not r2_dir.is_dir():
        raise SystemExit(f"Missing {r2_dir}")
    r1_days = discover_days(r1_dir)
    r2_days = discover_days(r2_dir)
    r2_sel = [d for d in r2_days if d[0] == r2_train_day]
    if not r2_sel:
        avail = ", ".join(d[0] for d in r2_days)
        raise SystemExit(
            f"No Round 2 day {r2_train_day!r} under {r2_dir}. Available: {avail}"
        )
    train = list(r1_days) + [r2_sel[0]]
    return train, r2_days


def _score_train(
    trader_cls: Type,
    train_pairs: List[Tuple[str, Path, Path]],
    products: List[str],
    initial_cash: float,
    objective_product: str | None,
) -> float:
    total = 0.0
    for _dk, p_csv, t_csv in train_pairs:
        if objective_product:
            total += float(
                pnl_for_day(trader_cls, p_csv, t_csv, objective_product, initial_cash)["pnl"]
            )
        else:
            day_sum = 0.0
            for sym in products:
                day_sum += float(pnl_for_day(trader_cls, p_csv, t_csv, sym, initial_cash)["pnl"])
            total += day_sum
    return total


def _val_rows(
    trader_cls: Type,
    r2_days: List[Tuple[str, Path, Path]],
    products: List[str],
    initial_cash: float,
    objective_product: str | None,
) -> Tuple[List[Tuple[str, float, Dict[str, float]]], float]:
    rows: List[Tuple[str, float, Dict[str, float]]] = []
    for day_key, p_csv, t_csv in r2_days:
        per: Dict[str, float] = {}
        for sym in products:
            per[sym] = float(pnl_for_day(trader_cls, p_csv, t_csv, sym, initial_cash)["pnl"])
        combined = per[objective_product] if objective_product else sum(per.values())
        rows.append((day_key, combined, per))
    total = sum(r[1] for r in rows)
    return rows, total


def _print_val_table(
    rows: List[Tuple[str, float, Dict[str, float]]],
    objective_product: str | None,
    label: str,
) -> None:
    print(label)
    print(f"{'Day':<10}{'Objective PnL':>16}")
    print("-" * 26)
    for day_key, obj_pnl, _per in rows:
        print(f"{day_key:<10}{obj_pnl:16,.3f}")
    print("-" * 26)
    print(f"{'Sum':<10}{sum(r[1] for r in rows):16,.3f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tune trader PARAMS on R1 + one R2 day; validate on Round 2."
    )
    ap.add_argument(
        "--trader",
        type=Path,
        required=True,
        help="Path to trader .py (must use self.params dict)",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Data root (default: <repo>/Data)",
    )
    ap.add_argument(
        "--r2-train-day",
        required=True,
        help="Round 2 day key included in training (e.g. -1, 0, 1)",
    )
    ap.add_argument(
        "--objective-product",
        default=None,
        metavar="SYMBOL",
        help="Maximize this product's PnL only; omit for combined PnL (all products)",
    )
    ap.add_argument(
        "--space",
        type=Path,
        required=True,
        help="JSON search space (float / int / categorical)",
    )
    ap.add_argument("--trials", type=int, default=50, help="Number of trials (default 50)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for random search / Optuna")
    ap.add_argument("--cash", type=float, default=0.0, help="Initial cash for replay")
    ap.add_argument(
        "--val-exclude-r2-train-day",
        action="store_true",
        help="Also print validation rows excluding the R2 training day",
    )
    args = ap.parse_args()

    data_root = args.data if args.data is not None else ROOT / "Data"
    trader_path = args.trader
    if not trader_path.is_file():
        alt = ROOT / trader_path
        if alt.is_file():
            trader_path = alt
        else:
            raise SystemExit(f"Trader file not found: {args.trader}")

    r2_train_key = str(args.r2_train_day).strip()
    space = _load_space(args.space)
    base_cls = load_trader(trader_path.resolve())
    train_pairs, r2_days = _train_pairs(data_root, r2_train_key)
    products = discover_products_union(train_pairs)
    if args.objective_product and args.objective_product not in products:
        raise SystemExit(
            f"Objective product {args.objective_product!r} not in training tape union: {products}"
        )

    best_params: Dict[str, Any] | None = None
    best_score = float("-inf")

    mode = "optuna" if _HAS_OPTUNA else "random"
    print(f"Optimizer: {mode} (install optuna for TPE search)")
    print(f"Trader:    {trader_path}")
    print(f"Training:  ROUND1 all days + ROUND2 day {r2_train_key!r} ({len(train_pairs)} day-files)")
    print(f"Objective: {'combined PnL' if not args.objective_product else args.objective_product}")
    print(f"Trials:    {args.trials}")
    print()

    if _HAS_OPTUNA:

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_optuna(trial, space)
            cls = make_tuning_trader_class(base_cls, params)
            return _score_train(cls, train_pairs, products, args.cash, args.objective_product)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        best_score = float(study.best_value)
    else:
        rng = random.Random(args.seed)
        for _ in range(args.trials):
            params = _sample_random(rng, space)
            cls = make_tuning_trader_class(base_cls, params)
            sc = _score_train(cls, train_pairs, products, args.cash, args.objective_product)
            if sc > best_score:
                best_score = sc
                best_params = dict(params)

    if best_params is None:
        raise SystemExit("No trial completed")

    print(f"Best train score: {best_score:,.3f}")
    print("Best params (merge into your trader PARAMS):")
    print(json.dumps(best_params, indent=2, sort_keys=True))
    print()

    best_cls = make_tuning_trader_class(base_cls, best_params)
    r2_products = discover_products_union(r2_days)
    val_prods = sorted(set(products) | set(r2_products))
    val_rows, _ = _val_rows(best_cls, r2_days, val_prods, args.cash, args.objective_product)

    obj_label = args.objective_product or "combined"
    _print_val_table(val_rows, args.objective_product, f"Validation: ROUND2 all days ({obj_label})")

    if args.val_exclude_r2_train_day:
        strict = [r for r in val_rows if r[0] != r2_train_key]
        if strict:
            _print_val_table(
                strict,
                args.objective_product,
                f"Strict holdout: ROUND2 excluding train day {r2_train_key!r} ({obj_label})",
            )


if __name__ == "__main__":
    main()
