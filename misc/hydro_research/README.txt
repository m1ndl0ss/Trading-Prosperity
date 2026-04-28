HYDROGEL_PACK local research (oscillation stats + strategy tape replay)
========================================================================

Layout
------
  oscillation/     Part A: mid/spread/ACF stats and R3 Donchian JSON
                   Script: hydro_oscillation_stats.py

  strategies/      HYDRO-only Trader ablations and production extracts
                   Module: hydro_strategy_ablations.py

  compare/         Part C: run all strategies x R3d0, R4d1, R4d2
                   Script: hydro_compare_strategies.py

  output/
    oscillation/   hydro_oscillation_stats.csv, hydro_donchian_r3d0.json
    compare/       hydro_strategy_matrix.csv, hydro_research_notes.txt

Shared engine: ../replay_engine.py (market_trades injection for Mark logic).

From repo root (Trading-Prosperity), run:
  python misc/hydro_research/oscillation/hydro_oscillation_stats.py
  python misc/hydro_research/compare/hydro_compare_strategies.py

Or from misc:
  python hydro_research/oscillation/hydro_oscillation_stats.py
  python hydro_research/compare/hydro_compare_strategies.py
