"""Deprecated: use `python misc/counterparty_eod_pnl.py "Mark 38"`."""
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    sys.argv = [sys.argv[0], "Mark 38"] + sys.argv[1:]
    from counterparty_eod_pnl import main  # noqa: E402

    raise SystemExit(main())
