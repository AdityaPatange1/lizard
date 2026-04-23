"""Entry point so ``python -m lizard`` mirrors the ``lizard`` console script."""

from __future__ import annotations

from lizard.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
