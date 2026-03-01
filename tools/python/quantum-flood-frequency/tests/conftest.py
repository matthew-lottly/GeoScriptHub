"""Pytest configuration – ensure ``src/`` is on the import path."""

from __future__ import annotations

import sys
from pathlib import Path

# Insert the ``src`` directory so ``import quantum_flood_frequency``
# resolves without ``pip install -e .``
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
