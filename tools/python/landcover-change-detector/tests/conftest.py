"""Pytest configuration — add src/ to sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure local src importable
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
