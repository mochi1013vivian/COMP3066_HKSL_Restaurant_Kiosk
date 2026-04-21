"""Backward-compatible entrypoint for the real-time app.

Run:
    python src/app.py
"""

from __future__ import annotations

try:
    from .inference_realtime import main
except ImportError:  # pragma: no cover - supports `python src/app.py`
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from inference_realtime import main


if __name__ == "__main__":
    main()
