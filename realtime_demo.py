"""Project-root launcher for realtime demo.

Allows running:
    python realtime_demo.py --presentation-mode

Optional extras:
    python realtime_demo.py --presentation-mode --tts
    python realtime_demo.py --presentation-mode --speech-recognition
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "src" / "app" / "realtime_demo.py"
    runpy.run_path(str(target), run_name="__main__")
