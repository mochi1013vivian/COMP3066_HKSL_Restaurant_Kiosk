"""Entrypoint for realtime HKSL PyTorch demo."""

from __future__ import annotations

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parent / "app" / "realtime_demo.py"
    runpy.run_path(str(script), run_name="__main__")
