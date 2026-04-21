"""Audio feedback helper for accepted token events.

Provides soft confirmation chime only when words are added to final sentence.
No sound during unstable detection (live prediction stage).
Optional text-to-speech on demand.
"""

from __future__ import annotations

import os
import sys


def play_confirmation_sound(enabled: bool = True) -> None:
    """Play a soft confirmation chime when a word is added to the final sentence."""
    if not enabled:
        return

    try:
        if sys.platform == "darwin":
            # macOS: Use Bell or Glass sound (softer than Glass.aiff)
            os.system("afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1 &")
        else:
            # Other platforms: quiet beep
            print("\a", end="", flush=True)
    except Exception:
        pass


def play_accept_sound(enabled: bool = True) -> None:
    """Deprecated: use play_confirmation_sound instead."""
    play_confirmation_sound(enabled)


def speak_text(text: str, voice: str = "Victoria") -> None:
    """Use system text-to-speech to speak the given text (macOS only).
    
    Args:
        text: Text to speak
        voice: Voice name (macOS: Victoria, Alex, etc.)
    """
    if not text or not text.strip():
        return
    
    try:
        if sys.platform == "darwin":
            # Escape quotes for shell
            safe_text = text.replace('"', '\\"')
            os.system(f'echo "{safe_text}" | say -v {voice} 2>/dev/null &')
        else:
            # Linux/Windows: placeholder (would need espeak or similar)
            print(f"[TTS] {text}", file=sys.stderr)
    except Exception:
        pass
