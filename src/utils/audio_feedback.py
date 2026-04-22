"""Audio feedback helper for accepted token events.

Provides soft confirmation chime only when words are added to final sentence.
No sound during unstable detection (live prediction stage).
Optional text-to-speech on demand.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional, Tuple


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


def recognize_speech_once(timeout: float = 4.0, phrase_time_limit: float = 6.0) -> Optional[str]:
    """Capture one microphone utterance and return recognized text.

    Uses SpeechRecognition (Google Web Speech API backend).
    Returns None when recognition fails or dependency/hardware is unavailable.
    """
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        return None


def normalize_waiter_phrase(text: str) -> Optional[str]:
    """Normalize free-form speech to target waiter phrases.

    Returns one of:
    - "No problem"
    - "What can I help you?"
    - "How would you like to pay?"
    """
    if not text:
        return None

    norm = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return None

    if any(key in norm for key in ("no problem", "no problems", "not a problem", "sure")):
        return "No problem"

    if any(
        key in norm
        for key in (
            "what can i help you",
            "what can i help you with",
            "how can i help you",
            "how may i help you",
        )
    ):
        return "What can I help you?"

    if any(
        key in norm
        for key in (
            "how would you like to pay",
            "how do you like to pay",
            "how would you pay",
            "how are you paying",
            "method of payment",
            "cash or card",
        )
    ):
        return "How would you like to pay?"

    return None

    try:
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.7

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        text = recognizer.recognize_google(audio)
        text = text.strip()
        return text or None
    except Exception:
        return None


def speech_backend_status() -> str:
    """Return a short status string for speech-recognition readiness."""
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        return "SpeechRecognition not installed"

    try:
        names = sr.Microphone.list_microphone_names()
        if not names:
            return "No microphone device found"
        return f"Speech STT ON ({len(names)} mic device(s))"
    except Exception as exc:
        return f"Microphone unavailable: {exc}"


def recognize_speech_once_verbose(
    timeout: float = 4.0,
    phrase_time_limit: float = 6.0,
) -> Tuple[Optional[str], str]:
    """Capture one utterance and return (text, status_message)."""
    try:
        import speech_recognition as sr  # type: ignore
    except Exception:
        return None, "SpeechRecognition package missing"

    try:
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.7

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        text = recognizer.recognize_google(audio).strip()
        if not text:
            return None, "No words recognized"
        return text, "Speech recognized"
    except sr.WaitTimeoutError:
        return None, "No speech detected (timeout)"
    except sr.UnknownValueError:
        return None, "Audio heard but not understood"
    except sr.RequestError as exc:
        return None, f"Recognition service error: {exc}"
    except OSError as exc:
        return None, f"Microphone access error: {exc}"
    except Exception as exc:
        return None, f"Speech error: {exc}"
