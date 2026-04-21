"""Closed-domain order sentence builder for restaurant demo with duplicate blocking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from ..config.labels import token_to_display
except ImportError:  # pragma: no cover
    from config.labels import token_to_display


@dataclass
class OrderSentenceBuilder:
    """Builds customer order sentences with duplicate word blocking.
    
    Prevents accidental repetition when user holds a sign.
    Tracks last accepted token for consecutive duplicate detection.
    """
    tokens: List[str] = field(default_factory=list)
    last_token: Optional[str] = field(default=None)
    last_token_time: float = field(default_factory=time.time)

    def add_token(self, token: str) -> None:
        """Add token to sentence only if not a consecutive duplicate."""
        token = token.strip()
        if token:
            # Don't add same word twice in a row
            if token != self.last_token:
                self.tokens.append(token)
                self.last_token = token
                self.last_token_time = time.time()

    def undo(self) -> None:
        """Remove the last token from sentence."""
        if self.tokens:
            self.tokens.pop()
            self.last_token = self.tokens[-1] if self.tokens else None

    def clear(self) -> None:
        """Clear all tokens from sentence."""
        self.tokens.clear()
        self.last_token = None
        self.last_token_time = time.time()

    def build_text(self) -> str:
        """Build readable sentence from tokens."""
        if not self.tokens:
            return "(empty)"
        words = []
        for token in self.tokens:
            display = token_to_display(token)
            if not words or display != words[-1]:
                words.append(display)
        sent = " ".join(words)
        return sent[0].upper() + sent[1:] if sent else "(empty)"
