"""Closed-domain order sentence builder for restaurant demo with duplicate blocking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

QUANTITY_TOKENS = {"one", "two", "three", "four", "five"}
ITEM_TOKENS = {"hamburger", "fries", "apple_pie", "hash_brown"}
CONNECTOR_TOKENS = {"and", "with"}
POLITENESS_TOKENS = {"thank_you"}

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
        """Build readable sentence from tokens with semi-strict grammar cleanup.

        This keeps the app forgiving while preventing obviously broken output.
        Supported patterns include:
        - I want + quantity + item
        - I want + quantity + item + and/with + quantity? + item
        - Thank you
        """
        if not self.tokens:
            return "(empty)"

        order_tokens = [t for t in self.tokens if t not in POLITENESS_TOKENS]
        has_thank_you = any(t in POLITENESS_TOKENS for t in self.tokens)

        order_text = self._build_order_text(order_tokens)

        if order_text and has_thank_you:
            return f"{order_text} Thank you."
        if order_text:
            return order_text
        if has_thank_you:
            return "Thank you."
        return "(empty)"

    def _build_order_text(self, tokens: List[str]) -> str:
        if not tokens:
            return ""

        has_intent = "want" in tokens or ("i" in tokens and any(t in ITEM_TOKENS for t in tokens))
        segments: List[str] = []
        connectors: List[str] = []

        pending_qty: Optional[str] = None
        pending_item: Optional[str] = None
        pending_connector: Optional[str] = None

        for token in tokens:
            if token == "i":
                continue
            if token == "want":
                has_intent = True
                continue

            if token in QUANTITY_TOKENS:
                if pending_item is not None:
                    segments.append(self._format_segment(pending_qty, pending_item))
                    connectors.append(pending_connector or "and")
                    pending_item = None
                    pending_connector = None
                pending_qty = token
                continue

            if token in ITEM_TOKENS:
                if pending_item is None:
                    pending_item = token
                else:
                    segments.append(self._format_segment(pending_qty, pending_item))
                    connectors.append(pending_connector or "and")
                    pending_qty = None
                    pending_item = token
                    pending_connector = None
                continue

            if token in CONNECTOR_TOKENS:
                if pending_item is not None:
                    if segments:
                        connectors.append(pending_connector or "and")
                    segments.append(self._format_segment(pending_qty, pending_item))
                    pending_qty = None
                    pending_item = None
                    pending_connector = token
                continue

        if pending_item is not None:
            if segments and pending_connector is not None:
                connectors.append(pending_connector)
            segments.append(self._format_segment(pending_qty, pending_item))

        if not segments:
            return ""

        body = segments[0]
        for connector, segment in zip(connectors, segments[1:]):
            body += f" {token_to_display(connector)} {segment}"

        sentence = f"I want {body}" if has_intent else body
        sentence = " ".join(sentence.split()).strip()
        if not sentence:
            return ""
        return sentence[0].upper() + sentence[1:] + "."

    def _format_segment(self, quantity: Optional[str], item: str) -> str:
        parts: List[str] = []
        if quantity:
            parts.append(token_to_display(quantity))
        parts.append(token_to_display(item))
        return " ".join(parts)
