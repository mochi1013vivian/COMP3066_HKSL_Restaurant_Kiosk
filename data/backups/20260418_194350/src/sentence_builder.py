"""Utilities to convert recognized tokens into order-friendly text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from .labels import ITEM_LABELS, INTENT_LABELS, QUANTITY_LABELS, SERVICE_LABELS, token_to_display
except ImportError:  # pragma: no cover - supports `python src/*.py`
    from labels import ITEM_LABELS, INTENT_LABELS, QUANTITY_LABELS, SERVICE_LABELS, token_to_display


INTENT_SET = set(INTENT_LABELS)
QUANTITY_SET = set(QUANTITY_LABELS)
ITEM_SET = set(ITEM_LABELS)
SERVICE_SET = set(SERVICE_LABELS)


@dataclass
class OrderSentenceBuilder:
    """Build an on-screen order sentence from recognized sign tokens."""

    tokens: List[str] = field(default_factory=list)
    custom_display_map: Dict[str, str] = field(default_factory=dict)

    def clear(self) -> None:
        self.tokens.clear()

    def undo(self) -> None:
        if self.tokens:
            self.tokens.pop()

    def add_token(self, token: str) -> None:
        token = token.strip()
        if not token:
            return

        # Support command-like control tokens if user adds them in future datasets.
        if token in {"clear", "reset"}:
            self.clear()
            return
        if token in {"undo", "backspace"}:
            self.undo()
            return

        self.tokens.append(token)

    def _display_token(self, token: str) -> str:
        if token in self.custom_display_map:
            return self.custom_display_map[token]
        return token_to_display(token)

    def to_display_tokens(self) -> List[str]:
        return [self._display_token(t) for t in self.tokens]

    def build_sentence(self) -> str:
        words = self.to_display_tokens()
        if not words:
            return ""

        sentence = " ".join(words).strip()
        return sentence[0].upper() + sentence[1:]

    def build_structured_order(self) -> Dict[str, Optional[str] | List[str]]:
        """Parse token sequence into a lightweight order structure.

        This does not replace full NLP, but provides useful slots for restaurant UI:
        - intent
        - quantity
        - items
        - service_options
        - free_tokens
        """
        token_set = set(self.tokens)

        intent = None
        intent_components = set()
        if {"i", "want"}.issubset(token_set):
            intent = "i_want"
            intent_components = {"i", "want"}
        else:
            intent = next((t for t in self.tokens if t in INTENT_SET), None)
            if intent is not None:
                intent_components = {intent}

        quantity = next((t for t in self.tokens if t in QUANTITY_SET), None)

        items = [t for t in self.tokens if t in ITEM_SET]
        service_options = [t for t in self.tokens if t in SERVICE_SET]

        known = intent_components | set([x for x in [quantity] if x is not None]) | set(items) | set(service_options)
        free_tokens = [t for t in self.tokens if t not in known]

        return {
            "intent": intent,
            "quantity": quantity,
            "items": items,
            "service_options": service_options,
            "free_tokens": free_tokens,
        }

    def build_staff_text(self) -> str:
        """Generate display text optimized for staff reading."""
        order = self.build_structured_order()

        intent = order["intent"]
        quantity = order["quantity"]
        items = order["items"]
        service_options = order["service_options"]

        if intent and items:
            intent_text = self._display_token(intent)
            item_text = " and ".join(self._display_token(item) for item in items)

            parts = [intent_text]
            if quantity:
                parts.append(self._display_token(quantity))
            parts.append(item_text)
            if service_options:
                parts.append("(" + ", ".join(self._display_token(s) for s in service_options) + ")")

            sentence = " ".join(parts)
            return sentence[0].upper() + sentence[1:]

        # Fallback to direct token rendering when structure is incomplete.
        return self.build_sentence()
