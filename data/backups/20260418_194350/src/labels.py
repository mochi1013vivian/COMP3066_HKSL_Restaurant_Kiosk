"""Shared sign label definitions for the ordering assistant."""

from __future__ import annotations

from typing import Dict, List

# High-level intent tokens (word-level control words).
INTENT_LABELS = ["i", "want", "can_i_have", "may_i_have"]

# Quantity tokens (can be expanded to include digits/letters later).
QUANTITY_LABELS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]

# Restaurant-specific menu tokens.
ITEM_LABELS = [
    "hamburger",
    "apple_pie",
    "hash_brown",
    "fries",
    "rice",
    "chicken",
    "nuggets",
    "pizza",
    "cola",
    "water",
    "orange_juice",
    "chocolate_milkshake",
    "strawberry_sundae",
    "corn",
    "sprite",
]

# Service / context tokens.
SERVICE_LABELS = ["takeaway", "dine_in", "bill", "thank_you", "please"]

# Consolidated default set used by collection scripts.
DEFAULT_LABELS = INTENT_LABELS + QUANTITY_LABELS + ITEM_LABELS + SERVICE_LABELS

# Backward-compatible alias used by older scripts.
LABELS = DEFAULT_LABELS

# Tokens that are often dynamic in real sign language usage.
POTENTIALLY_DYNAMIC_LABELS = ["bill", "thank_you", "please"]

# Display mapping for UI sentence construction.
TOKEN_DISPLAY_MAP: Dict[str, str] = {
    "i": "I",
    "want": "want",
    "i_want": "I want",
    "can_i_have": "Can I have",
    "may_i_have": "May I have",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "dine_in": "dine in",
    "takeaway": "takeaway",
    "apple_pie": "apple pie",
    "hash_brown": "hash brown",
    "hamburger": "hamburger",
    "orange_juice": "orange juice",
    "strawberry_sundae": "strawberry sundae",
    "chocolate_milkshake": "chocolate milkshake",
    "thank_you": "thank you",
}


def token_to_display(token: str) -> str:
    """Convert internal token name to human-readable text."""
    return TOKEN_DISPLAY_MAP.get(token, token.replace("_", " "))


def load_label_set(path: str | None = None) -> List[str]:
    """Load labels from a text file (one token per line) or use defaults."""
    if path is None:
        return list(DEFAULT_LABELS)

    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token and not token.startswith("#"):
                labels.append(token)

    if not labels:
        raise ValueError(f"Label file is empty: {path}")

    return labels
