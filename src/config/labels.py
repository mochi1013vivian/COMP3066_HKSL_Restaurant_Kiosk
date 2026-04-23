"""Closed-domain HKSL restaurant-ordering vocabulary definitions.

Focused on the coursework ordering prototype rather than broader communication tasks.
All vocabulary verified against HKSLLEX (HKU) and HKSL Browser (CUHK).
"""

from __future__ import annotations

from typing import Dict, List

# PHASE 1 (Safest - already implemented or ready)
PHASE_1_STABLE = ["i", "want", "one", "two", "three", "four", "five"]

# PHASE 1 additions selected for immediate collection
PHASE_1_ADDITIONS = ["hamburger", "fries", "apple_pie", "hash_brown", "thank_you"]

# PHASE 2 additions (next batch - medium priority)
PHASE_2_ADDITIONS = []

# PHASE 3 additions (currently unused)
PHASE_3_ADDITIONS = []

# PHASE 4 additions (ordering glue words)
PHASE_4_ADDITIONS = ["and", "with"]

# Food/Drink items (already or to be added)
FOOD_ITEMS = [
    "hamburger",
    "apple_pie",
    "hash_brown",
    "fries",
]

# Action verbs (to be added progressively)
ACTION_VERBS = ["want"]

# Connectors / glue words (useful for multi-item orders)
CONNECTOR_WORDS = ["and", "with"]

# Service words (politeness, confirmation)
SERVICE_WORDS = ["thank_you"]

# Active, truthful runtime baseline (current reliable default = 11 tokens)
ACTIVE_BASELINE_LABELS = [
    "i",
    "want",
    "one",
    "two",
    "hamburger",
    "fries",
    "apple_pie",
    "hash_brown",
    "and",
    "with",
    "thank_you",
]

# Intended submission target (planned 14-token set; not default runtime behavior)
TARGET_SUBMISSION_LABELS = [
    "i",
    "want",
    "one",
    "two",
    "three",
    "four",
    "five",
    "hamburger",
    "fries",
    "apple_pie",
    "hash_brown",
    "and",
    "with",
    "thank_you",
]

# Backward-compatible alias used by existing scripts/tools.
# Keep this mapped to the active baseline to preserve current safe behavior.
DEFAULT_LABELS = list(ACTIVE_BASELINE_LABELS)

# EXPANDED set is explicitly the target submission scope.
EXPANDED_LABELS = list(TARGET_SUBMISSION_LABELS)

TOKEN_DISPLAY_MAP: Dict[str, str] = {
    # Phase 0 (original)
    "i": "I",
    "want": "want",
    "one": "one",
    "two": "two",
    "three": "three",
    "four": "four",
    "five": "five",
    "hamburger": "hamburger",
    "apple_pie": "apple pie",
    "hash_brown": "hash brown",
    "fries": "fries",
    "thank_you": "thank you",
    "and": "and",
    "with": "with",
}

EMOJI_MAP: Dict[str, str] = {
    "i": "👤",
    "want": "👉",
    "one": "1️⃣",
    "two": "2️⃣",
    "three": "3️⃣",
    "four": "4️⃣",
    "five": "5️⃣",
    "hamburger": "🍔",
    "fries": "🍟",
    "apple_pie": "🥧",
    "hash_brown": "🟨",
    "and": "➕",
    "with": "🔗",
    "thank_you": "🙏",
}

NEUTRAL_EMOJI = "◯"


def load_label_set(path: str | None = None) -> List[str]:
    """Load label set from file or use default."""
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


def get_vocabulary_info() -> Dict[str, list]:
    """Return structured vocabulary metadata."""
    return {
        "current_default": DEFAULT_LABELS,
        "active_baseline": ACTIVE_BASELINE_LABELS,
        "target_submission": TARGET_SUBMISSION_LABELS,
        "current_phase_0": DEFAULT_LABELS,
        "phase_1_additions": PHASE_1_ADDITIONS,
        "phase_2_additions": PHASE_2_ADDITIONS,
        "phase_3_additions": PHASE_3_ADDITIONS,
        "phase_4_additions": PHASE_4_ADDITIONS,
        "food_items": FOOD_ITEMS,
        "action_verbs": ACTION_VERBS,
        "connector_words": CONNECTOR_WORDS,
        "service_words": SERVICE_WORDS,
    }


def token_to_display(token: str) -> str:
    return TOKEN_DISPLAY_MAP.get(token, token.replace("_", " "))


def token_to_emoji(token: str) -> str:
    return EMOJI_MAP.get(token, NEUTRAL_EMOJI)
