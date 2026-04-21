"""Evaluation metric helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


def top_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], top_k: int = 5) -> List[Tuple[str, str, int]]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    pairs: List[Tuple[str, str, int]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append((labels[i], labels[j], count))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
