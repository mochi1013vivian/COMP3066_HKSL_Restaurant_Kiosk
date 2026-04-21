"""Dataset loader for landmark sequence CSV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from ..features.sequence_preprocess import normalize_sequence, unflatten_sequence
except ImportError:  # pragma: no cover
    from features.sequence_preprocess import normalize_sequence, unflatten_sequence


@dataclass
class SequenceDataBundle:
    X: np.ndarray
    y: np.ndarray
    classes: List[str]


def load_sequence_dataframe(path: Path, window_size: int, feature_mode: str = "hands") -> SequenceDataBundle:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column")

    feature_cols = [c for c in df.columns if c.startswith("f")]
    X_rows: List[np.ndarray] = []
    labels: List[str] = []

    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        if not label:
            continue
        feat = pd.to_numeric(row[feature_cols], errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(feat).any():
            continue

        seq = unflatten_sequence(feat, window_size, feature_mode=feature_mode)
        seq = normalize_sequence(seq, feature_mode=feature_mode)
        X_rows.append(seq)
        labels.append(label)

    if not X_rows:
        raise ValueError("No valid sequence samples found.")

    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([class_to_idx[x] for x in labels], dtype=np.int64)
    X = np.asarray(X_rows, dtype=np.float32)
    return SequenceDataBundle(X=X, y=y, classes=classes)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
