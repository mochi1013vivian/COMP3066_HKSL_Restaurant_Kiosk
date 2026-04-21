"""PCA visualization for dynamic sign dataset.

This follows the classic classroom workflow:
1) Load tabular data with pandas
2) Apply PCA to 2D
3) Plot class clusters to inspect separability
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from .extract_sequences import sequence_feature_dim
except ImportError:  # pragma: no cover
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    from extract_sequences import sequence_feature_dim


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "raw" / "landmarks_dynamic.csv")
DEFAULT_FIG_PATH = str(PROJECT_ROOT / "data" / "processed" / "pca_dynamic_2d.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dynamic sign data with PCA.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--include-deltas", action="store_true", default=True)
    parser.add_argument("--save-path", default=DEFAULT_FIG_PATH)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dynamic dataset: {data_path}")

    df = pd.read_csv(data_path)
    expected_dim = sequence_feature_dim(args.window_size, args.include_deltas)
    feature_cols = [f"f{i}" for i in range(expected_dim)]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns (first missing: {missing[0]})")
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(str).str.strip()

    valid = (~X.isna().any(axis=1)) & y.ne("")
    X = X[valid]
    y = y[valid]

    if len(X) < 3:
        raise ValueError("Need at least 3 valid samples for PCA visualization.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print(
        "PCA explained variance ratio: "
        f"PC1={explained[0]:.4f}, PC2={explained[1]:.4f}, total={explained.sum():.4f}"
    )
    print("Class counts:")
    for label, count in y.value_counts().sort_index().items():
        print(f"  {label}: {count}")

    plt.figure(figsize=(9, 6))
    for label in sorted(y.unique()):
        idx = y == label
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=label,
            alpha=0.75,
            s=35,
        )

    plt.xlabel(f"Principal Component 1 ({explained[0] * 100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({explained[1] * 100:.1f}%)")
    plt.title("2D PCA Projection of Dynamic HKSL Dataset")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=args.dpi)
    print(f"Saved PCA figure: {save_path}")

    # Also show interactively for notebook / local exploration.
    plt.show()


if __name__ == "__main__":
    main()
