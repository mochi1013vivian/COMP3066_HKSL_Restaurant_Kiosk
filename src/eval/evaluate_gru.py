"""Evaluate trained GRU model and export confusion diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

try:
    from ..config.settings import DEFAULT_MODEL_PATH, DEFAULT_SEQUENCE_CSV, DEFAULT_WINDOW_SIZE, REPORT_DIR
    from ..data.dataset_sequence import SequenceDataset, load_sequence_dataframe
    from ..features.mediapipe_extractor import FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE
    from ..models.gru_classifier import GRUClassifier
    from ..utils.io_paths import ensure_dir
    from ..utils.metrics import top_confusion_pairs
except ImportError:  # pragma: no cover
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_ROOT = os.path.dirname(CURRENT_DIR)
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)

    from config.settings import DEFAULT_MODEL_PATH, DEFAULT_SEQUENCE_CSV, DEFAULT_WINDOW_SIZE, REPORT_DIR
    from data.dataset_sequence import SequenceDataset, load_sequence_dataframe
    from features.mediapipe_extractor import FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE
    from models.gru_classifier import GRUClassifier
    from utils.io_paths import ensure_dir
    from utils.metrics import top_confusion_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GRU model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_SEQUENCE_CSV)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        choices=["auto", FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE],
        default="auto",
        help="Feature mode for preprocessing. 'auto' reads from model checkpoint (fallback hands_pose).",
    )
    parser.add_argument(
        "--with-arms",
        action="store_true",
        help="Convenience switch for hands+upper-body pose evaluation (same as --feature-mode hands_pose).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    if args.feature_mode == "auto":
        feature_mode = ckpt.get("feature_mode", FEATURE_MODE_HANDS_POSE)
    else:
        feature_mode = args.feature_mode

    if args.with_arms:
        feature_mode = FEATURE_MODE_HANDS_POSE

    bundle = load_sequence_dataframe(args.data, window_size=args.window_size, feature_mode=feature_mode)
    X, y, classes = bundle.X, bundle.y, bundle.classes

    _, X_eval, _, y_eval = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    eval_ds = SequenceDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False)

    model = GRUClassifier(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        num_classes=len(ckpt["classes"]),
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            logits = model(xb)
            preds.extend(torch.argmax(logits, dim=1).numpy().tolist())

    y_pred = np.asarray(preds, dtype=np.int64)
    acc = accuracy_score(y_eval, y_pred)
    report_text = classification_report(y_eval, y_pred, target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(y_eval, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_eval, y_pred, labels=list(range(len(classes))))

    ensure_dir(args.report_dir)
    (args.report_dir / "metrics.json").write_text(
        json.dumps({"accuracy": float(acc), "samples": int(len(y_eval))}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(report_dict).transpose().to_csv(args.report_dir / "class_report.csv", index=True)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(args.report_dir / "confusion_matrix.csv")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=35, ha="right")
    plt.yticks(ticks, classes)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(args.report_dir / "confusion_matrix.png", dpi=160)

    top_pairs = top_confusion_pairs(y_eval, y_pred, classes, top_k=8)
    (args.report_dir / "top_confusions.txt").write_text(
        "\n".join([f"{a} -> {b}: {n}" for a, b, n in top_pairs]) or "No confusions.",
        encoding="utf-8",
    )

    print(f"Eval accuracy: {acc:.4f} ({len(y_eval)} samples) | feature_mode={feature_mode}")
    print(report_text)
    print("Top confusion pairs:")
    if top_pairs:
        for a, b, n in top_pairs:
            print(f"  {a} -> {b}: {n}")
    else:
        print("  None")


if __name__ == "__main__":
    main()
