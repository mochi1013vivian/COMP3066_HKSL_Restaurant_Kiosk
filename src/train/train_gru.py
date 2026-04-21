"""Train GRU model for closed-domain HKSL sequence recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from ..config.settings import DEFAULT_CLASS_NAMES_PATH, DEFAULT_MODEL_PATH, DEFAULT_SEQUENCE_CSV, DEFAULT_WINDOW_SIZE
    from ..data.dataset_sequence import SequenceDataset, load_sequence_dataframe
    from ..features.mediapipe_extractor import FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE
    from ..models.gru_classifier import GRUClassifier
    from ..utils.io_paths import ensure_dir
    from ..utils.seed import set_seed
except ImportError:  # pragma: no cover
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_ROOT = os.path.dirname(CURRENT_DIR)
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)

    from config.settings import DEFAULT_CLASS_NAMES_PATH, DEFAULT_MODEL_PATH, DEFAULT_SEQUENCE_CSV, DEFAULT_WINDOW_SIZE
    from data.dataset_sequence import SequenceDataset, load_sequence_dataframe
    from features.mediapipe_extractor import FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE
    from models.gru_classifier import GRUClassifier
    from utils.io_paths import ensure_dir
    from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU sequence model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_SEQUENCE_CSV)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--class-out", type=Path, default=DEFAULT_CLASS_NAMES_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        choices=[FEATURE_MODE_HANDS, FEATURE_MODE_HANDS_POSE],
        default=FEATURE_MODE_HANDS_POSE,
        help="Feature mode for preprocessing: hands baseline or hands+upper-body pose",
    )
    parser.add_argument(
        "--with-arms",
        action="store_true",
        help="Convenience switch for hands+upper-body pose training (same as --feature-mode hands_pose).",
    )
    return parser.parse_args()


def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = float((np.asarray(preds) == np.asarray(trues)).mean()) if trues else 0.0
    return avg_loss, acc, np.asarray(trues), np.asarray(preds)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    feature_mode = FEATURE_MODE_HANDS_POSE if args.with_arms else args.feature_mode

    bundle = load_sequence_dataframe(args.data, window_size=args.window_size, feature_mode=feature_mode)
    X, y, classes = bundle.X, bundle.y, bundle.classes

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(
        input_dim=X.shape[2],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(classes),
        dropout=args.dropout,
    ).to(device)

    counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * x.size(0)

        train_loss /= max(1, len(train_loader.dataset))
        val_loss, val_acc, y_true, y_pred = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    model.load_state_dict(best_state)
    _, val_acc, y_true, y_pred = evaluate_epoch(model, val_loader, criterion, device)
    print("\nValidation classification report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))

    checkpoint = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "window_size": args.window_size,
        "input_dim": int(X.shape[2]),
        "feature_mode": feature_mode,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seed": args.seed,
        "val_acc": float(val_acc),
    }

    ensure_dir(args.model_out.parent)
    torch.save(checkpoint, args.model_out)
    with open(args.class_out, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"Saved model checkpoint: {args.model_out}")
    print(f"Saved class names: {args.class_out}")


if __name__ == "__main__":
    main()
