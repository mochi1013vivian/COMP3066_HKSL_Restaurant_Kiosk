"""Safe dataset sanity-check utility (non-destructive).

Usage:
	python check.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
	target = Path("data/raw/landmarks_sequences_submission_hands_pose.csv")
	if not target.exists():
		raise FileNotFoundError(f"Dataset not found: {target}")

	df = pd.read_csv(target)
	if "label" not in df.columns:
		raise ValueError("Missing required 'label' column in dataset")

	print(f"Dataset: {target}")
	print(f"Rows: {len(df)}")
	print("Label distribution:")
	print(df["label"].value_counts())


if __name__ == "__main__":
	main()