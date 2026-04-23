"""Pair-focused confusion diagnostics with practical (heuristic) decision guidance.

This script reads:
- class_report.csv
- confusion_matrix.csv

from an evaluation report directory, then prints metrics focused on one confusing label pair
(e.g., hamburger vs hash_brown).

Important:
- Thresholds are heuristics by design and are configurable via CLI.
- Use this as decision support, not absolute ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair-focused confusion report with heuristic recommendations.")
    parser.add_argument("--report-dir", type=Path, required=True, help="Directory containing class_report.csv and confusion_matrix.csv")
    parser.add_argument("--label-a", default="hamburger", help="First label in the confusing pair")
    parser.add_argument("--label-b", default="hash_brown", help="Second label in the confusing pair")
    parser.add_argument(
        "--live-confusion-rate",
        type=float,
        default=None,
        help="Optional live confusion rate (0..1) from manual protocol, e.g., 0.24",
    )
    parser.add_argument(
        "--recollection-rounds",
        type=int,
        default=1,
        help="How many recollection rounds already completed (used in recommendation logic)",
    )

    # Heuristic thresholds (configurable)
    parser.add_argument("--good-confusion-rate", type=float, default=0.08)
    parser.add_argument("--mid-confusion-rate", type=float, default=0.15)
    parser.add_argument("--high-confusion-rate", type=float, default=0.25)
    parser.add_argument("--good-f1", type=float, default=0.90)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--live-high-confusion", type=float, default=0.20)
    return parser.parse_args()


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _row_support(cm: pd.DataFrame, label: str) -> int:
    return int(cm.loc[label, :].sum())


def _get_class_metric(class_report: pd.DataFrame, label: str, metric: str) -> float:
    return float(class_report.loc[label, metric])


def _validate_inputs(report_dir: Path, label_a: str, label_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_report_path = report_dir / "class_report.csv"
    confusion_path = report_dir / "confusion_matrix.csv"

    if not class_report_path.exists():
        raise FileNotFoundError(f"Missing class report: {class_report_path}")
    if not confusion_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {confusion_path}")

    class_report = pd.read_csv(class_report_path, index_col=0)
    confusion = pd.read_csv(confusion_path, index_col=0)

    for label in (label_a, label_b):
        if label not in class_report.index:
            raise ValueError(f"Label '{label}' not found in class_report.csv")
        if label not in confusion.index or label not in confusion.columns:
            raise ValueError(f"Label '{label}' not found in confusion_matrix.csv")

    return class_report, confusion


def main() -> None:
    args = parse_args()
    class_report, cm = _validate_inputs(args.report_dir, args.label_a, args.label_b)

    a, b = args.label_a, args.label_b

    a_to_b = int(cm.loc[a, b])
    b_to_a = int(cm.loc[b, a])
    support_a = _row_support(cm, a)
    support_b = _row_support(cm, b)

    rate_a_to_b = _safe_rate(a_to_b, support_a)
    rate_b_to_a = _safe_rate(b_to_a, support_b)
    pair_total = support_a + support_b
    pair_confused = a_to_b + b_to_a
    pair_confusion_rate = _safe_rate(pair_confused, pair_total)

    p_a = _get_class_metric(class_report, a, "precision")
    r_a = _get_class_metric(class_report, a, "recall")
    f1_a = _get_class_metric(class_report, a, "f1-score")
    p_b = _get_class_metric(class_report, b, "precision")
    r_b = _get_class_metric(class_report, b, "recall")
    f1_b = _get_class_metric(class_report, b, "f1-score")

    print("== Pair confusion report ==")
    print(f"Pair: {a} vs {b}")
    print(f"Report directory: {args.report_dir}")
    print("\n(Thresholds below are practical heuristics, not fixed truths.)")
    print(
        "Heuristics: "
        f"good_conf<= {args.good_confusion_rate:.2f}, "
        f"mid_conf<= {args.mid_confusion_rate:.2f}, "
        f"high_conf> {args.high_confusion_rate:.2f}, "
        f"good_f1>= {args.good_f1:.2f}, "
        f"min_support>= {args.min_support}"
    )

    print("\n-- Offline metrics --")
    print(f"{a}: support={support_a}, precision={p_a:.4f}, recall={r_a:.4f}, f1={f1_a:.4f}")
    print(f"{b}: support={support_b}, precision={p_b:.4f}, recall={r_b:.4f}, f1={f1_b:.4f}")
    print(f"{a} -> {b}: {a_to_b}/{support_a} = {rate_a_to_b:.4f}")
    print(f"{b} -> {a}: {b_to_a}/{support_b} = {rate_b_to_a:.4f}")
    print(f"pair_confusion_rate: {pair_confused}/{pair_total} = {pair_confusion_rate:.4f}")

    low_support = (support_a < args.min_support) or (support_b < args.min_support)
    offline_good = (
        rate_a_to_b <= args.good_confusion_rate
        and rate_b_to_a <= args.good_confusion_rate
        and f1_a >= args.good_f1
        and f1_b >= args.good_f1
    )
    offline_mid = (
        rate_a_to_b <= args.mid_confusion_rate
        and rate_b_to_a <= args.mid_confusion_rate
        and min(f1_a, f1_b) >= (args.good_f1 - 0.05)
    )

    print("\n-- Recommendation --")
    if offline_good:
        print("Offline status: GOOD for this pair under current heuristic thresholds.")
        if low_support:
            print(
                "Caution: support is still small for at least one label. "
                "Run the live pair test protocol before concluding collection is sufficient."
            )

        if args.live_confusion_rate is not None:
            print(f"Live confusion rate provided: {args.live_confusion_rate:.4f}")
            if args.live_confusion_rate > args.live_high_confusion:
                print(
                    "Live confusion remains high despite good offline metrics. "
                    "Prioritize upper-body/location feature improvement (pose/arm-relative signals) "
                    "instead of endless recollection."
                )
            else:
                print("Live confusion is acceptable under current heuristic threshold.")
    elif offline_mid:
        print(
            "Offline status: MODERATE. Do one more focused recollection round for this pair, "
            "then re-evaluate before making architecture changes."
        )
    else:
        print("Offline status: HIGH CONFUSION for this pair.")
        if args.recollection_rounds >= 2:
            print(
                "After >=2 recollection rounds, move to feature improvement (upper-body/location cues)."
            )
            print(
                "For live demo stability, consider temporarily dropping one of the pair labels until improved."
            )
        else:
            print("Run another focused recollection round, then reassess with this report.")


if __name__ == "__main__":
    main()
