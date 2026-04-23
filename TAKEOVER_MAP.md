# Takeover Map

## What is the active working baseline?

Use this bundle as the current truthful baseline:

- Dataset: `data/raw/landmarks_sequences.csv`
- Model: `models/best_gru_hands_pose_full.pt`
- Class names: `models/class_names.json`
- Eval reports: `data/processed/eval_reports_active_baseline/`
- Feature mode: `hands_pose`
- Vocabulary size: **11 tokens**

Current 11 baseline tokens:
- `and`
- `apple_pie`
- `fries`
- `hamburger`
- `hash_brown`
- `i`
- `one`
- `thank_you`
- `two`
- `want`
- `with`

## What is the target submission scope?

Desired next-step submission vocabulary is **14 tokens** in:
- `data/raw/labels_submission_hands_pose.txt`

The missing target tokens compared with the current 11-token working baseline are:
- `three`
- `four`
- `five`

## Important warning

Do **not** treat these as the active default bundle yet:
- `data/raw/landmarks_sequences_submission_hands_pose.csv`
- `models/best_gru_submission_hands_pose.pt`
- `models/class_names_submission_hands_pose.json`

Why:
- submission CSV currently contains only **3 labels** (`apple_pie`, `fries`, `hamburger`)
- submission checkpoint currently contains **7 classes** (`five`, `four`, `i`, `one`, `three`, `two`, `want`)
- these artifacts do **not** form one coherent runnable bundle

## Current state in one sentence

The repo is currently between:
- a **working 11-token active baseline**, and
- a **14-token intended submission target** that has not yet been rebuilt into one coherent model/data/report bundle.

## What to do next

1. Keep the 11-token bundle as the default working baseline.
2. Align docs/proposal text to this truthful current state.
3. Recollect and retrain the missing target tokens (`three`, `four`, `five`) into a coherent 14-token bundle.
4. Only after that, switch defaults to the submission bundle.
