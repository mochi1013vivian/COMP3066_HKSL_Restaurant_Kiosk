# Runbook: 11-Token Baseline → 14-Token Submission Bundle

> Date: 2026-04-23  
> Goal: Add `three`, `four`, `five` to the submission bundle and produce a coherent, evaluated 14-token model.

---

## 1. Current State

| Artifact | Path | Status |
|---|---|---|
| Landmark CSV (active) | `data/raw/landmarks_sequences.csv` | ✅ 11 tokens, coherent |
| Model (active) | `models/best_gru_hands_pose_full.pt` | ✅ 11 tokens, coherent |
| Class names (active) | `models/class_names.json` | ✅ 11 tokens |
| Eval report (active) | `data/processed/eval_reports_active_baseline/` | ✅ reference |
| Landmark CSV (submission) | `data/raw/landmarks_sequences_submission_hands_pose.csv` | ⚠️ incomplete — missing three/four/five |
| Model (submission) | `models/best_gru_submission_hands_pose.pt` | ❌ not coherent |
| Class names (submission) | `models/class_names_submission_hands_pose.json` | ❌ not coherent |

Label file for submission vocabulary: `data/raw/labels_submission_hands_pose.txt`  
(14 tokens: `i`, `want`, `hamburger`, `fries`, `hash_brown`, `apple_pie`, `and`, `with`, `thank_you`, `one`, `two`, `three`, `four`, `five`)

---

## 2. Tomorrow Morning Checklist

- [ ] Activate the virtual environment
- [ ] Verify webcam index (default `--camera-index 1`)
- [ ] Record 40 samples each for `three`, `four`, `five` into the submission CSV
- [ ] Verify row counts in the submission CSV after collection
- [ ] Train the rebuilt submission model
- [ ] Evaluate and confirm results
- [ ] Run the live demo with the new model

---

## 3. Recollection Commands (`three` / `four` / `five`)

Activate environment first:

```bash
source .venv/bin/activate
```

Collect `three`:

```bash
python -m src.data.collect_sequences \
  --label three \
  --label-file data/raw/labels_submission_hands_pose.txt \
  --output data/raw/landmarks_sequences_submission_hands_pose.csv \
  --feature-mode hands_pose \
  --samples 40 \
  --camera-index 1
```

Collect `four`:

```bash
python -m src.data.collect_sequences \
  --label four \
  --label-file data/raw/labels_submission_hands_pose.txt \
  --output data/raw/landmarks_sequences_submission_hands_pose.csv \
  --feature-mode hands_pose \
  --samples 40 \
  --camera-index 1
```

Collect `five`:

```bash
python -m src.data.collect_sequences \
  --label five \
  --label-file data/raw/labels_submission_hands_pose.txt \
  --output data/raw/landmarks_sequences_submission_hands_pose.csv \
  --feature-mode hands_pose \
  --samples 40 \
  --camera-index 1
```

Verify row counts after collection:

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/landmarks_sequences_submission_hands_pose.csv'); print(df['label'].value_counts().sort_index()); print('Total rows:', len(df))"
```

---

## 4. Training Command (Rebuilt Submission Bundle)

```bash
python -m src.train.train_gru \
  --data data/raw/landmarks_sequences_submission_hands_pose.csv \
  --model-out models/best_gru_submission_hands_pose.pt \
  --class-out models/class_names_submission_hands_pose.json \
  --feature-mode hands_pose \
  --epochs 40 \
  --hidden-dim 128 \
  --num-layers 2 \
  --dropout 0.25 \
  --patience 10 \
  --seed 42
```

---

## 5. Evaluation Command (Rebuilt Submission Bundle)

```bash
python -m src.eval.evaluate_gru \
  --data data/raw/landmarks_sequences_submission_hands_pose.csv \
  --model models/best_gru_submission_hands_pose.pt \
  --report-dir data/processed/eval_reports_submission_hands_pose \
  --feature-mode hands_pose \
  --window-size 20
```

Results will be written to `data/processed/eval_reports_submission_hands_pose/`.

---

## 6. Demo Run Command (Rebuilt Submission Bundle)

```bash
python -m src.app.realtime_demo \
  --model models/best_gru_submission_hands_pose.pt \
  --feature-mode auto \
  --camera-index 1 \
  --sound
```

Add `--presentation-mode` for the live demo path.

---

## 7. Verification Checklist

- [ ] `data/raw/landmarks_sequences_submission_hands_pose.csv` has exactly 14 unique labels
- [ ] Training completes without shape/dimension errors
- [ ] `models/class_names_submission_hands_pose.json` contains all 14 tokens including `three`, `four`, `five`
- [ ] `data/processed/eval_reports_submission_hands_pose/metrics.json` is generated successfully
- [ ] Live demo correctly recognises `three`, `four`, `five`
- [ ] No intended submission label is missing from the evaluation outputs
