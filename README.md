# HKSL Realtime Ordering Assistant (PyTorch)

COMP3066 prototype for **closed-domain HKSL fast-food ordering** using a webcam,
MediaPipe landmarks, and a PyTorch GRU sequence classifier.

## Current repo state

This repository currently has two important scopes:

1. **Active baseline (truthful current default)**
	- 11-token working model/data bundle
	- dataset: `data/raw/landmarks_sequences.csv`
	- model: `models/best_gru_hands_pose_full.pt`
	- classes: `models/class_names.json`
	- reports: `data/processed/eval_reports_active_baseline/`

2. **Target submission scope (not yet the active default)**
	- intended 14-token submission vocabulary in `data/raw/labels_submission_hands_pose.txt`
	- related partial artifacts exist, but they are not yet a coherent runnable default bundle

If you are taking over this repo, treat the **11-token active baseline** as the current reliable starting point.

**Design focus:**
- MediaPipe hand landmarks (Lab 8 style workflow)
- PyTorch GRU sequence model (Lab 7 style train/eval)
- **Realtime stabilization** with duplicate-word blocking
- **Presentation-mode optimization** for live demonstrations
- High-contrast UI emphasizing final sentence output

## Active baseline vocabulary (default)
- `i`
- `want`
- `one`
- `two`
- `hamburger`
- `fries`
- `hash_brown`
- `apple_pie`
- `and`
- `with`
- `thank_you`

## Target submission vocabulary (planned next step)

The intended 14-token submission set is:

- `i`
- `want`
- `hamburger`
- `fries`
- `hash_brown`
- `apple_pie`
- `and`
- `with`
- `thank_you`
- `one`
- `two`
- `three`
- `four`
- `five`

The repo is therefore currently in a transition state:
- **working baseline** = 11 tokens
- **desired submission target** = 14 tokens

### Clean recollection plan

If you want to rebuild the dataset from scratch, delete the active collection CSV and use the phase files below so old data does not mix with the new recording cycle:

- `data/raw/labels_phase1_core.txt` → `i`, `want`, `one`, `two`, `three`, `four`, `five`
- `data/raw/labels_phase2_foods.txt` → `hamburger`, `fries`, `hash_brown`, `apple_pie`
- `data/raw/labels_phase3_connectors_service.txt` → `and`, `with`, `thank_you`

Recommended reset targets for the fresh run:

- `data/raw/landmarks_sequences_submission_hands_pose.csv`
- `models/best_gru_submission_hands_pose.pt`
- `models/class_names_submission_hands_pose.json`
- `data/processed/eval_reports_submission_hands_pose/`

Use `thank_you` in the code/labels, even if you display it as “thank you” in the UI.

You can edit `src/config/labels.py` for controlled vocabulary changes.

---

## Project structure

- `src/data/collect_sequences.py` — sequence collector (C once → countdown → auto-record)
- `src/train/train_gru.py` — PyTorch GRU training
- `src/eval/evaluate_gru.py` — evaluation + confusion diagnostics
- `src/app/realtime_demo.py` — realtime demo app with stabilization
- `src/features/mediapipe_extractor.py` — shared landmark extraction
- `src/features/sequence_preprocess.py` — shared sequence preprocessing

Canonical paths:
- raw data: `data/raw/landmarks_sequences.csv`
- model checkpoint: `models/best_gru_hands_pose_full.pt`
- class names: `models/class_names.json`
- evaluation outputs: `data/processed/eval_reports_active_baseline/`

---

## Setup

```bash
cd /Users/vivianckk/Downloads/hksl_fastfood_vscode
conda activate base
pip install -r requirements.txt
```

If you prefer `.venv`, make sure `torch` is installed there first.

---

## 1) Check camera index

```bash
python src/list_cameras.py
```

---

## 2) Collect sequence data

Example for one label:

```bash
python src/data/collect_sequences.py --label i --samples 40 --camera-index 1
```

For the improved similarity-reduction workflow, collect with hands + upper-body context:

```bash
python src/data/collect_sequences.py --label fries --samples 60 --camera-index 1 --with-arms
```

### Phase-by-phase expansion example

Use the three clean batch files below for recollection:

```bash
python src/data/collect_sequences.py --label i --label-file data/raw/labels_phase1_core.txt --samples 50 --camera-index 1 --output data/raw/landmarks_sequences_submission_hands_pose.csv --feature-mode hands_pose --with-arms
python src/data/collect_sequences.py --label hamburger --label-file data/raw/labels_phase2_foods.txt --samples 50 --camera-index 1 --output data/raw/landmarks_sequences_submission_hands_pose.csv --feature-mode hands_pose --with-arms
python src/data/collect_sequences.py --label and --label-file data/raw/labels_phase3_connectors_service.txt --samples 50 --camera-index 1 --output data/raw/landmarks_sequences_submission_hands_pose.csv --feature-mode hands_pose --with-arms
```

Notes:
- Keep using the same explicit output CSV while you expand phase by phase.
- Add only one small phase at a time so retraining stays understandable.

### Collector behavior
- Press **C once** → starts **3-second countdown**
- Then recording runs **continuously automatically**
- It keeps collecting until `--samples` is reached (or you stop)
- Press **S** to stop session
- Press **Q** to quit

Repeat collection for each label.

---

## 3) Train model (GRU)

```bash
python src/train/train_gru.py --data data/raw/landmarks_sequences.csv --window-size 20 --epochs 40 --batch-size 32 --feature-mode hands_pose
```

Outputs:
- `models/best_gru_hands_pose_full.pt`
- `models/class_names.json`

---

## 4) Evaluate model

```bash
python src/eval/evaluate_gru.py --data data/raw/landmarks_sequences.csv --model models/best_gru_hands_pose_full.pt --report-dir data/processed/eval_reports_active_baseline --window-size 20 --feature-mode hands_pose
```

Outputs in `data/processed/eval_reports_active_baseline/`:
- `metrics.json`
- `class_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `top_confusions.txt`

### Pair-focused diagnostics

Use the helper script below after evaluation to inspect directional confusion and class-level metrics for an **in-scope** confusing pair:

```bash
python src/eval/pair_confusion_report.py \
	--report-dir data/processed/eval_reports_active_baseline \
	--label-a hamburger \
	--label-b hash_brown
```

If you ran a live protocol and observed confusion in realtime, pass it too:

```bash
python src/eval/pair_confusion_report.py \
	--report-dir data/processed/eval_reports_active_baseline \
	--label-a hamburger \
	--label-b hash_brown \
	--live-confusion-rate 0.24 \
	--recollection-rounds 1
```

Thresholds in this script are practical heuristics, not fixed truths. Adjust them with CLI flags based on your demo constraints.

---

## HKU-HKSLLEX reference integration (verification/enrichment)

If you have permission to use HKU-HKSLLEX for this course project, stage source files in:

- `data/external/hku_hksllex/`

Then normalize/import the reference data:

```bash
python src/data/import_hksllex.py --source-json data/external/hku_hksllex/data.json
```

or

```bash
python src/data/import_hksllex.py --source-csv data/external/hku_hksllex/data.csv
```

This produces:
- `data/processed/hksllex_reference.json`
- `data/processed/hksllex_reference.csv`
- `data/processed/hksllex_label_map.json`

Notes:
- Keep HKU-HKSLLEX as a trusted reference layer for vocabulary verification.
- Keep your own webcam-collected data as the primary training source.

---

## Explicit data-path discipline (important)

To avoid mixing datasets (e.g., `landmarks_sequences.csv` vs `landmarks_sequences_submission_hands_pose.csv`), always pass `--data` explicitly in both training and evaluation.

- Good: train and eval both point to the same explicit dataset path
- Avoid: relying on defaults when multiple sequence CSVs exist

For focused troubleshooting, use a real confusing pair from your current report output rather than an old archived example.

---

## 5) Run realtime demo

### Normal mode (stable, high-accuracy):
```bash
python src/app.py --camera-index 1 --sound
```

### Presentation mode (low-latency, optimized for live demos):
```bash
python src/app.py --camera-index 1 --sound --presentation-mode
```

### With text-to-speech (optional extra: reads final sentence on Enter):
```bash
python src/app.py --camera-index 1 --sound --tts
```

Optional TTS / microphone speech features are presentation add-ons, not part of the core ordering-recognition scope.

### Realtime controls
- **Confirm Order button**: freeze the current sentence and send it to the kiosk-style confirmation panel
- **Start New Order button**: clear the confirmed order and begin a new one
- **X**: reset the current order session
- **Z / Backspace**: undo last word before confirmation
- **C**: confirm current order
- **N**: start a new order
- **Enter**: speak the current or confirmed sentence (if `--tts` enabled)
- **Q / Esc**: quit

### Confirm-order flow

The realtime app now includes a simple restaurant-style order confirmation flow:

1. Build the order by signing as usual.
2. Press **Confirm Order** when the sentence is ready.
3. The app freezes the current order summary and shows a confirmation panel with:
	- `Order Confirmed`
	- `Your order has been sent`
	- a queue badge such as `A12`
	- `Now Preparing` / `Sent to Kitchen` style feedback
4. Press **Start New Order** to clear the confirmed order and begin another demo order.

The order number is **auto-generated** from a session counter (starting at `A12` for demo readability).

### Stability parameters (all modes)
- `--accept-confidence` (default `0.78`, presentation: `0.80`)
- `--stable-frames` (default `6`, presentation: `6`)
- `--accept-cooldown` (default `1.0`s, presentation: `0.35`s)
- `--repeat-block-seconds` (default `2.0`s, presentation: `1.8`s)
- `--no-sign-frames` (default `11` frames, ~367ms @ 30fps)

### Suggested precision pass for the new dataset

After recollecting, train with the following pattern:

```bash
python src/train/train_gru.py --data data/raw/landmarks_sequences_submission_hands_pose.csv --window-size 20 --epochs 40 --batch-size 32 --feature-mode hands_pose --with-arms
python src/eval/evaluate_gru.py --data data/raw/landmarks_sequences_submission_hands_pose.csv --window-size 20 --feature-mode hands_pose --with-arms
python src/app.py --camera-index 1 --sound --presentation-mode --feature-mode hands_pose
```

For live sensitivity tuning:

- raise `--accept-confidence` if false positives appear
- raise `--stable-frames` if words are accepted too early
- lower `--accept-cooldown` only after the dataset is stable
- keep `--no-sign-frames` high enough to block repeats

Increase confidence and stable-frames for fewer false accepts.

### Presentation mode features
- **Camera resolution**: 960×540 (low latency)
- **Hand detection**: 2 hands max (kept consistent with train/collect contract)
- **Model complexity**: 0 (lite, real-time)
- **Display**: low-latency rendering with frame skipping (`--skip-frames 1` in presentation mode)
- **Thresholds**: confidence 0.80, stable-frames 6, cooldown 0.35s, repeat-block 1.8s

---

## Use cases

### Primary: Deaf Customer Food Ordering
- Customer walks into restaurant and uses sign language
- App recognizes signs and builds food order
- Final sentence (e.g., "I want one hamburger and fries") is displayed/spoken
- Staff can read the order from the screen or hear TTS output
- **UI emphasis**: Large final sentence as main visual output

This repo currently focuses on the **customer ordering** scenario. Any broader communication features should be treated as extra experiments, not the core submission story.

---

## Recommended presentation mode thresholds

After testing on live demonstrations, these values optimize for real-time accuracy and user experience:

### ✅ Presentation Mode (Low-Latency, Live Demo)
- `--accept-confidence`: **0.80**
- `--stable-frames`: **6**
- `--accept-cooldown`: **0.35** seconds
- `--repeat-block-seconds`: **1.8** seconds
- `--no-sign-frames`: **11** frames → @ 30fps ≈ 367ms of no-sign required before same word can be added again
- `--min-detection-confidence`: **0.60**
- **Camera**: 960×540 resolution
- **MediaPipe model**: Complexity 0 (Lite), max_num_hands=2

**Rationale:**
- **0.80 confidence**: Keeps acceptance practical while reducing obvious false accepts
- **6-frame buffer**: Balances responsiveness and stability for live demo rhythm
- **11-frame no-sign gap** (~367ms): Prevents accidentally repeating words when user holds a sign too long
- **0.35s cooldown**: Allows fast, natural rhythm for multiple-word orders
- **1.8s repeat block**: Prevents same word appearing twice in quick succession

### Archived optional speech web demo

`speech_demo.html` is an archived standalone browser experiment for speech phrase matching.
It is optional and not part of the core HKSL ordering pipeline/runtime path.

### 🔍 Normal Mode (Higher Accuracy, Less Latency Emphasis)
- Uses defaults: confidence 0.78, stable-frames 6, cooldown 1.0s
- Better for non-real-time, accuracy-critical scenarios

---

## Accuracy notes

- If classes are mixed up, inspect `top_confusions.txt` and recollect those confusion pairs.
- Keep class sample counts balanced.
- Record from multiple sessions/lighting/angles.
- Use same signer style in collection and demo when possible.
- The duplicate-word blocking prevents accidental repetition when holding a sign

---

## Course alignment & design goals

- **Lab 8 alignment**: Webcam + MediaPipe landmark extraction + realtime loop
- **Lab 7 alignment**: Modular train/eval scripts, validation split, saved model checkpoint
- **Assistive tech focus**: Real-world application (restaurant ordering)
- **Stability design**: Duplicate-word blocking + multi-frame confirmation prevents user frustration

This is deliberately **not** a general sign-language translator.
It is a focused HKSL restaurant-ordering assistive prototype
with optimized thresholds for live demonstrations and low-latency real-world use.
