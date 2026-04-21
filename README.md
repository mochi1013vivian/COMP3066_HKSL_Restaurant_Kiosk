# HKSL Realtime Communication Assistant (PyTorch)

**Assistive technology supporting:**
1. **Deaf customers ordering food** at restaurants (primary demo use case)
2. **Deaf restaurant staff communicating** with coworkers (secondary impact scenario)

COMP3066 prototype using closed-domain HKSL for realistic restaurant environments.

**Design focus:**
- MediaPipe hand landmarks (Lab 8 style workflow)
- PyTorch GRU sequence model (Lab 7 style train/eval)
- **Realtime stabilization** with duplicate-word blocking
- **Presentation-mode optimization** for live demonstrations
- High-contrast UI emphasizing final sentence output

## Current vocabulary (default)
- `i`
- `want`
- `one`
- `two`
- `three`
- `four`
- `five`
- `hamburger`
- `apple_pie`
- `hash_brown`
- `fries`
- `cola`

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
- model checkpoint: `models/best_gru.pt`
- evaluation outputs: `data/processed/eval_reports/`

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

For the new Phase 4 glue words, collect them one at a time using the dedicated label file:

```bash
python src/data/collect_sequences.py --label and --label-file data/raw/labels_phase4.txt --samples 50 --camera-index 1 --output data/raw/landmarks_sequences_round1.csv
python src/data/collect_sequences.py --label with --label-file data/raw/labels_phase4.txt --samples 50 --camera-index 1 --output data/raw/landmarks_sequences_round1.csv
```

Notes:
- `apple_pie` is already in the current vocabulary.
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
python src/train/train_gru.py --data data/raw/landmarks_sequences_round1.csv --window-size 20 --epochs 30 --batch-size 32 --feature-mode hands_pose
```

Outputs:
- `models/best_gru.pt`
- `models/class_names.json`

---

## 4) Evaluate model

```bash
python src/eval/evaluate_gru.py --data data/raw/landmarks_sequences_round1.csv --window-size 20 --feature-mode hands_pose
```

Outputs in `data/processed/eval_reports/`:
- `metrics.json`
- `class_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `top_confusions.txt`

### Pair-focused diagnostics (example: `table` vs `fries`)

Use the helper script below after evaluation to inspect directional confusion and class-level metrics for a confusing pair:

```bash
python src/eval/pair_confusion_report.py \
	--report-dir data/processed/eval_reports \
	--label-a table \
	--label-b fries
```

If you ran a live protocol and observed confusion in realtime, pass it too:

```bash
python src/eval/pair_confusion_report.py \
	--report-dir data/processed/eval_reports \
	--label-a table \
	--label-b fries \
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

To avoid mixing datasets (e.g., `landmarks_sequences.csv` vs `landmarks_sequences_round1.csv`), always pass `--data` explicitly in both training and evaluation.

- Good: train and eval both point to the same explicit dataset path
- Avoid: relying on defaults when multiple sequence CSVs exist

For focused `table`/`fries` troubleshooting, use:
- `TABLE_FRIES_COLLECTION_EVAL_PROTOCOL.md` (collection + offline eval + live test checklist)

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

### With text-to-speech (reads final sentence on Enter):
```bash
python src/app.py --camera-index 1 --sound --tts
```

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
- `--repeat-block-seconds` (default `2.0`s, presentation: `2.2`s)
- `--no-sign-frames` (default `11` frames, ~367ms @ 30fps)

Increase confidence and stable-frames for fewer false accepts.

### Presentation mode features
- **Camera resolution**: 640×480 (low latency)
- **Hand detection**: 1 hand max (faster, cleaner)
- **Model complexity**: 0 (lite, real-time)
- **Display**: Smooth 30+ FPS with large sentence output
- **Thresholds**: Stricter confidence, faster acceptance

---

## Use cases

### Primary: Deaf Customer Food Ordering
- Customer walks into restaurant and uses sign language
- App recognizes signs and builds food order
- Final sentence (e.g., "i want hamburger and cola") is displayed/spoken
- Staff can read the order from the screen or hear TTS output
- **UI emphasis**: Large final sentence as main visual output

### Secondary: Staff Communication
- Deaf restaurant staff can communicate with coworkers using same app
- Builds short messages in real-time during service
- Coworkers see/hear the message on shared display
- Same stabilization and duplicate-blocking logic prevents mishearing

Both scenarios use the same model and logic — just different social contexts.

---

## Recommended presentation mode thresholds

After testing on live demonstrations, these values optimize for real-time accuracy and user experience:

### ✅ Presentation Mode (Low-Latency, Live Demo)
- `--accept-confidence`: **0.83** → Stricter; fewer false positives
- `--stable-frames`: **8** → 8 frames @ 30fps ≈ 267ms confirmation window
- `--accept-cooldown`: **0.4** seconds → Fast successive words possible
- `--repeat-block-seconds`: **2.2** seconds → Prevent rapid repetition
- `--no-sign-frames`: **11** frames → @ 30fps ≈ 367ms of no-sign required before same word can be added again
- `--min-detection-confidence`: **0.65** → Tighter hand detection; fewer spurious frames
- **Camera**: 640×480 resolution
- **MediaPipe model**: Complexity 0 (Lite), max_num_hands=1

**Rationale:**
- **0.83 confidence**: Prevents spurious words during hand movement between signs
- **8-frame buffer** (~267ms): Feels snappy while remaining stable
- **11-frame no-sign gap** (~367ms): Prevents accidentally repeating words when user holds a sign too long
- **0.4s cooldown**: Allows fast, natural rhythm for multiple-word orders
- **2.2s repeat block**: Prevents same word appearing twice in quick succession

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
- **Assistive tech focus**: Real-world application (restaurant ordering + staff communication)
- **Stability design**: Duplicate-word blocking + multi-frame confirmation prevents user frustration

This is deliberately **not** a general sign-language translator.
It is a focused HKSL restaurant-ordering and staff-communication assistive prototype
with optimized thresholds for live demonstrations and low-latency real-world use.
