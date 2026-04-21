# HKSL Real-Time Sign-Language Food Ordering Assistant

COMP3066 group project prototype for deaf and hard-of-hearing restaurant interactions.

This project follows:
- **Lab 8 logic**: webcam → MediaPipe landmarks → feature preprocessing → classifier → realtime prediction
- **Lab 7 structure**: separate modules for collection, training, evaluation, and inference

## What the system does

The app recognizes hand signs from a webcam, converts them into tokens, and builds a simple order sentence for restaurant staff.

Current pipeline:
1. Webcam capture
2. Hand landmark extraction (MediaPipe Hands)
3. Feature preprocessing
4. Model training/loading
5. Realtime token prediction
6. Sentence building
7. On-screen UI display

## Project files

- `src/collect_data.py` — collect labeled landmark samples from webcam
- `src/extract_landmarks.py` — shared landmark extraction and preprocessing
- `src/train_model.py` — train and save the model
- `src/evaluate_model.py` — evaluate the trained model
- `src/collect_data_dynamic.py` — collect dynamic sequence landmark samples
- `src/train_model_dynamic.py` — train dynamic temporal model (RandomForest)
- `src/evaluate_model_dynamic.py` — evaluate dynamic temporal model
- `src/inference_realtime.py` — realtime webcam prediction loop
- `src/sentence_builder.py` — convert tokens into order text
- `src/ui_display.py` — draw the on-screen interface
- `src/list_cameras.py` — list available cameras on your Mac
- `src/app.py` — shortcut entrypoint for the realtime app

## Setup

1. Open this folder in VS Code.
2. Activate your Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Handoff quick guide (for daily use)

Use this section if you just want to run the project without reading the full document.

### 0) Open terminal at project root

```bash
cd /Users/vivianckk/Downloads/hksl_fastfood_vscode
source .venv/bin/activate
```

### 1) Check camera index

```bash
python src/list_cameras.py
```

- Usually MacBook camera = `1`
- Usually iPhone camera = `0`

### 1.5) If you want a clean restart

If you want to recode everything from scratch, move the old dataset and models out of the way first:

```bash
mv data/raw/landmarks.csv data/raw/landmarks_old.csv
mv models/sign_model.joblib models/sign_model_old.joblib
mv models/class_names.joblib models/class_names_old.joblib
mv src/data/raw/landmarks.csv src/data/raw/landmarks_old.csv
mv src/models/sign_model.joblib src/models/sign_model_old.joblib
mv src/models/class_names.joblib src/models/class_names_old.joblib
```

Then collect the new vocabulary below.

### 2) Run app directly (no retraining)

```bash
python src/app.py --camera-index 1
```

### 3) Add new training data (when needed)

Static sign example:

```bash
python src/collect_data.py --label hamburger --samples 80 --camera-index 1
```

Dynamic sign example:

```bash
python src/collect_data_dynamic.py --label thank_you --samples 40 --camera-index 1
```

### 4) Retrain models

```bash
python src/train_model.py
python src/train_model_dynamic.py
```

### 5) Evaluate models

```bash
python src/evaluate_model.py --show-confusion
python src/evaluate_model_dynamic.py --show-confusion
```

### 6) Run app after retraining

```bash
python src/app.py --camera-index 1
```

---

### One-command checklist (copy/paste block)

```bash
cd /Users/vivianckk/Downloads/hksl_fastfood_vscode
source .venv/bin/activate
python src/list_cameras.py
python src/evaluate_model.py --show-confusion
python src/evaluate_model_dynamic.py --show-confusion
python src/app.py --camera-index 1
```

### Important: where to run commands

All commands in this README assume your terminal is at the **project root**:

```bash
/Users/vivianckk/Downloads/hksl_fastfood_vscode
```

If your terminal is inside `src/`, remove the `src/` prefix from commands.

Examples:

```bash
# From project root (recommended)
python src/train_model.py

# From inside src/
python train_model.py
```

### Common issue: “I retrained but app still predicts old labels”

This happens when collection/training is run from inside `src/` and files are created in:
- `src/data/raw/landmarks.csv`
- `src/models/sign_model.joblib`

But the app may still load from project-root:
- `data/raw/landmarks.csv`
- `models/sign_model.joblib`

The scripts are now fixed to default to project-root paths. If you already have data/models in `src/`, recover with:

```bash
# Run from project root
cp src/data/raw/landmarks.csv data/raw/landmarks.csv
cp src/models/sign_model.joblib models/sign_model.joblib
cp src/models/class_names.joblib models/class_names.joblib
```

Then evaluate/run:

```bash
python src/evaluate_model.py --show-confusion
python src/app.py --camera-index 1
```

## Step-by-step guide for adding a new label

If you want to add a new sign such as `hamburger`, follow these steps in order.

### Step 1: add the label to the vocabulary

Edit `src/labels.py` and add the new token to `ITEM_LABELS` or the correct label group.

Example:
- add `hamburger`, `apple_pie`, `hash_brown` to `ITEM_LABELS`
- keep `i` and `want` as separate word tokens for more precise signing

If you want the label to display nicely on screen, also add it to `TOKEN_DISPLAY_MAP`.

### Step 2: collect training samples for that label

Use the webcam to collect many examples of the new sign:

```bash
python src/collect_data.py --label hamburger --samples 80 --camera-index 1
```

For movement-based signs (dynamic), collect short sequences:

```bash
python src/collect_data_dynamic.py --label thank_you --samples 40 --camera-index 1
```

While collecting:
- press `C` to capture one sample
- press `Q` to quit

Try to collect samples from:
- different angles
- different lighting
- different signers

### Step 3: collect more samples for existing labels if needed

If some labels are underrepresented, top them up too:

```bash
python src/collect_data.py --label fries --samples 50 --camera-index 1
python src/collect_data.py --label cola --samples 50 --camera-index 1
python src/collect_data.py --label i --samples 50 --camera-index 1
python src/collect_data.py --label want --samples 50 --camera-index 1
```

Balanced data usually gives more reliable results than a single large class.

### Step 4: retrain the model

After adding new data, retrain the classifier:

```bash
python src/train_model.py
```

Then train the dynamic temporal model:

```bash
python src/train_model_dynamic.py
```

If you are already inside `src/`, use:

```bash
python train_model.py
```

Optional:

```bash
python src/train_model.py --n-estimators 400 --test-size 0.2
```

### Step 5: evaluate the result

Check performance on a holdout split:

```bash
python src/evaluate_model.py --show-confusion
```

Evaluate dynamic model too:

```bash
python src/evaluate_model_dynamic.py --show-confusion
```

If you are already inside `src/`, use:

```bash
python evaluate_model.py --show-confusion
```

### Step 6: run the realtime app

Start the assistant after training:

```bash
python src/app.py --camera-index 1
```

`app.py` now runs in dynamic-first mode (movement-based recognition when dynamic model is available).

Or run the module directly:

```bash
python src/inference_realtime.py --camera-index 1
```

Optional direct command with explicit dynamic model path:

```bash
python src/inference_realtime.py --camera-index 1 --use-dynamic --model-dynamic models/sign_model_dynamic.joblib
```

## Camera selection on Mac

On a Mac with both a built-in MacBook camera and an iPhone camera, you may need to choose the correct index.

First, list available cameras:

```bash
python src/list_cameras.py
```

### Use iPhone camera

If your iPhone shows up as camera index `0`, use:

```bash
python src/collect_data.py --label hamburger --samples 80 --camera-index 1
python src/app.py --camera-index 1
```

### Use MacBook camera

If your MacBook camera is camera index `1`, use:

```bash
python src/collect_data.py --label hamburger --samples 80 --camera-index 1
python src/app.py --camera-index 1
```

If the camera index is different on another machine, use the number shown by `src/list_cameras.py`.

## Realtime controls

In the live app:
- `A` or `Space` — add the current stable token to the order sentence
- Keep signs as separate words when possible: `I`, `want`, `apple pie`, `hash brown`, `hamburger`
- `U` — undo the last token
- `C` — clear the order sentence
- `T` — toggle auto-add mode
- `Q` or `ESC` — quit

## Dynamic-first notes

- Primary runtime path is movement-aware (dynamic model) when `models/sign_model_dynamic.joblib` exists.
- If dynamic model is missing, app prints a warning and uses static fallback.
- To force dynamic-only behavior (disable static fallback):

```bash
python src/inference_realtime.py --camera-index 1 --use-dynamic --disable-static-fallback
```

## Recommended workflow

When you want to improve the project, use this loop:

1. Add label to `src/labels.py`
2. Collect more samples with `src/collect_data.py`
3. Retrain with `src/train_model.py`
4. Evaluate with `src/evaluate_model.py`
5. Test in realtime with `src/app.py`

## Design notes

- Runtime is now **dynamic-first** when `models/sign_model_dynamic.joblib` is available.
- Static model is retained as fallback/backward compatibility.
- Dynamic model currently uses landmark sequence features + temporal RandomForest.
- Training and inference share the same landmark preprocessing base in `src/extract_landmarks.py`.
