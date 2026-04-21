"""Import and normalize HKU-HKSLLEX reference data for this project.

This script does NOT train a model. It creates normalized reference artifacts that
help with vocabulary verification and label mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize HKU-HKSLLEX source data.")
    parser.add_argument("--source-json", type=Path, default=None, help="Path to HKU-HKSLLEX JSON data")
    parser.add_argument("--source-csv", type=Path, default=None, help="Path to HKU-HKSLLEX CSV data")
    parser.add_argument("--out-json", type=Path, default=Path("data/processed/hksllex_reference.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/processed/hksllex_reference.csv"))
    parser.add_argument("--map-out", type=Path, default=Path("data/processed/hksllex_label_map.json"))
    parser.add_argument(
        "--project-label-file",
        type=Path,
        default=None,
        help="Optional label file (one token per line). If omitted, use src/config/labels.py DEFAULT_LABELS.",
    )
    return parser.parse_args()


def _canonical_token(text: str) -> str:
    token = text.strip().lower()
    token = token.replace("&", " and ")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _records_from_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        # Common patterns for datasets
        for key in ("data", "items", "records", "entries", "signs"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # Fallback: treat dict values as records if many are dict-like
        dict_values = [x for x in payload.values() if isinstance(x, dict)]
        if dict_values:
            return dict_values

    raise ValueError(f"Unsupported JSON structure in {path}")


def _records_from_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _pick(record: Dict[str, Any], keys: Iterable[str]) -> str:
    lower_index = {k.lower(): k for k in record.keys()}
    for wanted in keys:
        real = lower_index.get(wanted.lower())
        if real is not None:
            value = record.get(real)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return ""


def _normalize_record(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    # HKSLLEX JSON schema uses nested structures (e.g. HKU_HKSLLEX_ID + translation blocks)
    sign_id = ""
    hku_id_obj = record.get("HKU_HKSLLEX_ID")
    if isinstance(hku_id_obj, dict):
        sign_id = str(hku_id_obj.get("id", "")).strip()

    if not sign_id:
        sign_id = _pick(record, ["id", "hksllex_id", "entry_id", "identifier"])

    english = ""
    zh_hk = ""
    cantonese = ""
    disamb_en = ""
    disamb_zh = ""

    tr_obj = record.get("translation")
    if isinstance(tr_obj, dict):
        en_obj = tr_obj.get("EN-GB") or tr_obj.get("en-GB") or tr_obj.get("EN")
        if isinstance(en_obj, dict):
            english = str(en_obj.get("primary", "")).strip()
            disamb_en = str(en_obj.get("disambiguation", "")).strip()

        zh_obj = tr_obj.get("ZH-HK") or tr_obj.get("zh-HK")
        if isinstance(zh_obj, dict):
            zh_hk = str(zh_obj.get("primary", "")).strip()
            disamb_zh = str(zh_obj.get("disambiguation", "")).strip()

        yue_obj = tr_obj.get("ZH-Yue") or tr_obj.get("zh-YUE") or tr_obj.get("zh-Yue")
        if isinstance(yue_obj, dict):
            cantonese = str(yue_obj.get("primary", "")).strip()

    if not english:
        english = _pick(record, ["english", "english_gloss", "gloss", "word", "lemma"])
    if not zh_hk:
        zh_hk = _pick(record, ["written chinese (hong kong)", "written_chinese_hk", "chinese", "zh_hk"])
    if not cantonese:
        cantonese = _pick(record, ["cantonese", "jyutping", "spoken_cantonese"])
    if not disamb_en:
        disamb_en = _pick(record, ["disambiguation (english)", "disambiguation_english", "disambiguation"])
    if not disamb_zh:
        disamb_zh = _pick(record, ["disambiguation (written chinese)", "disambiguation_chinese"])

    media = _pick(record, ["videoFile", "image", "video", "video_path", "webp", "media"])

    label_guess = _canonical_token(english) if english else ""

    if not sign_id:
        sign_id = f"row_{idx:06d}"

    return {
        "hksllex_id": sign_id,
        "english": english,
        "project_label_guess": label_guess,
        "written_chinese_hk": zh_hk,
        "cantonese": cantonese,
        "disambiguation_english": disamb_en,
        "disambiguation_chinese": disamb_zh,
        "media_ref": media,
        "source": "HKU-HKSLLEX",
    }


def _load_project_labels(label_file: Path | None) -> List[str]:
    if label_file is not None:
        labels: List[str] = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token and not token.startswith("#"):
                labels.append(token)
        return sorted(set(labels))

    current_dir = Path(__file__).resolve().parent
    src_root = current_dir.parent
    project_root = src_root.parent

    for p in (project_root, src_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    for import_path in ("src.config.labels", "config.labels"):
        try:
            mod = __import__(import_path, fromlist=["DEFAULT_LABELS"])
            labels = getattr(mod, "DEFAULT_LABELS", [])
            return sorted(set(str(x).strip() for x in labels if str(x).strip()))
        except Exception:
            continue

    return []


def _build_label_map(records: List[Dict[str, Any]], project_labels: List[str]) -> Dict[str, Dict[str, Any]]:
    by_guess: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        guess = str(r.get("project_label_guess", "")).strip()
        if not guess:
            continue
        by_guess.setdefault(guess, []).append(r)

    mapping: Dict[str, Dict[str, Any]] = {}
    for label in project_labels:
        candidates = by_guess.get(label, [])
        if candidates:
            c = candidates[0]
            mapping[label] = {
                "project_label": label,
                "hksllex_id": c.get("hksllex_id", ""),
                "english": c.get("english", ""),
                "written_chinese_hk": c.get("written_chinese_hk", ""),
                "cantonese": c.get("cantonese", ""),
                "verified": "review",
                "variant_notes": c.get("disambiguation_english", ""),
                "media_permitted": "permission-confirmed-by-user",
                "source": "HKU-HKSLLEX",
            }
        else:
            mapping[label] = {
                "project_label": label,
                "hksllex_id": "",
                "english": "",
                "written_chinese_hk": "",
                "cantonese": "",
                "verified": "not_found",
                "variant_notes": "",
                "media_permitted": "permission-confirmed-by-user",
                "source": "HKU-HKSLLEX",
            }

    return mapping


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.source_json and not args.source_csv:
        raise SystemExit("Provide at least one source: --source-json or --source-csv")

    raw_records: List[Dict[str, Any]] = []
    if args.source_json:
        if not args.source_json.exists():
            raise FileNotFoundError(args.source_json)
        raw_records.extend(_records_from_json(args.source_json))

    if args.source_csv:
        if not args.source_csv.exists():
            raise FileNotFoundError(args.source_csv)
        raw_records.extend(_records_from_csv(args.source_csv))

    normalized = [_normalize_record(r, i) for i, r in enumerate(raw_records, start=1)]

    _ensure_parent(args.out_json)
    args.out_json.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(args.out_csv, normalized)

    project_labels = _load_project_labels(args.project_label_file)
    label_map = _build_label_map(normalized, project_labels)

    _ensure_parent(args.map_out)
    args.map_out.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Imported records: {len(raw_records)}")
    print(f"Normalized records: {len(normalized)}")
    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_csv}")
    print(f"Project labels mapped: {len(label_map)}")
    print(f"Wrote: {args.map_out}")


if __name__ == "__main__":
    main()
