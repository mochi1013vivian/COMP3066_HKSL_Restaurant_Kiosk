# HKU-HKSLLEX External Source Staging

This folder stores externally sourced HKU-HKSLLEX files before they are normalized
into project-specific reference files.

## Source
- Repository: https://github.com/ldlhku/HKU-HKSLLEX
- Browser: https://ldlhku.github.io/HKU-HKSLLEX/sign_browser_github.html

## Intended use in this project
- Vocabulary verification
- Label/reference enrichment
- Variant notes for manual collection planning

Training should continue to rely primarily on this project's own collected webcam data.

## Permission + attribution
Use of HKU-HKSLLEX data in this course project is based on user-confirmed permission.
Keep attribution to Language Development Lab, HKU in reports/presentations.

## Suggested files to place here
- `data.json` (from HKU-HKSLLEX)
- `data.csv` (from HKU-HKSLLEX)
- optional metadata notes and permission evidence

## Next step
Run:

- `python src/data/import_hksllex.py --source-json data/external/hku_hksllex/data.json`

or

- `python src/data/import_hksllex.py --source-csv data/external/hku_hksllex/data.csv`

Outputs are written into `data/processed/`.
