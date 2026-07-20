# Human Weakness Index (HWI)

This repository contains the implementation artefact for the MSc dissertation:
"Deepfake-Enabled Hybrid Phishing Detection Through a Human Weakness Index".

## Current Phase

The current phase is raw-data integration and inventory validation.

This phase covers:

- validating the four approved raw-data categories
- discovering downloaded datasets safely
- generating raw-data inventory reports
- documenting metadata that still requires verification
- keeping dataset categories as separate research tracks

This phase does not include:

- EDA
- preprocessing
- feature engineering
- HWI design or scoring
- model training
- SHAP or LIME
- dashboard development

## Approved Raw Dataset Categories

The active raw-data folders are:

- `data/raw/awareness/`
- `data/raw/ai_emails/`
- `data/raw/emails/`
- `data/raw/phishing_urls/`

The old `webpage` and `breaches` raw categories were removed and are not part
of the active scope.

## Research Guardrails

- Use secondary public datasets only
- Treat HWI as a proxy-based vulnerability estimation framework
- Do not treat URL, email, or generated-message rows as human participants
- Keep the four dataset categories as separate experimental tracks
- Do not overwrite files in `data/raw/`
- Do not claim future work is already completed

## Repository Layout

```text
HWI/
|-- data/
|   |-- raw/
|   |   |-- awareness/
|   |   |-- ai_emails/
|   |   |-- emails/
|   |   `-- phishing_urls/
|   |-- interim/
|   `-- processed/
|-- notebooks/
|   `-- 01_Data_Loading.ipynb
|-- src/
|-- outputs/
|   `-- reports/
|-- references/
|-- logs/
|-- tests/
|-- dashboard/
|-- requirements-core.txt
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Core Modules

- `src/config.py`: central configuration, approved raw categories, path constants, and reproducibility settings
- `src/utils.py`: project-root detection, logging, JSON writing, and small helpers
- `src/data_loader.py`: category-aware discovery, safe loading, metadata resolution, manifest generation, and dataset inventory reporting

## Raw Data Status

- Raw data is no longer empty
- The approved categories currently contain 13 supported CSV files
- Inventory generation reports parse/load failures per file without modifying the raw source data

## Generated Inventory Outputs

Running the inventory workflow creates:

- `outputs/reports/dataset_inventory.csv`
- `outputs/reports/dataset_inventory.json`
- `outputs/reports/data_quality_summary.csv`
- `outputs/reports/raw_file_manifest.csv`
- `outputs/reports/unsupported_raw_files.csv`
- `outputs/reports/dataset_summaries/*.json`

## Setup

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
```

## Validation Commands

```powershell
python -m pytest -v
python -c "from src.config import APPROVED_RAW_DATASET_CATEGORIES; print(APPROVED_RAW_DATASET_CATEGORIES)"
python -c "from src.data_loader import build_dataset_inventory; print('data_loader import successful')"
```

## Notebook

`notebooks/01_Data_Loading.ipynb` is the loading and inventory notebook for the
current phase. It validates the environment, checks the four approved raw
categories, previews discovered files safely, and regenerates inventory reports.

## Current Status Summary

- Foundation code exists and remains active
- Downloaded raw datasets are present in the four approved categories
- Raw-data inventory integration is in progress
- EDA and modelling have not started
- Dataset tracks remain separate



# Awareness Dataset: Initial EDA Conclusion

- The dataset contains 5,000 rows and 12 columns.
- The complete dataset was analysed because its size is below the 50,000-row EDA limit.
- No loading errors occurred.
- `clicked_link` is the leading candidate outcome variable.
- `hover_time_ms` and `session_duration_sec` are behavioural proxy variables.
- `reported_email` may represent security awareness, but its timing must be confirmed to avoid target leakage.
- `user_id` should be excluded from modelling.
- `geo_location` should be excluded initially because it contains precise coordinates and lacks a clear theoretical link to human weakness.
- The dataset measures interaction behaviour and context more directly than digital literacy, trust or psychological awareness.
- The final HWI dimensions must therefore be based only on variables genuinely available and scientifically defensible.