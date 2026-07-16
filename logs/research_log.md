# Research Log

## 2026-07-15

### Repository foundation audit

- Confirmed that the repository scaffold exists under `C:\DBS\Sem 2\Aplied Research\HWI`
- Confirmed that `data/raw/` is currently empty
- Confirmed that `notebooks/01_Data_Loading.ipynb` existed as valid but empty JSON
- Confirmed that `requirements-core.txt` and the planned future notebooks were missing
- Confirmed that `src/data_loader.py` and `src/utils.py` were empty placeholders

### Foundation implementation

- Strengthened `src/config.py` with typed constants and safe generated-directory creation
- Implemented safe dataset discovery, loading, summarisation, and inventory reporting
- Added manual metadata support for unit-of-analysis and licence annotations
- Rebuilt `notebooks/01_Data_Loading.ipynb` for dataset discovery and inventory generation
- Added foundational pytest coverage for config and data-loader behaviours

### Scientific guardrails retained

- No raw data were edited or overwritten
- No model training was started
- No exploratory analysis beyond dataset inventory was started
- No user-level HWI claims were inferred from technical records

## 2026-07-16

### Raw-data integration milestone

- Confirmed the four approved raw categories: `awareness`, `ai_emails`, `emails`, and `phishing_urls`
- Confirmed that raw datasets have now been downloaded into those four categories
- Confirmed that deleted `webpage` and `breaches` raw categories are absent
- Froze the current raw-data scope to those four separate tracks
- Started raw-data inventory integration against the downloaded files

### Scope preserved

- No EDA was performed
- No preprocessing was performed
- No HWI calculation was performed
- No model training was performed
- Scientific separation of units of analysis was preserved across awareness, email, AI-email, and URL tracks
