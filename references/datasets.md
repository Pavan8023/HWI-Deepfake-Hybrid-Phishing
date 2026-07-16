# Dataset Inventory and Selection Notes

This project uses public secondary datasets only. The current raw-data scope is
frozen to four approved categories, and those categories must remain separate
experimental tracks.

## Approved Raw Dataset Categories

Exactly these four raw-data folders are active:

- `data/raw/awareness/`
- `data/raw/ai_emails/`
- `data/raw/emails/`
- `data/raw/phishing_urls/`

The following deleted categories are not active and must not be restored in this
task:

- `data/raw/webpage/`
- `data/raw/breaches/`

## Scientific Handling Rules

- Rows in `awareness` require unit-of-analysis verification before any human-centred claim.
- Rows in `ai_emails` represent messages or generated messages, not people.
- Rows in `emails` represent email records, not people.
- Rows in `phishing_urls` represent URL records, not people.
- Datasets must not be merged row-by-row across categories.
- HWI design decisions remain pending until the awareness dataset columns and outcomes are inspected more deeply.
- URL and email tracks may support technical or contextual analyses later, but they are not direct human-participant datasets.
- Metadata fields marked as requiring verification must remain documented honestly until source documentation is confirmed.

## Research Roles By Category

### awareness

- Main human-centred HWI dataset track
- May contain awareness, decision, behaviour, trust, literacy, or security-practice proxies
- Final HWI dimensions and weights are not decided in this task

### ai_emails

- Human-generated versus LLM-generated email comparison track
- Suitable later for comparing manipulation, urgency, authority, and persuasion cues
- Each row should be treated as a message record unless documentation proves otherwise

### emails

- Large phishing-versus-legitimate email analysis track
- Suitable later for text classification and contextual-risk analysis
- Each row should be treated as an email record unless documentation proves otherwise

### phishing_urls

- Large-scale technical threat classification track
- Suitable later for technical-threat or technical-exposure analysis
- This track must not be presented as person-level HWI evidence

## Actual Supported Dataset Files Found

The following supported raw tabular files were present during the July 16, 2026 inspection:

- `awareness/phishing_awareness_dataset.csv`
- `ai_emails/human-generated/legit.csv`
- `ai_emails/human-generated/phishing.csv`
- `ai_emails/llm-generated/legit.csv`
- `ai_emails/llm-generated/phishing.csv`
- `emails/CEAS_08.csv`
- `emails/Enron.csv`
- `emails/Ling.csv`
- `emails/Nazario.csv`
- `emails/Nigerian_Fraud.csv`
- `emails/phishing_email.csv`
- `emails/SpamAssasin.csv`
- `phishing_urls/malicious_phish.csv`

## Unsupported or Archived Raw Files

- No unsupported raw files were found during the same inspection
- No ZIP archives were found during the same inspection

## Structural Verification Notes

- `ai_emails/llm-generated/phishing.csv` triggered a strict CSV parsing error because of inconsistent field counts
- That file remains in scope, but its structure requires manual verification before relying on full-row inventory statistics
- This is a loading-validation issue only; it does not justify editing the raw file

## Metadata File

Manual metadata is stored in:

- `references/dataset_metadata.json`

Use relative paths under `data/raw/` as keys whenever possible.

Fields that still require manual confirmation should remain:

- `source_platform`
- `source_reference`
- `unit_of_analysis`
- `licence`
- `target_column`
- `label_meaning`

## Current Status

- The project now has downloaded raw data in the four approved categories
- The categories remain separate tracks
- Raw-data inventory integration is active
- No EDA, preprocessing, feature engineering, HWI scoring, or model training has started
