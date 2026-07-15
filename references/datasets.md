# Dataset Inventory and Selection Notes

This project uses public secondary datasets only. Datasets are analysed through
separate experimental tracks unless a scientifically valid shared identifier and
matching unit of analysis exist.

## Scientific Handling Rules

- A URL row is not automatically a human participant
- An email row is not automatically an individual user
- A webpage row is not automatically a behavioural observation
- Public phishing datasets may support technical detection without supporting
  direct human susceptibility claims

## Intended Experimental Tracks

### Track A: Phishing and malicious URL datasets

Purpose:

- technical phishing detection
- URL feature engineering
- technical exposure analysis
- scalability and class-imbalance experiments

### Track B: Webpage phishing datasets

Purpose:

- HTML and domain feature analysis
- website legitimacy classification
- contextual exposure scoring

### Track C: Email phishing datasets

Purpose:

- text-based phishing classification
- persuasion and urgency proxy extraction
- trust manipulation and contextual cues

### Track D: Human-awareness or cybersecurity-behaviour datasets

Purpose:

- awareness and literacy proxies
- warning recognition
- security-practice indicators
- better support for the human-centred HWI

## Metadata File

The inventory system reads optional manual metadata from:

- `references/dataset_metadata.json`

Use that file to document:

- `unit_of_analysis`
- `licence`
- `notes`

Example structure:

```json
{
  "datasets": {
    "phishing_urls/example.csv": {
      "unit_of_analysis": "Each row represents a URL observation, not a person.",
      "licence": "Add the source licence here.",
      "notes": "Suitable for technical phishing detection only unless linked to behavioural data."
    }
  }
}
```

Keys may be:

- the dataset path relative to `data/raw/`
- the filename
- the filename stem

Prefer the relative path because it is the least ambiguous.

## Current Status

- No raw datasets are present in the repository at the time of this update
- No datasets have been merged
- No target variables have been validated yet
- No HWI proxy mapping has started yet
