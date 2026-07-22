from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "final_results"
)

AWARENESS_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "model_training"
    / "awareness_model_metrics.json"
)

AWARENESS_EVALUATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "evaluation"
    / "awareness_final_evaluation.json"
)

AI_EMAIL_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "ai_email_model"
    / "ai_email_model_metrics.json"
)

AI_EMAIL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "ai_email_model"
    / "ai_email_model_comparison.csv"
)

AI_EMAIL_DUPLICATE_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "ai_email_model"
    / "ai_email_duplicate_audit.csv"
)

URL_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "url_risk_model"
    / "url_model_metrics.json"
)

URL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "url_risk_model"
    / "url_model_comparison.csv"
)

URL_DATASET_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "url_risk_model"
    / "url_dataset_summary.json"
)

HWI_PROFILES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "hwi_framework"
    / "hwi_evidence_profiles.csv"
)

HWI_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "hwi_framework"
    / "hwi_profile_selection_audit.csv"
)

PERFORMANCE_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "model_performance_summary.csv"
)

DATASET_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "dataset_and_model_summary.csv"
)

HWI_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "hwi_framework_summary.csv"
)

JSON_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "final_results_summary.json"
)

MARKDOWN_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "dissertation_results_summary.md"
)


def require_file(path: Path) -> None:
    """Confirm that an expected result file exists."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}"
        )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON result file."""

    require_file(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    if not isinstance(loaded, dict):
        raise TypeError(
            f"Expected a JSON object in {path}"
        )

    return loaded


def select_model_row(
    comparison: pd.DataFrame,
    model_name: str,
) -> pd.Series:
    """Select one model from a comparison table."""

    matches = comparison.loc[
        comparison["model_name"] == model_name
    ]

    if matches.empty:
        raise KeyError(
            f"Model '{model_name}' was not found."
        )

    return matches.iloc[0]


def build_performance_summary() -> pd.DataFrame:
    """Build a table of the selected model from each track."""

    awareness_metrics = read_json(
        AWARENESS_METRICS_PATH
    )

    awareness_evaluation = read_json(
        AWARENESS_EVALUATION_PATH
    )

    ai_metrics = read_json(
        AI_EMAIL_METRICS_PATH
    )

    ai_comparison = pd.read_csv(
        AI_EMAIL_COMPARISON_PATH
    )

    url_metrics = read_json(
        URL_METRICS_PATH
    )

    url_comparison = pd.read_csv(
        URL_COMPARISON_PATH
    )

    awareness_model = str(
        awareness_metrics["best_hwi_model"]
    )

    ai_model = str(
        ai_metrics["selected_model"]
    )

    url_model = str(
        url_metrics["selected_model"]
    )

    awareness_model_metrics = (
        awareness_metrics["models"][
            awareness_model
        ]["metrics"]
    )

    ai_row = select_model_row(
        ai_comparison,
        ai_model,
    )

    url_row = select_model_row(
        url_comparison,
        url_model,
    )

    records = [
        {
            "track": "Behavioural awareness",
            "selected_model": awareness_model,
            "target": "clicked_link",
            "accuracy": awareness_model_metrics[
                "accuracy"
            ],
            "precision": awareness_model_metrics[
                "precision"
            ],
            "recall": awareness_model_metrics[
                "recall"
            ],
            "f1": awareness_model_metrics[
                "f1"
            ],
            "roc_auc": awareness_evaluation[
                "roc_auc"
            ],
            "pr_auc": awareness_evaluation[
                "pr_auc"
            ],
            "brier_score": awareness_evaluation[
                "brier_score"
            ],
            "supported_interpretation": (
                "Exploratory behavioural susceptibility "
                "estimate; predictive evidence unsupported."
            ),
        },
        {
            "track": "AI email",
            "selected_model": ai_model,
            "target": (
                "phishing versus legitimate email"
            ),
            "accuracy": float(
                ai_row["accuracy"]
            ),
            "precision": float(
                ai_row["precision"]
            ),
            "recall": float(
                ai_row["recall"]
            ),
            "f1": float(
                ai_row["f1"]
            ),
            "roc_auc": float(
                ai_row["roc_auc"]
            ),
            "pr_auc": float(
                ai_row["pr_auc"]
            ),
            "brier_score": float(
                ai_row["brier_score"]
            ),
            "supported_interpretation": (
                "Email-level phishing and persuasion-risk "
                "classification; not human susceptibility."
            ),
        },
        {
            "track": "URL technical risk",
            "selected_model": url_model,
            "target": "benign versus malicious URL",
            "accuracy": float(
                url_row["accuracy"]
            ),
            "precision": float(
                url_row["precision"]
            ),
            "recall": float(
                url_row["recall"]
            ),
            "f1": float(
                url_row["f1"]
            ),
            "roc_auc": float(
                url_row["roc_auc"]
            ),
            "pr_auc": float(
                url_row["pr_auc"]
            ),
            "brier_score": float(
                url_row["brier_score"]
            ),
            "supported_interpretation": (
                "URL-level technical maliciousness; "
                "not human susceptibility."
            ),
        },
    ]

    return pd.DataFrame(records)


def build_dataset_summary() -> pd.DataFrame:
    """Build a concise dataset and methodology table."""

    duplicate_audit = pd.read_csv(
        AI_EMAIL_DUPLICATE_AUDIT_PATH
    )

    url_summary = read_json(
        URL_DATASET_SUMMARY_PATH
    )

    exact_duplicates = int(
        duplicate_audit[
            "exact_duplicate_rows"
        ].sum()
    )

    template_duplicates = int(
        duplicate_audit[
            "template_duplicate_rows"
        ].sum()
    )

    records = [
        {
            "track": "Behavioural awareness",
            "modelling_rows": 5000,
            "predictor_count": 46,
            "split_method": (
                "Stratified 80/20 split"
            ),
            "special_controls": (
                "Training-only preprocessing; target and "
                "possible post-outcome variables excluded."
            ),
        },
        {
            "track": "AI email",
            "modelling_rows": 4000,
            "predictor_count": 30,
            "split_method": (
                "Group-aware train/test split using "
                "template_group_id"
            ),
            "special_controls": (
                f"{exact_duplicates} exact duplicate rows and "
                f"{template_duplicates} template duplicate rows "
                "identified; shared train/test templates = 0; "
                "feature-leakage audit found no confirmed leakage."
            ),
        },
        {
            "track": "URL technical risk",
            "modelling_rows": int(
                url_summary["model_rows"]
            ),
            "predictor_count": int(
                url_summary["feature_count"]
            ),
            "split_method": (
                "Deduplicated, stratified modelling sample "
                "with 80/20 split"
            ),
            "special_controls": (
                f"{url_summary['duplicates_removed']} duplicate "
                "URLs removed; original URL retained only as "
                "metadata."
            ),
        },
    ]

    return pd.DataFrame(records)


def build_hwi_summary() -> pd.DataFrame:
    """Load and simplify the scenario evidence profiles."""

    profiles = pd.read_csv(
        HWI_PROFILES_PATH
    )

    audit = pd.read_csv(
        HWI_AUDIT_PATH
    )

    if audit["records_are_paired"].any():
        raise ValueError(
            "HWI profile audit unexpectedly reports paired records."
        )

    selected_columns = [
        "profile_name",
        "behavioural_hwi_score",
        "behavioural_hwi_category",
        "behavioural_evidence_status",
        "email_persuasion_risk_score",
        "email_persuasion_category",
        "url_technical_risk_score",
        "url_risk_category",
        "interpretation",
        "methodological_warning",
    ]

    return profiles[selected_columns].copy()


def build_markdown(
    performance: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    hwi_summary: pd.DataFrame,
) -> str:
    """Build a dissertation-ready result summary."""

    awareness = performance.loc[
        performance["track"]
        == "Behavioural awareness"
    ].iloc[0]

    ai_email = performance.loc[
        performance["track"]
        == "AI email"
    ].iloc[0]

    url = performance.loc[
        performance["track"]
        == "URL technical risk"
    ].iloc[0]

    lines = [
        "# Final Results Summary",
        "",
        "## Behavioural susceptibility track",
        "",
        (
            "The selected behavioural model achieved "
            f"ROC-AUC {awareness['roc_auc']:.3f}, "
            f"PR-AUC {awareness['pr_auc']:.3f}, and "
            f"Brier score {awareness['brier_score']:.3f}. "
            "Its discrimination was close to random chance. "
            "The model-derived HWI is therefore treated as an "
            "exploratory proxy rather than a validated measure "
            "of individual susceptibility."
        ),
        "",
        "## AI-email track",
        "",
        (
            "The selected AI-email model achieved "
            f"ROC-AUC {ai_email['roc_auc']:.3f}, "
            f"PR-AUC {ai_email['pr_auc']:.3f}, and "
            f"Brier score {ai_email['brier_score']:.3f}. "
            "Template-group separation was enforced and no "
            "shared template groups occurred between training "
            "and testing. A feature-leakage audit found no "
            "confirmed label-encoding feature. These results "
            "support email-level phishing classification, not "
            "measurement of human weakness."
        ),
        "",
        "## URL technical-risk track",
        "",
        (
            "The selected URL model achieved "
            f"ROC-AUC {url['roc_auc']:.3f}, "
            f"PR-AUC {url['pr_auc']:.3f}, and "
            f"Brier score {url['brier_score']:.3f}. "
            "The output is interpreted as URL-level technical "
            "maliciousness and not as an individual human "
            "susceptibility score."
        ),
        "",
        "## HWI framework",
        "",
        (
            "The final framework reports the behavioural HWI "
            "estimate, email persuasion risk, and URL technical "
            "risk as separate evidence dimensions. The records "
            "were selected independently and were not paired. "
            "No composite numeric score was calculated because "
            "the datasets contain no verified shared user, "
            "message, URL, or attack-session identifier."
        ),
        "",
        "### Demonstration profiles",
        "",
    ]

    for row in hwi_summary.to_dict(
        orient="records"
    ):
        lines.extend(
            [
                f"#### {row['profile_name']}",
                "",
                (
                    f"- Behavioural HWI: "
                    f"{row['behavioural_hwi_score']:.2f} "
                    f"({row['behavioural_hwi_category']}; "
                    f"evidence: "
                    f"{row['behavioural_evidence_status']})"
                ),
                (
                    f"- Email persuasion risk: "
                    f"{row['email_persuasion_risk_score']:.2f} "
                    f"({row['email_persuasion_category']})"
                ),
                (
                    f"- URL technical risk: "
                    f"{row['url_technical_risk_score']:.2f} "
                    f"({row['url_risk_category']})"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Overall conclusion",
            "",
            (
                "The implementation demonstrates how behavioural "
                "susceptibility, AI-email phishing characteristics, "
                "and URL maliciousness can be analysed within one "
                "transparent framework while remaining separate "
                "empirical dimensions. The available awareness "
                "data did not support reliable individual-level "
                "susceptibility classification. The email and URL "
                "tracks provided strong attack-characteristic "
                "classification but cannot independently validate "
                "the Human Weakness Index."
            ),
            "",
            "## Methodological limitation",
            "",
            (
                "The project used secondary datasets with different "
                "units of analysis. The scenario profiles are "
                "demonstrations only and do not represent verified "
                "real-world users or attacks."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Generate all consolidated final-result files."""

    required_paths = [
        AWARENESS_METRICS_PATH,
        AWARENESS_EVALUATION_PATH,
        AI_EMAIL_METRICS_PATH,
        AI_EMAIL_COMPARISON_PATH,
        AI_EMAIL_DUPLICATE_AUDIT_PATH,
        URL_METRICS_PATH,
        URL_COMPARISON_PATH,
        URL_DATASET_SUMMARY_PATH,
        HWI_PROFILES_PATH,
        HWI_AUDIT_PATH,
    ]

    for path in required_paths:
        require_file(path)

    performance = (
        build_performance_summary()
    )

    dataset_summary = (
        build_dataset_summary()
    )

    hwi_summary = (
        build_hwi_summary()
    )

    markdown = build_markdown(
        performance,
        dataset_summary,
        hwi_summary,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance.to_csv(
        PERFORMANCE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    dataset_summary.to_csv(
        DATASET_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    hwi_summary.to_csv(
        HWI_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    payload = {
        "model_performance": (
            performance.to_dict(
                orient="records"
            )
        ),
        "dataset_and_model_summary": (
            dataset_summary.to_dict(
                orient="records"
            )
        ),
        "hwi_framework_summary": (
            hwi_summary.to_dict(
                orient="records"
            )
        ),
        "overall_interpretation": (
            "The behavioural HWI is exploratory and unsupported "
            "as a reliable individual susceptibility classifier. "
            "The email and URL models provide separate contextual "
            "attack-risk indicators."
        ),
    }

    with JSON_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    MARKDOWN_OUTPUT_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print("=" * 80)
    print("FINAL RESULTS SUMMARY COMPLETE")
    print("=" * 80)
    print()
    print(
        performance[
            [
                "track",
                "selected_model",
                "roc_auc",
                "pr_auc",
                "brier_score",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()
    print(f"Performance: {PERFORMANCE_OUTPUT_PATH}")
    print(f"Dataset summary: {DATASET_OUTPUT_PATH}")
    print(f"HWI summary: {HWI_OUTPUT_PATH}")
    print(f"JSON summary: {JSON_OUTPUT_PATH}")
    print(f"Markdown summary: {MARKDOWN_OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())