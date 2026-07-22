from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation import save_evaluation_outputs  # noqa: E402


HWI_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "awareness_test_hwi.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "model_training"
    / "awareness_model_metrics.json"
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "evaluation"
)

FIGURES_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "evaluation"
)


def main() -> int:
    """Evaluate the selected awareness susceptibility model."""

    if not HWI_PATH.exists():
        raise FileNotFoundError(
            f"HWI prediction file not found: {HWI_PATH}"
        )

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Model metrics file not found: {METRICS_PATH}"
        )

    hwi_data = pd.read_csv(
        HWI_PATH
    )

    with METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    required_columns = {
        "actual_clicked_link",
        "susceptibility_probability",
    }

    missing_columns = (
        required_columns.difference(
            hwi_data.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "HWI prediction file is missing: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    model_name = str(
        metrics["best_hwi_model"]
    )

    summary = save_evaluation_outputs(
        y_true=hwi_data[
            "actual_clicked_link"
        ],
        probabilities=hwi_data[
            "susceptibility_probability"
        ],
        reports_directory=(
            REPORTS_DIRECTORY
        ),
        figures_directory=(
            FIGURES_DIRECTORY
        ),
        model_name=model_name,
    )

    print("=" * 80)
    print("FINAL AWARENESS MODEL EVALUATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(
        f"ROC-AUC: "
        f"{summary['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC: "
        f"{summary['pr_auc']:.4f}"
    )
    print(
        f"Brier score: "
        f"{summary['brier_score']:.4f}"
    )
    print(
        "Evidence classification:",
        summary[
            "evidence_classification"
        ],
    )

    interval = summary[
        "bootstrap_roc_auc"
    ]

    print(
        "ROC-AUC bootstrap interval:",
        (
            f"{interval['roc_auc_lower']:.4f} – "
            f"{interval['roc_auc_upper']:.4f}"
        ),
    )

    print(
        "Mean probability difference:",
        (
            f"{summary['mean_probability_difference']:.4f}"
        ),
    )

    print(
        f"Reports: {REPORTS_DIRECTORY}"
    )
    print(
        f"Figures: {FIGURES_DIRECTORY}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())