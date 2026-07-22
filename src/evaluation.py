from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class EvaluationError(RuntimeError):
    """Raised when model evaluation cannot be completed safely."""


def validate_binary_evaluation_inputs(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary outcomes and positive-class probabilities."""

    targets = np.asarray(y_true, dtype=int)
    probability_values = np.asarray(
        probabilities,
        dtype=float,
    )

    if targets.ndim != 1:
        raise EvaluationError(
            "Targets must be one-dimensional."
        )

    if probability_values.ndim != 1:
        raise EvaluationError(
            "Probabilities must be one-dimensional."
        )

    if len(targets) != len(probability_values):
        raise EvaluationError(
            "Targets and probabilities must have equal lengths."
        )

    if len(targets) == 0:
        raise EvaluationError(
            "Evaluation inputs cannot be empty."
        )

    if set(np.unique(targets)) != {0, 1}:
        raise EvaluationError(
            "Targets must contain both binary classes 0 and 1."
        )

    if (
        np.isnan(probability_values).any()
        or np.isinf(probability_values).any()
    ):
        raise EvaluationError(
            "Probabilities contain missing or infinite values."
        )

    if (
        (probability_values < 0).any()
        or (probability_values > 1).any()
    ):
        raise EvaluationError(
            "Probabilities must remain between zero and one."
        )

    return targets, probability_values


def classify_model_evidence(
    roc_auc: float,
) -> str:
    """
    Classify discriminatory evidence conservatively.

    This is a project reporting rule, not a universal statistical standard.
    """

    if roc_auc < 0.55:
        return "unsupported"

    if roc_auc < 0.65:
        return "weak"

    if roc_auc < 0.75:
        return "moderate"

    return "strong"


def bootstrap_roc_auc_interval(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    *,
    iterations: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Estimate a percentile bootstrap confidence interval for ROC-AUC."""

    targets, probability_values = (
        validate_binary_evaluation_inputs(
            y_true,
            probabilities,
        )
    )

    if iterations < 100:
        raise ValueError(
            "iterations must be at least 100."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    generator = np.random.default_rng(
        random_state
    )

    sample_size = len(targets)
    estimates: list[float] = []

    for _ in range(iterations):
        indices = generator.integers(
            0,
            sample_size,
            size=sample_size,
        )

        sampled_targets = targets[indices]

        if np.unique(sampled_targets).size < 2:
            continue

        sampled_probabilities = (
            probability_values[indices]
        )

        estimates.append(
            float(
                roc_auc_score(
                    sampled_targets,
                    sampled_probabilities,
                )
            )
        )

    if not estimates:
        raise EvaluationError(
            "No valid bootstrap estimates were generated."
        )

    alpha = 1.0 - confidence_level

    lower = float(
        np.quantile(
            estimates,
            alpha / 2,
        )
    )

    upper = float(
        np.quantile(
            estimates,
            1 - alpha / 2,
        )
    )

    return {
        "iterations_requested": iterations,
        "iterations_completed": len(estimates),
        "confidence_level": confidence_level,
        "roc_auc_mean": float(
            np.mean(estimates)
        ),
        "roc_auc_lower": lower,
        "roc_auc_upper": upper,
    }


def build_probability_summary(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Summarise probabilities separately for observed outcome classes."""

    targets, probability_values = (
        validate_binary_evaluation_inputs(
            y_true,
            probabilities,
        )
    )

    frame = pd.DataFrame(
        {
            "actual_class": targets,
            "probability": probability_values,
        }
    )

    records: list[dict[str, Any]] = []

    for target_class in [0, 1]:
        values = frame.loc[
            frame["actual_class"] == target_class,
            "probability",
        ]

        records.append(
            {
                "actual_class": target_class,
                "count": int(len(values)),
                "mean_probability": float(
                    values.mean()
                ),
                "median_probability": float(
                    values.median()
                ),
                "standard_deviation": float(
                    values.std()
                ),
                "minimum": float(values.min()),
                "first_quartile": float(
                    values.quantile(0.25)
                ),
                "third_quartile": float(
                    values.quantile(0.75)
                ),
                "maximum": float(values.max()),
            }
        )

    return pd.DataFrame(records)


def calculate_evaluation_summary(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    *,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Calculate final discrimination, calibration and HWI-alignment results."""

    targets, probability_values = (
        validate_binary_evaluation_inputs(
            y_true,
            probabilities,
        )
    )

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between zero and one."
        )

    predictions = (
        probability_values >= threshold
    ).astype(int)

    roc_auc = float(
        roc_auc_score(
            targets,
            probability_values,
        )
    )

    pr_auc = float(
        average_precision_score(
            targets,
            probability_values,
        )
    )

    brier_score = float(
        brier_score_loss(
            targets,
            probability_values,
        )
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    negative_probabilities = (
        probability_values[targets == 0]
    )

    positive_probabilities = (
        probability_values[targets == 1]
    )

    negative_mean = float(
        negative_probabilities.mean()
    )

    positive_mean = float(
        positive_probabilities.mean()
    )

    return {
        "rows": int(len(targets)),
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier_score,
        "evidence_classification": (
            classify_model_evidence(
                roc_auc
            )
        ),
        "negative_mean_probability": (
            negative_mean
        ),
        "positive_mean_probability": (
            positive_mean
        ),
        "mean_probability_difference": (
            positive_mean - negative_mean
        ),
        "probability_minimum": float(
            probability_values.min()
        ),
        "probability_maximum": float(
            probability_values.max()
        ),
        "probability_mean": float(
            probability_values.mean()
        ),
        "probability_standard_deviation": (
            float(
                probability_values.std()
            )
        ),
        "confusion_matrix": (
            matrix.astype(int).tolist()
        ),
        "scientific_interpretation": (
            "The HWI is an exploratory model-derived "
            "susceptibility probability. Discrimination "
            "close to ROC-AUC 0.50 indicates that the "
            "available behavioural and contextual predictors "
            "do not reliably distinguish clicking outcomes."
        ),
    }


def plot_roc_curve(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a receiver operating characteristic curve."""

    targets, probability_values = (
        validate_binary_evaluation_inputs(
            y_true,
            probabilities,
        )
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            targets,
            probability_values,
        )
    )

    roc_auc = roc_auc_score(
        targets,
        probability_values,
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.3f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(
        "Awareness susceptibility model ROC curve"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return path


def plot_precision_recall_curve(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a precision–recall curve."""

    targets, probability_values = (
        validate_binary_evaluation_inputs(
            y_true,
            probabilities,
        )
    )

    precision, recall, _ = (
        precision_recall_curve(
            targets,
            probability_values,
        )
    )

    average_precision = (
        average_precision_score(
            targets,
            probability_values,
        )
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=(
            f"Average precision = "
            f"{average_precision:.3f}"
        ),
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Awareness susceptibility precision–recall curve"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return path


def plot_calibration_curve(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    output_path: str | Path,
    *,
    bins: int = 10,
) -> Path:
    """Save a probability-calibration curve."""

    targets, probability_values = (
        validate_binary_evaluation_inputs(
            y_true,
            probabilities,
        )
    )

    observed_fraction, predicted_mean = (
        calibration_curve(
            targets,
            probability_values,
            n_bins=bins,
            strategy="quantile",
        )
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        predicted_mean,
        observed_fraction,
        marker="o",
        label="Selected model",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive proportion")
    plt.title(
        "Awareness susceptibility calibration curve"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return path


def plot_confusion_matrix(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    output_path: str | Path,
    *,
    threshold: float = 0.50,
) -> Path:
    """Save a confusion matrix at the selected threshold."""

    targets, probability_values = (
        validate_binary_evaluation_inputs(
            y_true,
            probabilities,
        )
    )

    predictions = (
        probability_values >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Did not click",
            "Clicked",
        ],
    )

    display.plot()
    plt.title(
        "Awareness susceptibility confusion matrix"
    )
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return path


def plot_probability_distribution(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    output_path: str | Path,
) -> Path:
    """Plot predicted probability distributions by observed class."""

    targets, probability_values = (
        validate_binary_evaluation_inputs(
            y_true,
            probabilities,
        )
    )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(9, 6))
    plt.hist(
        probability_values[targets == 0],
        bins=25,
        alpha=0.6,
        label="Did not click",
    )
    plt.hist(
        probability_values[targets == 1],
        bins=25,
        alpha=0.6,
        label="Clicked",
    )
    plt.xlabel(
        "Predicted susceptibility probability"
    )
    plt.ylabel("Frequency")
    plt.title(
        "Predicted susceptibility probability distribution"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return path


def save_evaluation_outputs(
    *,
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    reports_directory: str | Path,
    figures_directory: str | Path,
    model_name: str,
) -> dict[str, Any]:
    """Generate and save the complete final awareness evaluation."""

    reports_path = Path(reports_directory)
    figures_path = Path(figures_directory)

    reports_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = calculate_evaluation_summary(
        y_true,
        probabilities,
    )

    bootstrap = bootstrap_roc_auc_interval(
        y_true,
        probabilities,
    )

    probability_summary = (
        build_probability_summary(
            y_true,
            probabilities,
        )
    )

    figure_paths = {
        "roc_curve": plot_roc_curve(
            y_true,
            probabilities,
            figures_path
            / "awareness_roc_curve.png",
        ),
        "precision_recall_curve": (
            plot_precision_recall_curve(
                y_true,
                probabilities,
                figures_path
                / "awareness_precision_recall_curve.png",
            )
        ),
        "calibration_curve": (
            plot_calibration_curve(
                y_true,
                probabilities,
                figures_path
                / "awareness_calibration_curve.png",
            )
        ),
        "confusion_matrix": (
            plot_confusion_matrix(
                y_true,
                probabilities,
                figures_path
                / "awareness_confusion_matrix.png",
            )
        ),
        "probability_distribution": (
            plot_probability_distribution(
                y_true,
                probabilities,
                figures_path
                / "awareness_probability_distribution.png",
            )
        ),
    }

    summary["model_name"] = model_name
    summary["bootstrap_roc_auc"] = bootstrap
    summary["figures"] = {
        key: str(value)
        for key, value in figure_paths.items()
    }

    summary_path = (
        reports_path
        / "awareness_final_evaluation.json"
    )

    probability_summary_path = (
        reports_path
        / "awareness_probability_summary.csv"
    )

    probability_summary.to_csv(
        probability_summary_path,
        index=False,
        encoding="utf-8",
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    return summary