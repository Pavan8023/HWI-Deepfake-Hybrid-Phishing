from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from typing import cast


class HWIError(RuntimeError):
    """Raised when an HWI operation cannot be completed safely."""


RiskDirection = Literal["higher_is_risk", "lower_is_risk"]


@dataclass(frozen=True)
class HWIThresholds:
    """
    Thresholds used to classify a 0–100 HWI score.

    These thresholds are provisional until calibration and outcome
    validation are completed.
    """

    low_max: float = 30.0
    medium_max: float = 70.0

    def validate(self) -> None:
        """Validate threshold ordering and range."""

        if not 0 <= self.low_max < self.medium_max <= 100:
            raise HWIError(
                "HWI thresholds must satisfy "
                "0 <= low_max < medium_max <= 100."
            )


@dataclass(frozen=True)
class IndicatorSpecification:
    """
    Definition of one transparent HWI proxy indicator.

    No risk direction is inferred automatically. The researcher must
    explicitly provide it based on dataset documentation and theory.
    """

    column: str
    weight: float
    direction: RiskDirection


@dataclass(frozen=True)
class HWIConfiguration:
    """Configuration used to generate and label HWI scores."""

    score_column: str = "hwi_score"
    category_column: str = "hwi_category"
    probability_column: str = "susceptibility_probability"
    thresholds: HWIThresholds = HWIThresholds()

    def validate(self) -> None:
        """Validate the configuration."""

        self.thresholds.validate()

        for value in (
            self.score_column,
            self.category_column,
            self.probability_column,
        ):
            if not value.strip():
                raise HWIError(
                    "HWI output-column names cannot be empty."
                )


DEFAULT_HWI_CONFIGURATION = HWIConfiguration()


def validate_probability_values(
    probabilities: pd.Series | Sequence[float] | np.ndarray,
) -> pd.Series:
    """
    Validate and return susceptibility probabilities.

    Every probability must be finite and between zero and one.
    """

    probability_series = pd.Series(
        probabilities,
        dtype="float64",
    )

    if probability_series.empty:
        raise HWIError(
            "Susceptibility probabilities cannot be empty."
        )

    if probability_series.isna().any():
        raise HWIError(
            "Susceptibility probabilities contain missing values."
        )

    values = probability_series.to_numpy(
        dtype=float,
    )

    if np.isinf(values).any():
        raise HWIError(
            "Susceptibility probabilities contain infinite values."
        )

    outside_range = (
        (probability_series < 0)
        | (probability_series > 1)
    )

    if outside_range.any():
        invalid_values = (
            probability_series.loc[outside_range]
            .head(10)
            .tolist()
        )

        raise HWIError(
            "Susceptibility probabilities must be between "
            f"zero and one. Invalid examples: {invalid_values}"
        )

    return probability_series.reset_index(
        drop=True
    )


def probability_to_hwi(
    probabilities: pd.Series | Sequence[float] | np.ndarray,
    *,
    decimals: int = 2,
) -> pd.Series:
    """
    Convert model susceptibility probabilities into HWI scores.

    Formula:

        HWI = susceptibility probability × 100

    This does not train a model. It converts validated model output into
    the HWI scale.
    """

    if decimals < 0:
        raise ValueError(
            "decimals must be zero or greater."
        )

    probability_series = validate_probability_values(
        probabilities
    )

    return (
        probability_series
        .mul(100)
        .round(decimals)
        .rename("hwi_score")
    )


def classify_hwi_score(
    score: float,
    *,
    thresholds: HWIThresholds = HWIThresholds(),
) -> str:
    """Classify one HWI score as low, medium or high."""

    thresholds.validate()

    if not np.isfinite(score):
        raise HWIError(
            f"HWI score must be finite: {score}"
        )

    if not 0 <= score <= 100:
        raise HWIError(
            f"HWI score must be between 0 and 100: {score}"
        )

    if score <= thresholds.low_max:
        return "low"

    if score <= thresholds.medium_max:
        return "medium"

    return "high"


def classify_hwi_scores(
    scores: pd.Series | Sequence[float] | np.ndarray,
    *,
    thresholds: HWIThresholds = HWIThresholds(),
) -> pd.Series:
    """Classify a collection of HWI scores."""

    thresholds.validate()

    score_series = pd.Series(
        scores,
        dtype="float64",
    )

    if score_series.empty:
        raise HWIError(
            "HWI scores cannot be empty."
        )

    if score_series.isna().any():
        raise HWIError(
            "HWI scores contain missing values."
        )

    invalid_mask = (
        ~np.isfinite(score_series)
        | (score_series < 0)
        | (score_series > 100)
    )

    if invalid_mask.any():
        raise HWIError(
            "All HWI scores must be finite and between "
            "zero and 100."
        )

    validated_scores = score_series.astype(float)

    categories = validated_scores.map(
        lambda value: classify_hwi_score(
            cast(float, value),
            thresholds=thresholds,
        )
    )

    return categories.rename(
        "hwi_category"
    )


def enrich_with_model_hwi(
    dataframe: pd.DataFrame,
    probabilities: pd.Series | Sequence[float] | np.ndarray,
    *,
    configuration: HWIConfiguration = DEFAULT_HWI_CONFIGURATION,
) -> pd.DataFrame:
    """
    Add model-derived HWI values to a dataset.

    The input DataFrame is copied and never modified in place.
    """

    configuration.validate()

    if dataframe.empty:
        raise HWIError(
            "Cannot add HWI values to an empty DataFrame."
        )

    probability_series = validate_probability_values(
        probabilities
    )

    if len(probability_series) != len(dataframe):
        raise HWIError(
            "Probability count must match DataFrame row count. "
            f"Rows={len(dataframe)}, "
            f"probabilities={len(probability_series)}"
        )

    result = dataframe.reset_index(
        drop=True
    ).copy()

    hwi_scores = probability_to_hwi(
        probability_series
    )

    hwi_categories = classify_hwi_scores(
        hwi_scores,
        thresholds=configuration.thresholds,
    )

    result[
        configuration.probability_column
    ] = probability_series

    result[
        configuration.score_column
    ] = hwi_scores

    result[
        configuration.category_column
    ] = hwi_categories

    result["hwi_method"] = (
        "model_probability"
    )

    return result


def percentile_risk_score(
    series: pd.Series,
    *,
    direction: RiskDirection,
) -> pd.Series:
    """
    Convert a numeric indicator into a percentile-based risk score.

    The direction must be explicitly supplied.

    This is suitable for a provisional transparent proxy only. It is not
    independent validation of human susceptibility.
    """

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if numeric_series.isna().any():
        raise HWIError(
            f"Indicator '{series.name}' contains missing or "
            "non-numeric values."
        )

    if numeric_series.nunique() <= 1:
        return pd.Series(
            np.zeros(len(numeric_series)),
            index=series.index,
            dtype="float64",
            name=f"{series.name}_risk",
        )

    percentile = numeric_series.rank(
        method="average",
        pct=True,
    )

    if direction == "higher_is_risk":
        risk = percentile

    elif direction == "lower_is_risk":
        risk = 1.0 - percentile

    else:
        raise HWIError(
            f"Unsupported risk direction: {direction}"
        )

    return (
        risk.clip(0, 1)
        .rename(f"{series.name}_risk")
    )


def validate_indicator_specifications(
    dataframe: pd.DataFrame,
    specifications: Sequence[IndicatorSpecification],
    *,
    forbidden_columns: Sequence[str] = (
        "clicked_link",
        "phishing_label",
        "target",
        "label",
    ),
) -> None:
    """
    Validate proxy-indicator specifications.

    Target and label columns are forbidden by default to prevent circular
    HWI construction.
    """

    if not specifications:
        raise HWIError(
            "At least one indicator specification is required."
        )

    forbidden = {
        value.lower()
        for value in forbidden_columns
    }

    total_weight = 0.0
    seen_columns: set[str] = set()

    for specification in specifications:
        column = specification.column

        if column not in dataframe.columns:
            raise HWIError(
                f"HWI indicator column not found: {column}"
            )

        if column.lower() in forbidden:
            raise HWIError(
                f"Target or label column cannot be used as an "
                f"HWI input indicator: {column}"
            )

        if column in seen_columns:
            raise HWIError(
                f"Duplicate HWI indicator specification: {column}"
            )

        if specification.weight <= 0:
            raise HWIError(
                f"Indicator weight must be positive: {column}"
            )

        if specification.direction not in {
            "higher_is_risk",
            "lower_is_risk",
        }:
            raise HWIError(
                f"Invalid risk direction for {column}: "
                f"{specification.direction}"
            )

        seen_columns.add(column)
        total_weight += specification.weight

    if not np.isclose(
        total_weight,
        1.0,
        atol=1e-9,
    ):
        raise HWIError(
            "Indicator weights must sum to one. "
            f"Current sum: {total_weight}"
        )


def build_transparent_proxy_hwi(
    dataframe: pd.DataFrame,
    specifications: Sequence[IndicatorSpecification],
    *,
    configuration: HWIConfiguration = DEFAULT_HWI_CONFIGURATION,
    forbidden_columns: Sequence[str] = (
        "clicked_link",
        "phishing_label",
        "target",
        "label",
    ),
) -> pd.DataFrame:
    """
    Construct a provisional rule-based HWI proxy.

    This method must not be described as independently validated human
    susceptibility. It is provided for transparent exploratory comparison
    before model-derived probabilities are available.
    """

    configuration.validate()

    if dataframe.empty:
        raise HWIError(
            "Cannot create an HWI proxy from an empty DataFrame."
        )

    validate_indicator_specifications(
        dataframe,
        specifications,
        forbidden_columns=forbidden_columns,
    )

    result = dataframe.reset_index(
        drop=True
    ).copy()

    weighted_components: list[pd.Series] = []

    for specification in specifications:
        component = percentile_risk_score(
            result[specification.column],
            direction=specification.direction,
        )

        component_column = (
            f"hwi_component_{specification.column}"
        )

        result[component_column] = (
            component.mul(100).round(2)
        )

        weighted_components.append(
            component.mul(
                specification.weight
            )
        )

    weighted_score = sum(
        weighted_components,
        start=pd.Series(
            np.zeros(len(result)),
            index=result.index,
            dtype="float64",
        ),
    )

    result[
        configuration.score_column
    ] = (
        weighted_score
        .mul(100)
        .clip(0, 100)
        .round(2)
    )

    result[
        configuration.category_column
    ] = classify_hwi_scores(
        result[configuration.score_column],
        thresholds=configuration.thresholds,
    )

    result["hwi_method"] = (
        "transparent_proxy"
    )

    return result


def validate_hwi_distribution(
    dataframe: pd.DataFrame,
    *,
    score_column: str = "hwi_score",
    category_column: str = "hwi_category",
) -> dict[str, Any]:
    """Generate structural and distribution checks for HWI output."""

    required_columns = {
        score_column,
        category_column,
    }

    missing_columns = sorted(
        required_columns.difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise HWIError(
            "Required HWI columns are missing: "
            + ", ".join(missing_columns)
        )

    scores = pd.to_numeric(
        dataframe[score_column],
        errors="coerce",
    )

    if scores.isna().any():
        raise HWIError(
            "HWI score column contains missing or non-numeric values."
        )

    invalid_score_count = int(
        (
            ~np.isfinite(scores)
            | (scores < 0)
            | (scores > 100)
        ).sum()
    )

    category_counts = (
        dataframe[category_column]
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "rows": int(len(dataframe)),
        "score_column": score_column,
        "category_column": category_column,
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "standard_deviation": float(
            scores.std()
        ),
        "first_quartile": float(
            scores.quantile(0.25)
        ),
        "third_quartile": float(
            scores.quantile(0.75)
        ),
        "invalid_score_count": (
            invalid_score_count
        ),
        "category_counts": {
            str(key): int(value)
            for key, value in (
                category_counts.items()
            )
        },
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "total_missing_values": int(
            dataframe.isna().sum().sum()
        ),
    }


def validate_hwi_against_binary_outcome(
    dataframe: pd.DataFrame,
    *,
    outcome_column: str,
    score_column: str = "hwi_score",
    positive_values: Sequence[Any] = (
        1,
        "1",
        True,
        "yes",
        "clicked",
    ),
) -> dict[str, Any]:
    """
    Perform a basic outcome-alignment check.

    This does not replace model evaluation. It only compares mean and median
    HWI values between observed binary outcome groups.
    """

    if outcome_column not in dataframe.columns:
        raise HWIError(
            f"Outcome column not found: {outcome_column}"
        )

    if score_column not in dataframe.columns:
        raise HWIError(
            f"HWI score column not found: {score_column}"
        )

    normalized_positive_values = {
        str(value).strip().lower()
        for value in positive_values
    }

    normalized_outcome = (
        dataframe[outcome_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    binary_outcome = normalized_outcome.isin(
        normalized_positive_values
    ).astype(int)

    if binary_outcome.nunique() != 2:
        raise HWIError(
            "Outcome validation requires two observed classes."
        )

    scores = pd.to_numeric(
        dataframe[score_column],
        errors="coerce",
    )

    comparison = pd.DataFrame(
        {
            "outcome": binary_outcome,
            "hwi_score": scores,
        }
    ).dropna()

    grouped = comparison.groupby(
        "outcome"
    )["hwi_score"]

    return {
        "negative_count": int(
            (comparison["outcome"] == 0).sum()
        ),
        "positive_count": int(
            (comparison["outcome"] == 1).sum()
        ),
        "negative_mean_hwi": float(
            grouped.mean().loc[0]
        ),
        "positive_mean_hwi": float(
            grouped.mean().loc[1]
        ),
        "negative_median_hwi": float(
            grouped.median().loc[0]
        ),
        "positive_median_hwi": float(
            grouped.median().loc[1]
        ),
        "mean_difference": float(
            grouped.mean().loc[1]
            - grouped.mean().loc[0]
        ),
        "interpretation_warning": (
            "A higher score among positive outcomes supports "
            "alignment but does not independently validate causation "
            "or generalisability."
        ),
    }


def save_hwi_outputs(
    dataframe: pd.DataFrame,
    *,
    dataset_output_path: str | Path,
    summary_output_path: str | Path,
    score_column: str = "hwi_score",
    category_column: str = "hwi_category",
    additional_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Save an HWI-enriched dataset and its distribution summary."""

    dataset_path = Path(
        dataset_output_path
    )
    summary_path = Path(
        summary_output_path
    )

    dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
        encoding="utf-8",
    )

    summary = validate_hwi_distribution(
        dataframe,
        score_column=score_column,
        category_column=category_column,
    )

    if additional_metadata:
        summary["metadata"] = dict(
            additional_metadata
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

    return {
        "dataset": dataset_path,
        "summary": summary_path,
    }


def hwi_configuration_to_dict(
    configuration: HWIConfiguration,
) -> dict[str, Any]:
    """Convert an HWI configuration into JSON-compatible data."""

    configuration.validate()

    return asdict(configuration)