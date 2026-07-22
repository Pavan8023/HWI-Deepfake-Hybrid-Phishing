from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    kruskal,
    mannwhitneyu,
)

from src.config import (
    FEATURE_VALIDATION_ALPHA,
    FEATURE_VALIDATION_FIGURES_DIR,
    FEATURE_VALIDATION_HIGH_CORRELATION,
    FEATURE_VALIDATION_REPORTS_DIR,
    FEATURE_VALIDATION_STATISTICS_DIR,
)
from src.utils import get_logger


LOGGER = get_logger(__name__)


class FeatureValidationError(RuntimeError):
    """Raised when feature validation cannot be completed safely."""


DEFAULT_LABEL_COLUMNS = {
    "source_type",
    "email_class",
    "source_label",
    "phishing_label",
    "original_label",
    "experiment_group",
}

EXPECTED_EXPERIMENT_GROUPS = {
    "human_legitimate",
    "human_phishing",
    "llm_legitimate",
    "llm_phishing",
}


def sanitize_filename(value: str) -> str:
    """Convert arbitrary text into a safe filename component."""

    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value.strip(),
    )
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_").lower()

    return cleaned or "feature"


def validate_ai_email_feature_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate the structure of the engineered AI-email feature dataset.

    This function does not alter the dataset.
    """

    if dataframe.empty:
        raise FeatureValidationError(
            "The AI-email feature dataset is empty."
        )

    required_columns = {
        "source_type",
        "email_class",
        "source_label",
        "phishing_label",
        "experiment_group",
    }

    missing_required_columns = sorted(
        required_columns.difference(dataframe.columns)
    )

    if missing_required_columns:
        raise FeatureValidationError(
            "Required columns are missing: "
            + ", ".join(missing_required_columns)
        )

    observed_groups = set(
        dataframe["experiment_group"]
        .dropna()
        .astype(str)
        .unique()
    )

    missing_groups = sorted(
        EXPECTED_EXPERIMENT_GROUPS.difference(
            observed_groups
        )
    )

    unexpected_groups = sorted(
        observed_groups.difference(
            EXPECTED_EXPERIMENT_GROUPS
        )
    )

    group_counts = (
        dataframe["experiment_group"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    return {
        "rows": int(len(dataframe)),
        "columns": int(dataframe.shape[1]),
        "missing_required_columns": (
            missing_required_columns
        ),
        "observed_groups": sorted(observed_groups),
        "missing_expected_groups": missing_groups,
        "unexpected_groups": unexpected_groups,
        "group_counts": {
            str(key): int(value)
            for key, value in group_counts.items()
        },
        "total_missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
    }


def select_numeric_feature_columns(
    dataframe: pd.DataFrame,
    *,
    excluded_columns: Iterable[str] = DEFAULT_LABEL_COLUMNS,
) -> list[str]:
    """
    Select numeric engineered features while excluding labels.

    Identifier, target, source and experimental-label columns are not
    considered input features.
    """

    excluded = set(excluded_columns)

    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return sorted(
        column
        for column in numeric_columns
        if column not in excluded
    )


def build_feature_quality_report(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Build a quality report for engineered features.

    Reports missingness, infinite values, uniqueness and constant status.
    """

    records: list[dict[str, Any]] = []

    for feature in feature_columns:
        if feature not in dataframe.columns:
            raise FeatureValidationError(
                f"Feature column not found: {feature}"
            )

        numeric_series = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

        finite_series = numeric_series.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        non_missing = finite_series.dropna()
        unique_count = int(non_missing.nunique())

        records.append(
            {
                "feature": feature,
                "dtype": str(dataframe[feature].dtype),
                "rows": int(len(dataframe)),
                "missing_count": int(
                    numeric_series.isna().sum()
                ),
                "missing_percentage": round(
                    float(
                        numeric_series.isna().mean()
                        * 100
                    ),
                    6,
                ),
                "infinite_count": int(
                    np.isinf(
                        numeric_series.to_numpy(
                            dtype=float,
                            na_value=np.nan,
                        )
                    ).sum()
                ),
                "unique_count": unique_count,
                "constant_feature": unique_count <= 1,
                "minimum": (
                    float(non_missing.min())
                    if not non_missing.empty
                    else None
                ),
                "maximum": (
                    float(non_missing.max())
                    if not non_missing.empty
                    else None
                ),
                "mean": (
                    float(non_missing.mean())
                    if not non_missing.empty
                    else None
                ),
                "median": (
                    float(non_missing.median())
                    if not non_missing.empty
                    else None
                ),
                "standard_deviation": (
                    float(non_missing.std())
                    if len(non_missing) > 1
                    else None
                ),
            }
        )

    return pd.DataFrame(records)


def build_group_descriptive_statistics(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    group_column: str = "experiment_group",
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each feature and experiment group.
    """

    if group_column not in dataframe.columns:
        raise FeatureValidationError(
            f"Group column not found: {group_column}"
        )

    records: list[dict[str, Any]] = []

    groups = sorted(
        dataframe[group_column]
        .dropna()
        .astype(str)
        .unique()
    )

    for feature in feature_columns:
        for group in groups:
            values = pd.to_numeric(
                dataframe.loc[
                    dataframe[group_column] == group,
                    feature,
                ],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

            values = values.dropna()

            if values.empty:
                continue

            first_quartile = float(
                values.quantile(0.25)
            )
            third_quartile = float(
                values.quantile(0.75)
            )

            records.append(
                {
                    "feature": feature,
                    "group": group,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "standard_deviation": (
                        float(values.std())
                        if len(values) > 1
                        else 0.0
                    ),
                    "minimum": float(values.min()),
                    "first_quartile": first_quartile,
                    "median": float(values.median()),
                    "third_quartile": third_quartile,
                    "maximum": float(values.max()),
                    "interquartile_range": (
                        third_quartile
                        - first_quartile
                    ),
                }
            )

    return pd.DataFrame(records)


def calculate_epsilon_squared(
    h_statistic: float,
    sample_size: int,
    group_count: int,
) -> float:
    """
    Calculate epsilon-squared effect size for Kruskal-Wallis.

    Values are bounded at zero because sampling variation can produce a
    slightly negative estimate.
    """

    if sample_size <= group_count:
        return 0.0

    effect_size = (
        h_statistic - group_count + 1
    ) / (
        sample_size - group_count
    )

    return max(0.0, float(effect_size))


def classify_effect_size(
    effect_size: float,
) -> str:
    """
    Classify an absolute effect size using broad exploratory thresholds.

    These labels are descriptive guidance, not universal scientific laws.
    """

    absolute_effect = abs(effect_size)

    if absolute_effect < 0.01:
        return "negligible"

    if absolute_effect < 0.06:
        return "small"

    if absolute_effect < 0.14:
        return "medium"

    return "large"


def run_kruskal_wallis_tests(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    group_column: str = "experiment_group",
    alpha: float = FEATURE_VALIDATION_ALPHA,
) -> pd.DataFrame:
    """
    Compare all four experimental groups using Kruskal-Wallis tests.

    Kruskal-Wallis is used because many count, ratio and text features may
    not be normally distributed.
    """

    if group_column not in dataframe.columns:
        raise FeatureValidationError(
            f"Group column not found: {group_column}"
        )

    groups = sorted(
        dataframe[group_column]
        .dropna()
        .astype(str)
        .unique()
    )

    records: list[dict[str, Any]] = []

    for feature in feature_columns:
        grouped_values: list[np.ndarray] = []

        total_sample_size = 0

        for group in groups:
            values = pd.to_numeric(
                dataframe.loc[
                    dataframe[group_column] == group,
                    feature,
                ],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            array = values.to_numpy(dtype=float)
            grouped_values.append(array)
            total_sample_size += len(array)

        non_empty_groups = [
            values
            for values in grouped_values
            if len(values) > 0
        ]

        unique_values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna().nunique()

        if (
            len(non_empty_groups) < 2
            or unique_values <= 1
        ):
            records.append(
                {
                    "feature": feature,
                    "h_statistic": np.nan,
                    "p_value": np.nan,
                    "epsilon_squared": 0.0,
                    "effect_size_category": "constant",
                    "group_count": len(
                        non_empty_groups
                    ),
                    "sample_size": total_sample_size,
                    "significant_raw": False,
                    "test_status": (
                        "not_tested_constant_or_insufficient"
                    ),
                }
            )
            continue

        test_result = kruskal(
            *non_empty_groups,
            nan_policy="omit",
        )

        h_statistic = float(
            test_result.statistic
        )
        p_value = float(test_result.pvalue)

        epsilon_squared = (
            calculate_epsilon_squared(
                h_statistic=h_statistic,
                sample_size=total_sample_size,
                group_count=len(
                    non_empty_groups
                ),
            )
        )

        records.append(
            {
                "feature": feature,
                "h_statistic": h_statistic,
                "p_value": p_value,
                "epsilon_squared": (
                    epsilon_squared
                ),
                "effect_size_category": (
                    classify_effect_size(
                        epsilon_squared
                    )
                ),
                "group_count": len(
                    non_empty_groups
                ),
                "sample_size": total_sample_size,
                "significant_raw": (
                    p_value < alpha
                ),
                "test_status": "completed",
            }
        )

    results = pd.DataFrame(records)

    if not results.empty:
        results["p_value_holm"] = holm_adjust(
            results["p_value"].tolist()
        )

        results["significant_holm"] = (
            results["p_value_holm"] < alpha
        ) & results["p_value_holm"].notna()

        results = results.sort_values(
            by=[
                "p_value_holm",
                "epsilon_squared",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        ).reset_index(drop=True)

    return results


def holm_adjust(
    p_values: Iterable[float],
) -> list[float]:
    """
    Apply the Holm family-wise error-rate correction.

    NaN values remain NaN.
    """

    p_value_list = list(p_values)

    valid_items = [
        (index, float(value))
        for index, value in enumerate(
            p_value_list
        )
        if pd.notna(value)
    ]

    adjusted = [
        float("nan")
        for _ in p_value_list
    ]

    if not valid_items:
        return adjusted

    sorted_items = sorted(
        valid_items,
        key=lambda item: item[1],
    )

    number_of_tests = len(sorted_items)
    running_maximum = 0.0

    for rank, (original_index, p_value) in enumerate(
        sorted_items,
        start=1,
    ):
        multiplier = (
            number_of_tests - rank + 1
        )

        candidate = min(
            1.0,
            p_value * multiplier,
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[original_index] = (
            running_maximum
        )

    return adjusted


def calculate_rank_biserial_correlation(
    u_statistic: float,
    first_sample_size: int,
    second_sample_size: int,
) -> float:
    """
    Calculate signed rank-biserial correlation from Mann-Whitney U.

    Positive values indicate that the first group tends to contain larger
    values than the second group.
    """

    denominator = (
        first_sample_size
        * second_sample_size
    )

    if denominator == 0:
        return 0.0

    return float(
        (2 * u_statistic / denominator)
        - 1
    )


def run_pairwise_mann_whitney_tests(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    group_column: str = "experiment_group",
    alpha: float = FEATURE_VALIDATION_ALPHA,
) -> pd.DataFrame:
    """
    Run pairwise Mann-Whitney U comparisons between experiment groups.

    Holm correction is applied across every generated pairwise test.
    """

    if group_column not in dataframe.columns:
        raise FeatureValidationError(
            f"Group column not found: {group_column}"
        )

    groups = sorted(
        dataframe[group_column]
        .dropna()
        .astype(str)
        .unique()
    )

    records: list[dict[str, Any]] = []

    for feature in feature_columns:
        for first_group, second_group in combinations(
            groups,
            2,
        ):
            first_values = pd.to_numeric(
                dataframe.loc[
                    dataframe[group_column]
                    == first_group,
                    feature,
                ],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            second_values = pd.to_numeric(
                dataframe.loc[
                    dataframe[group_column]
                    == second_group,
                    feature,
                ],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if (
                first_values.empty
                or second_values.empty
            ):
                continue

            combined_unique_values = pd.concat(
                [first_values, second_values],
                ignore_index=True,
            ).nunique()

            if combined_unique_values <= 1:
                continue

            test_result = mannwhitneyu(
                first_values,
                second_values,
                alternative="two-sided",
                method="auto",
            )

            u_statistic = float(
                test_result.statistic
            )
            p_value = float(
                test_result.pvalue
            )

            effect_size = (
                calculate_rank_biserial_correlation(
                    u_statistic=u_statistic,
                    first_sample_size=len(
                        first_values
                    ),
                    second_sample_size=len(
                        second_values
                    ),
                )
            )

            records.append(
                {
                    "feature": feature,
                    "group_1": first_group,
                    "group_2": second_group,
                    "group_1_count": int(
                        len(first_values)
                    ),
                    "group_2_count": int(
                        len(second_values)
                    ),
                    "group_1_median": float(
                        first_values.median()
                    ),
                    "group_2_median": float(
                        second_values.median()
                    ),
                    "u_statistic": u_statistic,
                    "p_value": p_value,
                    "rank_biserial_correlation": (
                        effect_size
                    ),
                    "absolute_effect_size": abs(
                        effect_size
                    ),
                    "effect_size_category": (
                        classify_effect_size(
                            effect_size
                        )
                    ),
                    "significant_raw": (
                        p_value < alpha
                    ),
                }
            )

    results = pd.DataFrame(records)

    if not results.empty:
        results["p_value_holm"] = holm_adjust(
            results["p_value"].tolist()
        )

        results["significant_holm"] = (
            results["p_value_holm"] < alpha
        )

        results = results.sort_values(
            by=[
                "p_value_holm",
                "absolute_effect_size",
            ],
            ascending=[
                True,
                False,
            ],
        ).reset_index(drop=True)

    return results


def run_binary_target_comparison(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    target_column: str,
    target_name: str,
    alpha: float = FEATURE_VALIDATION_ALPHA,
) -> pd.DataFrame:
    """
    Compare each numeric feature across a binary target.

    Suitable examples:

    - phishing_label: legitimate versus phishing
    - source_label: human versus LLM
    """

    if target_column not in dataframe.columns:
        raise FeatureValidationError(
            f"Target column not found: {target_column}"
        )

    target_values = sorted(
        dataframe[target_column]
        .dropna()
        .unique()
        .tolist()
    )

    if len(target_values) != 2:
        raise FeatureValidationError(
            f"{target_column} must contain exactly "
            "two non-null classes."
        )

    first_target = target_values[0]
    second_target = target_values[1]

    records: list[dict[str, Any]] = []

    for feature in feature_columns:
        first_values = pd.to_numeric(
            dataframe.loc[
                dataframe[target_column]
                == first_target,
                feature,
            ],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        second_values = pd.to_numeric(
            dataframe.loc[
                dataframe[target_column]
                == second_target,
                feature,
            ],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if (
            first_values.empty
            or second_values.empty
        ):
            continue

        combined_unique_values = pd.concat(
            [first_values, second_values],
            ignore_index=True,
        ).nunique()

        if combined_unique_values <= 1:
            continue

        test_result = mannwhitneyu(
            first_values,
            second_values,
            alternative="two-sided",
            method="auto",
        )

        u_statistic = float(
            test_result.statistic
        )
        p_value = float(
            test_result.pvalue
        )

        effect_size = (
            calculate_rank_biserial_correlation(
                u_statistic=u_statistic,
                first_sample_size=len(
                    first_values
                ),
                second_sample_size=len(
                    second_values
                ),
            )
        )

        records.append(
            {
                "comparison": target_name,
                "feature": feature,
                "class_0": first_target,
                "class_1": second_target,
                "class_0_count": int(
                    len(first_values)
                ),
                "class_1_count": int(
                    len(second_values)
                ),
                "class_0_mean": float(
                    first_values.mean()
                ),
                "class_1_mean": float(
                    second_values.mean()
                ),
                "class_0_median": float(
                    first_values.median()
                ),
                "class_1_median": float(
                    second_values.median()
                ),
                "u_statistic": u_statistic,
                "p_value": p_value,
                "rank_biserial_correlation": (
                    effect_size
                ),
                "absolute_effect_size": abs(
                    effect_size
                ),
                "effect_size_category": (
                    classify_effect_size(
                        effect_size
                    )
                ),
                "significant_raw": (
                    p_value < alpha
                ),
            }
        )

    results = pd.DataFrame(records)

    if not results.empty:
        results["p_value_holm"] = holm_adjust(
            results["p_value"].tolist()
        )

        results["significant_holm"] = (
            results["p_value_holm"] < alpha
        )

        results = results.sort_values(
            by=[
                "p_value_holm",
                "absolute_effect_size",
            ],
            ascending=[
                True,
                False,
            ],
        ).reset_index(drop=True)

    return results


def build_spearman_correlation_matrix(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Build a Spearman correlation matrix for engineered numeric features.
    """

    selected_columns = [
        column
        for column in feature_columns
        if column in dataframe.columns
    ]

    if len(selected_columns) < 2:
        return pd.DataFrame()

    numeric_frame = (
        dataframe[selected_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    non_constant_columns = [
        column
        for column in numeric_frame.columns
        if numeric_frame[column].nunique(
            dropna=True
        ) > 1
    ]

    if len(non_constant_columns) < 2:
        return pd.DataFrame()

    return numeric_frame[
        non_constant_columns
    ].corr(method="spearman")


def identify_high_correlations(
    correlation_matrix: pd.DataFrame,
    *,
    threshold: float = (
        FEATURE_VALIDATION_HIGH_CORRELATION
    ),
) -> pd.DataFrame:
    """
    List unique feature pairs whose absolute correlation meets or exceeds
    the configured threshold.

    Returns an empty DataFrame with the expected columns when no qualifying
    feature pairs are found.
    """

    output_columns = [
        "feature_1",
        "feature_2",
        "correlation",
        "absolute_correlation",
    ]

    if correlation_matrix.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    records: list[dict[str, Any]] = []
    columns = correlation_matrix.columns.tolist()

    for first_index, first_feature in enumerate(
        columns
    ):
        for second_feature in columns[
            first_index + 1:
        ]:
            raw_correlation = correlation_matrix.loc[
                first_feature,
                second_feature,
            ]

            if pd.isna(raw_correlation):
                continue

            correlation_value = float(
                raw_correlation
            )

            absolute_correlation = abs(
                correlation_value
            )

            if absolute_correlation >= threshold:
                records.append(
                    {
                        "feature_1": first_feature,
                        "feature_2": second_feature,
                        "correlation": (
                            correlation_value
                        ),
                        "absolute_correlation": (
                            absolute_correlation
                        ),
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=output_columns
        )

    result = pd.DataFrame(
        records,
        columns=output_columns,
    )

    return result.sort_values(
        by="absolute_correlation",
        ascending=False,
    ).reset_index(drop=True)

def build_feature_recommendations(
    quality_report: pd.DataFrame,
    kruskal_results: pd.DataFrame,
    phishing_results: pd.DataFrame,
    source_results: pd.DataFrame,
    high_correlations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create preliminary feature-retention recommendations.

    These are evidence summaries, not automatic final feature selection.
    """

    features = sorted(
        set(quality_report["feature"].tolist())
    )

    highly_correlated_features: set[str] = set()

    if not high_correlations.empty:
        highly_correlated_features.update(
            high_correlations[
                "feature_1"
            ].tolist()
        )
        highly_correlated_features.update(
            high_correlations[
                "feature_2"
            ].tolist()
        )

    records: list[dict[str, Any]] = []

    for feature in features:
        quality_row = quality_report.loc[
            quality_report["feature"] == feature
        ]

        constant_feature = bool(
            quality_row.iloc[0][
                "constant_feature"
            ]
        )

        missing_percentage = float(
            quality_row.iloc[0][
                "missing_percentage"
            ]
        )

        kruskal_row = kruskal_results.loc[
            kruskal_results["feature"]
            == feature
        ]

        phishing_row = phishing_results.loc[
            phishing_results["feature"]
            == feature
        ]

        source_row = source_results.loc[
            source_results["feature"]
            == feature
        ]

        group_effect = (
            float(
                kruskal_row.iloc[0][
                    "epsilon_squared"
                ]
            )
            if not kruskal_row.empty
            else 0.0
        )

        phishing_effect = (
            float(
                phishing_row.iloc[0][
                    "absolute_effect_size"
                ]
            )
            if not phishing_row.empty
            else 0.0
        )

        source_effect = (
            float(
                source_row.iloc[0][
                    "absolute_effect_size"
                ]
            )
            if not source_row.empty
            else 0.0
        )

        phishing_significant = (
            bool(
                phishing_row.iloc[0][
                    "significant_holm"
                ]
            )
            if not phishing_row.empty
            else False
        )

        source_significant = (
            bool(
                source_row.iloc[0][
                    "significant_holm"
                ]
            )
            if not source_row.empty
            else False
        )

        if constant_feature:
            recommendation = "remove_constant"
            reason = (
                "The feature contains no useful variation."
            )

        elif missing_percentage > 50:
            recommendation = (
                "review_high_missingness"
            )
            reason = (
                "More than half of the values are missing."
            )

        elif (
            phishing_significant
            and phishing_effect >= 0.10
        ):
            recommendation = (
                "retain_phishing_candidate"
            )
            reason = (
                "The feature shows a corrected "
                "phishing-class difference and "
                "non-negligible effect."
            )

        elif (
            source_significant
            and source_effect >= 0.10
        ):
            recommendation = (
                "retain_source_comparison"
            )
            reason = (
                "The feature differentiates human "
                "and LLM-generated messages."
            )

        elif group_effect >= 0.01:
            recommendation = (
                "retain_group_analysis"
            )
            reason = (
                "The feature varies across the four "
                "experimental groups."
            )

        else:
            recommendation = (
                "review_low_evidence"
            )
            reason = (
                "Current univariate evidence is weak; "
                "retain only with theoretical support."
            )

        records.append(
            {
                "feature": feature,
                "recommendation": recommendation,
                "reason": reason,
                "constant_feature": constant_feature,
                "missing_percentage": (
                    missing_percentage
                ),
                "four_group_effect": group_effect,
                "phishing_effect": phishing_effect,
                "source_effect": source_effect,
                "high_correlation_flag": (
                    feature
                    in highly_correlated_features
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        by=[
            "recommendation",
            "phishing_effect",
            "source_effect",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)


def plot_feature_boxplot(
    dataframe: pd.DataFrame,
    *,
    feature: str,
    group_column: str = "experiment_group",
    output_directory: str | Path = (
        FEATURE_VALIDATION_FIGURES_DIR
    ),
) -> Path | None:
    """Create one boxplot for a feature across experiment groups."""

    if feature not in dataframe.columns:
        raise FeatureValidationError(
            f"Feature column not found: {feature}"
        )

    if group_column not in dataframe.columns:
        raise FeatureValidationError(
            f"Group column not found: {group_column}"
        )

    groups = sorted(
        dataframe[group_column]
        .dropna()
        .astype(str)
        .unique()
    )

    boxplot_values: list[np.ndarray] = []
    valid_group_names: list[str] = []

    for group in groups:
        values = pd.to_numeric(
            dataframe.loc[
                dataframe[group_column] == group,
                feature,
            ],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if values.empty:
            continue

        boxplot_values.append(
            values.to_numpy(dtype=float)
        )
        valid_group_names.append(group)

    if (
        len(boxplot_values) < 2
        or pd.concat(
            [
                pd.Series(values)
                for values in boxplot_values
            ],
            ignore_index=True,
        ).nunique() <= 1
    ):
        return None

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        output_path
        / (
            f"{sanitize_filename(feature)}"
            "_group_boxplot.png"
        )
    )

    plt.figure(figsize=(12, 7))
    plt.boxplot(
        boxplot_values,
        tick_labels=valid_group_names,
        showfliers=False,
    )
    plt.xlabel("Experimental group")
    plt.ylabel(feature)
    plt.title(
        f"{feature} across AI-email groups"
    )
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return figure_path


def plot_group_mean_comparison(
    descriptive_statistics: pd.DataFrame,
    *,
    feature: str,
    output_directory: str | Path = (
        FEATURE_VALIDATION_FIGURES_DIR
    ),
) -> Path | None:
    """Create a group mean bar chart for one feature."""

    selected = descriptive_statistics.loc[
        descriptive_statistics["feature"]
        == feature
    ].copy()

    if selected.empty:
        return None

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        output_path
        / (
            f"{sanitize_filename(feature)}"
            "_group_means.png"
        )
    )

    selected = selected.sort_values("group")

    plt.figure(figsize=(11, 7))
    plt.bar(
        selected["group"],
        selected["mean"],
    )
    plt.xlabel("Experimental group")
    plt.ylabel(f"Mean {feature}")
    plt.title(
        f"Mean {feature} across AI-email groups"
    )
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return figure_path


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    *,
    output_directory: str | Path = (
        FEATURE_VALIDATION_FIGURES_DIR
    ),
) -> Path | None:
    """Create a Spearman-correlation heatmap."""

    if correlation_matrix.empty:
        return None

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = (
        output_path
        / "email_feature_spearman_heatmap.png"
    )

    figure_size = max(
        10,
        min(
            24,
            len(correlation_matrix.columns)
            * 0.55,
        ),
    )

    plt.figure(
        figsize=(figure_size, figure_size)
    )

    image = plt.imshow(
        correlation_matrix.to_numpy(),
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Spearman correlation",
    )

    positions = range(
        len(correlation_matrix.columns)
    )

    plt.xticks(
        positions,
        correlation_matrix.columns,
        rotation=90,
    )
    plt.yticks(
        positions,
        correlation_matrix.index,
    )
    plt.title(
        "Spearman correlation among email features"
    )
    plt.tight_layout()
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return figure_path


def save_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save JSON-compatible information using UTF-8."""

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    return path


def run_ai_email_feature_validation(
    dataframe: pd.DataFrame,
    *,
    alpha: float = FEATURE_VALIDATION_ALPHA,
    correlation_threshold: float = (
        FEATURE_VALIDATION_HIGH_CORRELATION
    ),
    maximum_plotted_features: int = 12,
    reports_directory: str | Path = (
        FEATURE_VALIDATION_REPORTS_DIR
    ),
    statistics_directory: str | Path = (
        FEATURE_VALIDATION_STATISTICS_DIR
    ),
    figures_directory: str | Path = (
        FEATURE_VALIDATION_FIGURES_DIR
    ),
) -> dict[str, Any]:
    """
    Run the complete AI-email feature-validation workflow.

    This function performs descriptive and inferential analysis only.
    It does not train a machine-learning model.
    """

    validation_summary = (
        validate_ai_email_feature_dataset(
            dataframe
        )
    )

    feature_columns = (
        select_numeric_feature_columns(
            dataframe
        )
    )

    if not feature_columns:
        raise FeatureValidationError(
            "No numeric engineered features were found."
        )

    quality_report = (
        build_feature_quality_report(
            dataframe,
            feature_columns,
        )
    )

    usable_features = quality_report.loc[
        ~quality_report["constant_feature"],
        "feature",
    ].tolist()

    descriptive_statistics = (
        build_group_descriptive_statistics(
            dataframe,
            usable_features,
        )
    )

    kruskal_results = (
        run_kruskal_wallis_tests(
            dataframe,
            usable_features,
            alpha=alpha,
        )
    )

    pairwise_results = (
        run_pairwise_mann_whitney_tests(
            dataframe,
            usable_features,
            alpha=alpha,
        )
    )

    phishing_results = (
        run_binary_target_comparison(
            dataframe,
            usable_features,
            target_column="phishing_label",
            target_name=(
                "legitimate_vs_phishing"
            ),
            alpha=alpha,
        )
    )

    source_results = (
        run_binary_target_comparison(
            dataframe,
            usable_features,
            target_column="source_label",
            target_name="human_vs_llm",
            alpha=alpha,
        )
    )

    correlation_matrix = (
        build_spearman_correlation_matrix(
            dataframe,
            usable_features,
        )
    )

    high_correlations = (
        identify_high_correlations(
            correlation_matrix,
            threshold=correlation_threshold,
        )
    )

    recommendations = (
        build_feature_recommendations(
            quality_report=quality_report,
            kruskal_results=kruskal_results,
            phishing_results=phishing_results,
            source_results=source_results,
            high_correlations=high_correlations,
        )
    )

    reports_path = Path(reports_directory)
    statistics_path = Path(
        statistics_directory
    )
    figures_path = Path(figures_directory)

    reports_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    statistics_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {
        "validation_summary": (
            reports_path
            / "ai_email_feature_validation_summary.json"
        ),
        "quality_report": (
            reports_path
            / "ai_email_feature_quality_report.csv"
        ),
        "recommendations": (
            reports_path
            / "ai_email_feature_recommendations.csv"
        ),
        "descriptive_statistics": (
            statistics_path
            / "ai_email_group_descriptive_statistics.csv"
        ),
        "kruskal_results": (
            statistics_path
            / "ai_email_kruskal_wallis.csv"
        ),
        "pairwise_results": (
            statistics_path
            / "ai_email_pairwise_mann_whitney.csv"
        ),
        "phishing_results": (
            statistics_path
            / "ai_email_phishing_feature_comparison.csv"
        ),
        "source_results": (
            statistics_path
            / "ai_email_source_feature_comparison.csv"
        ),
        "correlation_matrix": (
            statistics_path
            / "ai_email_spearman_correlation.csv"
        ),
        "high_correlations": (
            statistics_path
            / "ai_email_high_correlations.csv"
        ),
    }

    quality_report.to_csv(
        output_paths["quality_report"],
        index=False,
        encoding="utf-8",
    )

    recommendations.to_csv(
        output_paths["recommendations"],
        index=False,
        encoding="utf-8",
    )

    descriptive_statistics.to_csv(
        output_paths[
            "descriptive_statistics"
        ],
        index=False,
        encoding="utf-8",
    )

    kruskal_results.to_csv(
        output_paths["kruskal_results"],
        index=False,
        encoding="utf-8",
    )

    pairwise_results.to_csv(
        output_paths["pairwise_results"],
        index=False,
        encoding="utf-8",
    )

    phishing_results.to_csv(
        output_paths["phishing_results"],
        index=False,
        encoding="utf-8",
    )

    source_results.to_csv(
        output_paths["source_results"],
        index=False,
        encoding="utf-8",
    )

    correlation_matrix.to_csv(
        output_paths["correlation_matrix"],
        encoding="utf-8",
    )

    high_correlations.to_csv(
        output_paths["high_correlations"],
        index=False,
        encoding="utf-8",
    )

    top_features = (
        kruskal_results.loc[
            kruskal_results["test_status"]
            == "completed"
        ]
        .sort_values(
            "epsilon_squared",
            ascending=False,
        )
        .head(maximum_plotted_features)[
            "feature"
        ]
        .tolist()
    )

    generated_figures: list[str] = []

    for feature in top_features:
        boxplot_path = plot_feature_boxplot(
            dataframe,
            feature=feature,
            output_directory=figures_path,
        )

        if boxplot_path is not None:
            generated_figures.append(
                str(boxplot_path)
            )

        mean_plot_path = (
            plot_group_mean_comparison(
                descriptive_statistics,
                feature=feature,
                output_directory=figures_path,
            )
        )

        if mean_plot_path is not None:
            generated_figures.append(
                str(mean_plot_path)
            )

    correlation_plot_path = (
        plot_correlation_heatmap(
            correlation_matrix,
            output_directory=figures_path,
        )
    )

    if correlation_plot_path is not None:
        generated_figures.append(
            str(correlation_plot_path)
        )

    completed_kruskal = kruskal_results.loc[
        kruskal_results["test_status"]
        == "completed"
    ]

    summary = {
        **validation_summary,
        "alpha": alpha,
        "correlation_threshold": (
            correlation_threshold
        ),
        "numeric_feature_count": len(
            feature_columns
        ),
        "usable_feature_count": len(
            usable_features
        ),
        "constant_feature_count": int(
            quality_report[
                "constant_feature"
            ].sum()
        ),
        "significant_four_group_features": int(
            completed_kruskal[
                "significant_holm"
            ].sum()
        ),
        "significant_phishing_features": int(
            phishing_results[
                "significant_holm"
            ].sum()
        ),
        "significant_source_features": int(
            source_results[
                "significant_holm"
            ].sum()
        ),
        "high_correlation_pair_count": int(
            len(high_correlations)
        ),
        "top_four_group_features": (
            completed_kruskal.sort_values(
                "epsilon_squared",
                ascending=False,
            )
            .head(10)[
                [
                    "feature",
                    "epsilon_squared",
                    "effect_size_category",
                    "p_value_holm",
                ]
            ]
            .to_dict(orient="records")
        ),
        "generated_figures": (
            generated_figures
        ),
        "output_paths": {
            name: str(path)
            for name, path in (
                output_paths.items()
            )
        },
        "interpretation_warning": (
            "Statistical association does not establish "
            "causation, practical importance or human "
            "susceptibility. Results require theoretical "
            "and methodological interpretation."
        ),
    }

    save_json(
        summary,
        output_paths[
            "validation_summary"
        ],
    )

    LOGGER.info(
        "Feature validation completed for %s rows "
        "and %s usable features.",
        len(dataframe),
        len(usable_features),
    )

    return summary