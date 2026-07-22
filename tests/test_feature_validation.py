from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_validation import (
    FeatureValidationError,
    build_feature_quality_report,
    build_group_descriptive_statistics,
    build_spearman_correlation_matrix,
    calculate_epsilon_squared,
    calculate_rank_biserial_correlation,
    classify_effect_size,
    holm_adjust,
    identify_high_correlations,
    run_ai_email_feature_validation,
    run_binary_target_comparison,
    run_kruskal_wallis_tests,
    run_pairwise_mann_whitney_tests,
    select_numeric_feature_columns,
    validate_ai_email_feature_dataset,
)


def create_test_feature_dataset() -> pd.DataFrame:
    """Create a balanced synthetic four-group feature dataset."""

    records: list[dict[str, object]] = []

    group_definitions = [
        (
            "human_legitimate",
            "human",
            "legitimate",
            0,
            0,
            10,
        ),
        (
            "human_phishing",
            "human",
            "phishing",
            0,
            1,
            20,
        ),
        (
            "llm_legitimate",
            "llm",
            "legitimate",
            1,
            0,
            30,
        ),
        (
            "llm_phishing",
            "llm",
            "phishing",
            1,
            1,
            40,
        ),
    ]

    for (
        group,
        source_type,
        email_class,
        source_label,
        phishing_label,
        baseline,
    ) in group_definitions:
        for offset in range(10):
            records.append(
                {
                    "word_count": baseline + offset,
                    "url_count": phishing_label + (
                        offset % 2
                    ),
                    "constant_feature": 1,
                    "source_type": source_type,
                    "email_class": email_class,
                    "source_label": source_label,
                    "phishing_label": phishing_label,
                    "original_label": phishing_label,
                    "experiment_group": group,
                }
            )

    return pd.DataFrame(records)


def test_validate_ai_email_feature_dataset() -> None:
    dataframe = create_test_feature_dataset()

    summary = validate_ai_email_feature_dataset(
        dataframe
    )

    assert summary["rows"] == 40
    assert summary["missing_expected_groups"] == []
    assert summary["unexpected_groups"] == []


def test_validate_ai_email_feature_dataset_rejects_missing_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "word_count": [1, 2],
        }
    )

    with pytest.raises(FeatureValidationError):
        validate_ai_email_feature_dataset(
            dataframe
        )


def test_select_numeric_feature_columns_excludes_labels() -> None:
    dataframe = create_test_feature_dataset()

    features = select_numeric_feature_columns(
        dataframe
    )

    assert "word_count" in features
    assert "url_count" in features
    assert "source_label" not in features
    assert "phishing_label" not in features
    assert "original_label" not in features


def test_build_feature_quality_report_detects_constant_feature() -> None:
    dataframe = create_test_feature_dataset()

    report = build_feature_quality_report(
        dataframe,
        [
            "word_count",
            "constant_feature",
        ],
    )

    constant_value = report.loc[
        report["feature"]
        == "constant_feature",
        "constant_feature",
    ].iloc[0]

    assert bool(constant_value) is True


def test_build_group_descriptive_statistics() -> None:
    dataframe = create_test_feature_dataset()

    report = build_group_descriptive_statistics(
        dataframe,
        ["word_count"],
    )

    assert len(report) == 4
    assert set(report["group"]) == {
        "human_legitimate",
        "human_phishing",
        "llm_legitimate",
        "llm_phishing",
    }


def test_calculate_epsilon_squared() -> None:
    value = calculate_epsilon_squared(
        h_statistic=20,
        sample_size=100,
        group_count=4,
    )

    assert value == pytest.approx(
        17 / 96
    )


def test_classify_effect_size() -> None:
    assert classify_effect_size(0.001) == (
        "negligible"
    )
    assert classify_effect_size(0.03) == "small"
    assert classify_effect_size(0.10) == "medium"
    assert classify_effect_size(0.20) == "large"


def test_holm_adjust() -> None:
    adjusted = holm_adjust(
        [0.01, 0.04, 0.20]
    )

    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.08)
    assert adjusted[2] == pytest.approx(0.20)


def test_kruskal_wallis_detects_group_difference() -> None:
    dataframe = create_test_feature_dataset()

    results = run_kruskal_wallis_tests(
        dataframe,
        ["word_count"],
    )

    assert len(results) == 1

    epsilon_squared = results.at[
        0,
        "epsilon_squared",
    ]

    assert isinstance(
        epsilon_squared,
        (int, float, np.integer, np.floating),
    )

    assert float(epsilon_squared) > 0

    significant_holm = results.at[
        0,
        "significant_holm",
    ]

    assert bool(significant_holm) is True
    
def test_pairwise_mann_whitney_results() -> None:
    dataframe = create_test_feature_dataset()

    results = (
        run_pairwise_mann_whitney_tests(
            dataframe,
            ["word_count"],
        )
    )

    assert len(results) == 6
    assert "p_value_holm" in results.columns
    assert (
        "rank_biserial_correlation"
        in results.columns
    )


def test_rank_biserial_correlation() -> None:
    effect = (
        calculate_rank_biserial_correlation(
            u_statistic=100,
            first_sample_size=10,
            second_sample_size=10,
        )
    )

    assert effect == pytest.approx(1.0)


def test_binary_target_comparison() -> None:
    dataframe = create_test_feature_dataset()

    results = run_binary_target_comparison(
        dataframe,
        ["word_count"],
        target_column="phishing_label",
        target_name="legitimate_vs_phishing",
    )

    assert len(results) == 1
    assert results.loc[
        0,
        "comparison",
    ] == "legitimate_vs_phishing"


def test_spearman_correlation_matrix() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
            "c": [4, 3, 2, 1],
        }
    )

    correlation = (
        build_spearman_correlation_matrix(
            dataframe,
            ["a", "b", "c"],
        )
    )

    assert correlation.loc[
        "a",
        "b",
    ] == pytest.approx(1.0)

    assert correlation.loc[
        "a",
        "c",
    ] == pytest.approx(-1.0)


def test_identify_high_correlations() -> None:
    correlation = pd.DataFrame(
        [
            [1.0, 0.95, 0.10],
            [0.95, 1.0, 0.20],
            [0.10, 0.20, 1.0],
        ],
        columns=["a", "b", "c"],
        index=["a", "b", "c"],
    )

    result = identify_high_correlations(
        correlation,
        threshold=0.90,
    )

    assert len(result) == 1
    assert result.loc[0, "feature_1"] == "a"
    assert result.loc[0, "feature_2"] == "b"


def test_run_complete_feature_validation(
    tmp_path: Path,
) -> None:
    dataframe = create_test_feature_dataset()

    reports_directory = tmp_path / "reports"
    statistics_directory = tmp_path / "statistics"
    figures_directory = tmp_path / "figures"

    summary = run_ai_email_feature_validation(
        dataframe,
        reports_directory=reports_directory,
        statistics_directory=(
            statistics_directory
        ),
        figures_directory=figures_directory,
        maximum_plotted_features=2,
    )

    assert summary["rows"] == 40
    assert summary["usable_feature_count"] == 2

    assert (
        reports_directory
        / (
            "ai_email_feature_validation_"
            "summary.json"
        )
    ).exists()

    assert (
        reports_directory
        / "ai_email_feature_quality_report.csv"
    ).exists()

    assert (
        statistics_directory
        / "ai_email_kruskal_wallis.csv"
    ).exists()