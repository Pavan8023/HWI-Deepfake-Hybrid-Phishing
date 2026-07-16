from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda import (
    EDAError,
    build_categorical_summary,
    build_column_profile,
    build_duplicate_summary,
    build_numeric_correlation_matrix,
    build_numeric_summary,
    build_text_summary,
    detect_column_groups,
    ensure_dataframe,
    generate_eda_summary,
    load_eda_dataframe,
    run_dataset_eda,
    sanitize_filename,
)


def test_sanitize_filename() -> None:
    """Filename text should be converted into a safe lowercase value."""

    assert (
        sanitize_filename("Awareness Dataset.csv")
        == "awareness_dataset_csv"
    )


def test_ensure_dataframe_returns_existing_dataframe() -> None:
    """A DataFrame should be returned unchanged as a DataFrame."""

    dataframe = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    result = ensure_dataframe(dataframe)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert result["value"].tolist() == [1, 2, 3]


def test_ensure_dataframe_combines_chunks() -> None:
    """A DataFrame chunk iterator should be combined safely."""

    chunks = [
        pd.DataFrame({"value": [1, 2]}),
        pd.DataFrame({"value": [3, 4]}),
    ]

    result = ensure_dataframe(iter(chunks))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert result["value"].tolist() == [1, 2, 3, 4]


def test_ensure_dataframe_rejects_invalid_chunk() -> None:
    """Invalid objects inside a chunk iterator should raise an error."""

    invalid_chunks = [
        pd.DataFrame({"value": [1]}),
        "not-a-dataframe",
    ]

    with pytest.raises(EDAError):
        ensure_dataframe(iter(invalid_chunks))


def test_detect_column_groups() -> None:
    """Columns should be grouped into broad analytical categories."""

    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "category": ["low", "medium", "high"],
            "email_text": [
                (
                    "This is a considerably long email message "
                    "used for testing the text-column detector."
                ),
                (
                    "Another considerably long email message "
                    "used for testing the text-column detector."
                ),
                (
                    "A third considerably long email message "
                    "used for testing the text-column detector."
                ),
            ],
            "record_id": ["A1", "A2", "A3"],
            "constant": [1, 1, 1],
        }
    )

    groups = detect_column_groups(dataframe)

    assert "age" in groups["numeric_columns"]
    assert "category" in groups["categorical_columns"]
    assert "email_text" in groups["text_columns"]
    assert "record_id" in groups["possible_identifier_columns"]
    assert "constant" in groups["constant_columns"]


def test_build_column_profile() -> None:
    """The column profile should report missing and unique values."""

    dataframe = pd.DataFrame(
        {
            "value": [1, 2, np.nan],
            "label": ["safe", "unsafe", "safe"],
        }
    )

    profile = build_column_profile(dataframe)

    assert len(profile) == 2
    assert "missing_percentage" in profile.columns
    assert "unique_count" in profile.columns
    assert "sample_values" in profile.columns

    value_profile = profile.loc[
        profile["column"] == "value"
    ].iloc[0]

    assert value_profile["missing_count"] == 1


def test_build_numeric_summary() -> None:
    """Numeric statistics should include skewness and kurtosis."""

    dataframe = pd.DataFrame(
        {
            "score": [10, 20, 30, 40],
            "label": ["a", "b", "a", "b"],
        }
    )

    summary = build_numeric_summary(dataframe)

    assert not summary.empty
    assert "score" in summary["column"].tolist()
    assert "skewness" in summary.columns
    assert "kurtosis" in summary.columns
    assert "zero_count" in summary.columns


def test_build_numeric_summary_with_no_numeric_columns() -> None:
    """A dataset without numeric columns should return an empty summary."""

    dataframe = pd.DataFrame(
        {
            "label": ["a", "b", "c"],
        }
    )

    summary = build_numeric_summary(dataframe)

    assert summary.empty


def test_build_categorical_summary() -> None:
    """Categorical counts should be calculated correctly."""

    dataframe = pd.DataFrame(
        {
            "label": [
                "phishing",
                "legitimate",
                "phishing",
            ],
        }
    )

    summary = build_categorical_summary(dataframe)

    assert not summary.empty
    assert set(summary["value"]) == {
        "phishing",
        "legitimate",
    }

    phishing_row = summary.loc[
        summary["value"] == "phishing"
    ].iloc[0]

    assert phishing_row["count"] == 2


def test_build_text_summary() -> None:
    """Text statistics should report character and word lengths."""

    dataframe = pd.DataFrame(
        {
            "email_text": [
                "This is the first email.",
                "This is a longer second email message.",
            ]
        }
    )

    summary = build_text_summary(
        dataframe,
        text_columns=["email_text"],
    )

    assert not summary.empty
    assert summary.loc[0, "column"] == "email_text"

    average_word_count = summary.loc[0, "average_word_count"]
    maximum_character_length = summary.loc[
        0,
        "maximum_character_length",
    ]

    assert isinstance(
        average_word_count,
        (int, float, np.integer, np.floating),
    )
    assert isinstance(
        maximum_character_length,
        (int, float, np.integer, np.floating),
    )
    assert float(average_word_count) > 0
    assert float(maximum_character_length) > 0


def test_build_duplicate_summary() -> None:
    """Duplicate-row reporting should count repeated rows."""

    dataframe = pd.DataFrame(
        {
            "value": [1, 1, 2],
            "label": ["a", "a", "b"],
        }
    )

    summary = build_duplicate_summary(dataframe)

    assert (
        summary[
            "exact_duplicate_rows_after_first_occurrence"
        ]
        == 1
    )

    assert (
        summary[
            "rows_participating_in_duplicate_groups"
        ]
        == 2
    )


def test_numeric_correlation_matrix() -> None:
    """Perfectly related columns should have correlation equal to one."""

    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],
            "label": ["x", "y", "x", "y"],
        }
    )

    correlation = build_numeric_correlation_matrix(dataframe)

    assert not correlation.empty
    assert correlation.loc["a", "b"] == pytest.approx(1.0)


def test_numeric_correlation_with_one_numeric_column() -> None:
    """Correlation should be empty when fewer than two usable columns exist."""

    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "label": ["x", "y", "z"],
        }
    )

    correlation = build_numeric_correlation_matrix(dataframe)

    assert correlation.empty


def test_generate_eda_summary(tmp_path: Path) -> None:
    """The EDA summary should include the expected structural fields."""

    dataset_path = tmp_path / "sample.csv"

    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "label": ["a", "b", "a"],
        }
    )

    dataframe.to_csv(dataset_path, index=False)

    summary = generate_eda_summary(
        dataframe=dataframe,
        dataset_name="sample",
        dataset_path=dataset_path,
        sampled=False,
        loading_metadata={
            "sampling_method": "full_dataset",
        },
    )

    assert summary["analysed_rows"] == 3
    assert summary["columns"] == 2
    assert summary["sampled"] is False
    assert summary["loading_metadata"][
        "sampling_method"
    ] == "full_dataset"


def test_load_eda_dataframe_small_csv(
    tmp_path: Path,
) -> None:
    """A small CSV should be loaded completely."""

    dataset_path = tmp_path / "sample.csv"

    dataframe = pd.DataFrame(
        {
            "feature": range(10),
            "label": ["a"] * 10,
        }
    )

    dataframe.to_csv(dataset_path, index=False)

    loaded, sampled, metadata = load_eda_dataframe(
        dataset_path,
        sample_rows=100,
    )

    assert len(loaded) == 10
    assert sampled is False
    assert metadata["sampling_method"] == "full_dataset"


def test_load_eda_dataframe_large_csv_sample(
    tmp_path: Path,
) -> None:
    """A larger CSV should be sampled reproducibly."""

    dataset_path = tmp_path / "large.csv"

    dataframe = pd.DataFrame(
        {
            "feature": range(200),
            "label": ["a", "b"] * 100,
        }
    )

    dataframe.to_csv(dataset_path, index=False)

    loaded, sampled, metadata = load_eda_dataframe(
        dataset_path,
        sample_rows=50,
    )

    assert len(loaded) == 50
    assert sampled is True
    assert (
        metadata["sampling_method"]
        == "chunked_reproducible_sample"
    )


def test_load_eda_dataframe_missing_file(
    tmp_path: Path,
) -> None:
    """A missing dataset should raise FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        load_eda_dataframe(
            tmp_path / "missing.csv"
        )


def test_run_dataset_eda_creates_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running EDA should create reports and return result metadata."""

    dataset_path = tmp_path / "sample.csv"

    dataframe = pd.DataFrame(
        {
            "score": [10, 20, 30, 40],
            "label": [
                "low",
                "medium",
                "high",
                "medium",
            ],
        }
    )

    dataframe.to_csv(dataset_path, index=False)

    reports_directory = tmp_path / "reports"
    statistics_directory = tmp_path / "statistics"
    figures_directory = tmp_path / "figures"

    monkeypatch.setattr(
        "src.eda.EDA_REPORTS_DIR",
        reports_directory,
    )
    monkeypatch.setattr(
        "src.eda.EDA_STATISTICS_DIR",
        statistics_directory,
    )
    monkeypatch.setattr(
        "src.eda.EDA_FIGURES_DIR",
        figures_directory,
    )

    result = run_dataset_eda(
        dataset_path=dataset_path,
        dataset_name="sample_dataset",
        sample_rows=100,
    )

    assert result["rows_analysed"] == 4
    assert result["columns"] == 2
    assert result["sampled"] is False

    assert (
        reports_directory
        / "sample_dataset_column_profile.csv"
    ).exists()

    assert (
        reports_directory
        / "sample_dataset_eda_summary.json"
    ).exists()