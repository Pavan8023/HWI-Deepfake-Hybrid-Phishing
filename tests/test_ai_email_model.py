from __future__ import annotations

import pandas as pd
import pytest

from src.ai_email_model import (
    AIEmailModelError,
    prepare_ai_email_model_frame,
    select_ai_email_numeric_features,
    stratified_ai_email_split,
    validate_ai_email_dataset,
    validate_ai_email_split,
)


def create_ai_email_test_dataset(
    rows: int = 100,
) -> pd.DataFrame:
    """Create a balanced engineered AI-email dataset."""

    return pd.DataFrame(
        {
            "word_count": [
                30 + index
                for index in range(rows)
            ],
            "urgency_keyword_count": [
                index % 4
                for index in range(rows)
            ],
            "fear_keyword_count": [
                index % 3
                for index in range(rows)
            ],
            "trust_keyword_count": [
                index % 2
                for index in range(rows)
            ],
            "phishing_label": [
                index % 2
                for index in range(rows)
            ],
            "source_label": [
                (index // 2) % 2
                for index in range(rows)
            ],
            "original_label": [
                index % 2
                for index in range(rows)
            ],
            "source_type": [
                (
                    "human"
                    if index % 4 < 2
                    else "llm"
                )
                for index in range(rows)
            ],
            "email_class": [
                (
                    "legitimate"
                    if index % 2 == 0
                    else "phishing"
                )
                for index in range(rows)
            ],
            "experiment_group": [
                (
                    "human_legitimate"
                    if index % 4 == 0
                    else "human_phishing"
                    if index % 4 == 1
                    else "llm_legitimate"
                    if index % 4 == 2
                    else "llm_phishing"
                )
                for index in range(rows)
            ],
            "template_group_id": [
                f"template_{index // 2}"
                for index in range(rows)
            ],
        }
    )


def test_validate_ai_email_dataset() -> None:
    dataframe = create_ai_email_test_dataset()

    validate_ai_email_dataset(
        dataframe
    )


def test_validate_ai_email_dataset_rejects_missing_target() -> None:
    dataframe = (
        create_ai_email_test_dataset()
        .drop(columns=["phishing_label"])
    )

    with pytest.raises(AIEmailModelError):
        validate_ai_email_dataset(
            dataframe
        )


def test_select_ai_email_numeric_features_excludes_labels() -> None:
    dataframe = create_ai_email_test_dataset()

    features = (
        select_ai_email_numeric_features(
            dataframe
        )
    )

    assert "word_count" in features
    assert "urgency_keyword_count" in features

    assert "phishing_label" not in features
    assert "source_label" not in features
    assert "original_label" not in features


def test_prepare_ai_email_model_frame() -> None:
    dataframe = create_ai_email_test_dataset()

    features, target, metadata = (
        prepare_ai_email_model_frame(
            dataframe
        )
    )

    assert len(features) == len(dataframe)
    assert len(target) == len(dataframe)
    assert len(metadata) == len(dataframe)

    assert "phishing_label" not in features.columns
    assert "source_label" not in features.columns
    assert "experiment_group" in metadata.columns
    assert "template_group_id" in metadata.columns
    assert "template_group_id" not in features.columns


def test_stratified_ai_email_split() -> None:
    dataframe = create_ai_email_test_dataset(
        rows=100
    )

    features, target, metadata = (
        prepare_ai_email_model_frame(
            dataframe
        )
    )

    split = stratified_ai_email_split(
        features,
        target,
        metadata,
        test_size=0.20,
        random_state=42,
    )

    assert len(split.x_train) == 80
    assert len(split.x_test) == 20

    assert split.y_train.mean() == (
        pytest.approx(0.5)
    )

    assert split.y_test.mean() == (
        pytest.approx(0.5)
    )

    shared_template_groups = set(
        split.metadata_train["template_group_id"]
    ).intersection(
        set(split.metadata_test["template_group_id"])
    )

    assert shared_template_groups == set()


def test_validate_ai_email_split() -> None:
    dataframe = create_ai_email_test_dataset()

    features, target, metadata = (
        prepare_ai_email_model_frame(
            dataframe
        )
    )

    split = stratified_ai_email_split(
        features,
        target,
        metadata,
    )

    summary = validate_ai_email_split(
        split
    )

    assert summary["feature_count"] == 4
    assert summary["missing_train_values"] == 0
    assert summary["missing_test_values"] == 0
    assert summary["split_strategy"] == "group_shuffle_split"
    assert summary["shared_template_groups"] == 0
    assert summary["leakage_columns"] == []
