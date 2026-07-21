from __future__ import annotations

import pandas as pd
import pytest
import numpy as np


from src.feature_engineering import (
    FeatureEngineeringError,
    add_experimental_labels,
    build_ai_email_feature_dataset,
    build_common_email_text,
    calculate_character_entropy,
    calculate_lexical_diversity,
    calculate_uppercase_ratio,
    combine_subject_and_body,
    count_keyword_occurrences,
    engineer_email_features,
    extract_email_feature_frame,
    extract_email_text_features,
    normalise_text,
    tokenise_words,
    validate_feature_frame,
)


def test_normalise_text_handles_missing_values() -> None:
    assert normalise_text(None) == ""
    assert normalise_text(float("nan")) == ""
    assert normalise_text("hello") == "hello"


def test_tokenise_words() -> None:
    words = tokenise_words(
        "Verify your account immediately."
    )

    assert words == [
        "verify",
        "your",
        "account",
        "immediately",
    ]


def test_count_keyword_occurrences() -> None:
    count = count_keyword_occurrences(
        "Act now. Immediate action is required now.",
        {"now", "immediate", "action required"},
    )

    assert count == 3


def test_calculate_uppercase_ratio() -> None:
    ratio = calculate_uppercase_ratio(
        "ABcd!"
    )

    assert ratio == pytest.approx(0.5)


def test_calculate_lexical_diversity() -> None:
    diversity = calculate_lexical_diversity(
        ["verify", "account", "verify"]
    )

    assert diversity == pytest.approx(2 / 3)


def test_character_entropy_empty_text() -> None:
    assert calculate_character_entropy("") == 0.0


def test_extract_email_text_features() -> None:
    text = (
        "Dear User, urgent action required! "
        "Please verify your account now at "
        "https://example.com. Sincerely, Security Team."
    )

    features = extract_email_text_features(text)

    assert features["character_count"] > 0
    assert features["word_count"] > 0
    assert features["sentence_count"] > 0
    assert features["url_count"] == 1
    assert features["exclamation_count"] == 1
    assert features["urgency_keyword_count"] >= 2
    assert features["credential_keyword_count"] >= 1
    assert features["authority_keyword_count"] >= 1
    assert features["call_to_action_count"] >= 1
    assert features["greeting_present"] == 1
    assert features["signoff_present"] == 1
    assert features["suspicious_link_indicator"] == 1


def test_combine_subject_and_body() -> None:
    result = combine_subject_and_body(
        "Urgent account update",
        "Please verify your details.",
    )

    assert result == (
        "Urgent account update\n"
        "Please verify your details."
    )


def test_build_common_email_text_from_text_column() -> None:
    dataframe = pd.DataFrame(
        {
            "text": [
                "First email",
                "Second email",
            ]
        }
    )

    result = build_common_email_text(
        dataframe,
        text_column="text",
    )

    assert result.tolist() == [
        "First email",
        "Second email",
    ]


def test_build_common_email_text_from_subject_and_body() -> None:
    dataframe = pd.DataFrame(
        {
            "subject": [
                "Subject one",
                "Subject two",
            ],
            "body": [
                "Body one",
                "Body two",
            ],
        }
    )

    result = build_common_email_text(
        dataframe,
        subject_column="subject",
        body_column="body",
    )

    assert result.tolist() == [
        "Subject one\nBody one",
        "Subject two\nBody two",
    ]


def test_build_common_email_text_rejects_missing_column() -> None:
    dataframe = pd.DataFrame(
        {
            "body": ["Example"],
        }
    )

    with pytest.raises(FeatureEngineeringError):
        build_common_email_text(
            dataframe,
            text_column="text",
        )


def test_extract_email_feature_frame() -> None:
    texts = pd.Series(
        [
            "Hello world.",
            "Urgent: verify account now!",
        ]
    )

    feature_frame = extract_email_feature_frame(
        texts
    )

    assert len(feature_frame) == 2
    assert "word_count" in feature_frame.columns
    assert "urgency_keyword_count" in feature_frame.columns


def test_engineer_email_features() -> None:
    dataframe = pd.DataFrame(
        {
            "subject": [
                "Account notice",
            ],
            "body": [
                "Please verify your account now.",
            ],
        }
    )

    result = engineer_email_features(
        dataframe,
        subject_column="subject",
        body_column="body",
        retain_original_text=True,
    )

    assert len(result) == 1
    assert "combined_text" in result.columns

    word_count_value = result.at[0, "word_count"]

    assert isinstance(
        word_count_value,
        (int, float, np.integer, np.floating),
    )

    word_count = float(word_count_value)

    assert word_count > 0


def test_engineer_email_features_rejects_empty_dataframe() -> None:
    with pytest.raises(FeatureEngineeringError):
        engineer_email_features(
            pd.DataFrame(),
            text_column="text",
        )


def test_add_experimental_labels() -> None:
    feature_frame = pd.DataFrame(
        {
            "word_count": [10, 20],
        }
    )

    result = add_experimental_labels(
        feature_frame,
        source_type="llm",
        email_class="phishing",
    )

    assert result["source_type"].tolist() == [
        "llm",
        "llm",
    ]
    assert result["email_class"].tolist() == [
        "phishing",
        "phishing",
    ]
    assert result["source_label"].tolist() == [
        1,
        1,
    ]
    assert result["phishing_label"].tolist() == [
        1,
        1,
    ]


def test_add_experimental_labels_rejects_invalid_source() -> None:
    feature_frame = pd.DataFrame(
        {
            "word_count": [10],
        }
    )

    with pytest.raises(ValueError):
        add_experimental_labels(
            feature_frame,
            source_type="robot",
            email_class="phishing",
        )


def test_build_ai_email_feature_dataset() -> None:
    human_legitimate = pd.DataFrame(
        {
            "subject": ["Meeting update"],
            "body": ["The meeting starts at ten."],
            "label": [0],
        }
    )

    human_phishing = pd.DataFrame(
        {
            "subject": ["Urgent account warning"],
            "body": ["Verify your account now."],
            "label": [1],
        }
    )

    llm_legitimate = pd.DataFrame(
        {
            "text": ["Thank you for attending the meeting."],
            "label": [1],
        }
    )

    llm_phishing = pd.DataFrame(
        {
            "text": ["Urgent: click here to verify your password."],
            "label": [1],
        }
    )

    result = build_ai_email_feature_dataset(
        human_legitimate=human_legitimate,
        human_phishing=human_phishing,
        llm_legitimate=llm_legitimate,
        llm_phishing=llm_phishing,
    )

    assert len(result) == 4

    assert set(result["experiment_group"]) == {
        "human_legitimate",
        "human_phishing",
        "llm_legitimate",
        "llm_phishing",
    }

    llm_legitimate_row = result.loc[
        result["experiment_group"]
        == "llm_legitimate"
    ].iloc[0]

    assert llm_legitimate_row["phishing_label"] == 0
    assert llm_legitimate_row["original_label"] == 1


def test_validate_feature_frame() -> None:
    feature_frame = pd.DataFrame(
        {
            "word_count": [10, 20],
            "url_count": [0, 1],
            "email_class": [
                "legitimate",
                "phishing",
            ],
        }
    )

    summary = validate_feature_frame(
        feature_frame
    )

    assert summary["rows"] == 2
    assert summary["columns"] == 3
    assert summary["total_missing_values"] == 0
    assert summary["infinite_values"] == 0