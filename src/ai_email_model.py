from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)

from src.config import RANDOM_STATE, TEST_SIZE


class AIEmailModelError(RuntimeError):
    """Raised when AI-email modelling cannot proceed safely."""


AI_EMAIL_METADATA_COLUMNS = {
    "source_type",
    "email_class",
    "source_label",
    "phishing_label",
    "original_label",
    "experiment_group",
    "template_group_id",
    "combined_text",
}


@dataclass(frozen=True)
class AIEmailSplit:
    """Container holding one reproducible train/test split."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame
    feature_names: tuple[str, ...]


def validate_ai_email_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the engineered AI-email feature dataset.
    """

    if dataframe.empty:
        raise AIEmailModelError(
            "The AI-email feature dataset is empty."
        )

    required_columns = {
        "phishing_label",
        "source_type",
        "email_class",
        "experiment_group",
    }

    missing = sorted(
        required_columns.difference(
            dataframe.columns
        )
    )

    if missing:
        raise AIEmailModelError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    target = pd.to_numeric(
        dataframe["phishing_label"],
        errors="coerce",
    )

    if target.isna().any():
        raise AIEmailModelError(
            "phishing_label contains invalid values."
        )

    unique = set(target.unique())

    if unique != {0, 1}:
        raise AIEmailModelError(
            "phishing_label must contain only 0 and 1."
        )


def select_ai_email_numeric_features(
    dataframe: pd.DataFrame,
    *,
    excluded_columns: Sequence[str] = tuple(
        AI_EMAIL_METADATA_COLUMNS
    ),
) -> list[str]:
    """
    Return only numeric predictors while removing
    labels and metadata.
    """

    excluded = set(excluded_columns)

    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    selected = sorted(
        column
        for column in numeric_columns
        if column not in excluded
    )

    if not selected:
        raise AIEmailModelError(
            "No numeric predictors were found."
        )

    return selected

def prepare_ai_email_model_frame(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    Prepare predictors, target and metadata.

    The metadata is never used as model input.
    It is retained only for reporting and
    duplicate-aware splitting.
    """

    validate_ai_email_dataset(dataframe)

    feature_columns = (
        select_ai_email_numeric_features(
            dataframe
        )
    )

    features = (
        dataframe[feature_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    if features.isna().any().any():

        invalid_columns = (
            features.columns[
                features.isna().any()
            ].tolist()
        )

        raise AIEmailModelError(
            "Predictors contain missing values: "
            + ", ".join(invalid_columns)
        )

    target = (
        pd.to_numeric(
            dataframe["phishing_label"],
            errors="raise",
        )
        .astype("int64")
        .rename("phishing_label")
    )

    metadata_columns: list[str] = []

    optional_columns = (
        "template_group_id",
        "combined_text",
        "source_type",
        "email_class",
        "experiment_group",
        "original_label",
    )

    for column in optional_columns:

        if column in dataframe.columns:
            metadata_columns.append(column)

    metadata = (
        dataframe[metadata_columns]
        .copy()
        .reset_index(drop=True)
    )

    return (
        features.reset_index(drop=True),
        target.reset_index(drop=True),
        metadata,
    )


def build_ai_email_template_groups(
    metadata: pd.DataFrame,
) -> pd.Series:
    """
    Build duplicate-aware template hashes.

    If template_group_id is available, use it directly.
    Otherwise, fall back to hashing combined_text.
    If neither exists (unit tests), every row becomes its own group.
    """

    if "template_group_id" in metadata.columns:
        template_groups = (
            metadata["template_group_id"]
            .replace("", np.nan)
            .fillna(pd.NA)
        )

        if template_groups.isna().any():
            raise AIEmailModelError(
                "template_group_id contains missing values."
            )

        return (
            template_groups
            .astype(str)
            .reset_index(drop=True)
        )

    if "combined_text" not in metadata.columns:

        return pd.Series(
            np.arange(len(metadata)),
            index=metadata.index,
        )

    text = (
        metadata["combined_text"]
        .fillna("")
        .astype(str)
    )

    def normalise(value: str) -> str:

        value = value.lower()

        value = re.sub(
            r"https?://\S+|www\.\S+",
            "<url>",
            value,
        )

        value = re.sub(
            r"\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b",
            "<email>",
            value,
        )

        value = re.sub(
            r"\d+",
            "<number>",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        return hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()

    return text.map(normalise)

def stratified_ai_email_split(
    features: pd.DataFrame,
    target: pd.Series,
    metadata: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> AIEmailSplit:
    """
    Create a reproducible train/test split.

    If template_group_id or combined_text exists, GroupShuffleSplit is used so
    duplicate email templates cannot appear in both
    training and testing.

    Otherwise (unit tests), a normal stratified split is used.
    """

    if not (
        len(features)
        == len(target)
        == len(metadata)
    ):
        raise AIEmailModelError(
            "Feature, target and metadata row counts do not match."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    # ---------------------------------------------------------
    # REAL DATASET
    # ---------------------------------------------------------

    if (
        "template_group_id" in metadata.columns
        or "combined_text" in metadata.columns
    ):

        groups = build_ai_email_template_groups(
            metadata
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_index, test_index = next(
            splitter.split(
                features,
                target,
                groups=groups,
            )
        )

    # ---------------------------------------------------------
    # UNIT TESTS
    # ---------------------------------------------------------

    else:

        indices = np.arange(
            len(features)
        )

        train_index, test_index = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )

    return AIEmailSplit(
        x_train=(
            features.iloc[train_index]
            .reset_index(drop=True)
        ),
        x_test=(
            features.iloc[test_index]
            .reset_index(drop=True)
        ),
        y_train=(
            target.iloc[train_index]
            .reset_index(drop=True)
        ),
        y_test=(
            target.iloc[test_index]
            .reset_index(drop=True)
        ),
        metadata_train=(
            metadata.iloc[train_index]
            .reset_index(drop=True)
        ),
        metadata_test=(
            metadata.iloc[test_index]
            .reset_index(drop=True)
        ),
        feature_names=tuple(
            features.columns
        ),
    )

def validate_ai_email_split(
    split: AIEmailSplit,
) -> dict[str, Any]:
    """
    Validate the generated train/test split.
    """

    if list(split.x_train.columns) != list(
        split.x_test.columns
    ):
        raise AIEmailModelError(
            "Training and testing feature columns do not match."
        )

    forbidden_columns = {
        "phishing_label",
        "source_label",
        "original_label",
        "experiment_group",
        "template_group_id",
        "combined_text",
    }

    leakage_columns = sorted(
        forbidden_columns.intersection(
            split.x_train.columns
        )
    )

    if leakage_columns:
        raise AIEmailModelError(
            "Target leakage detected: "
            + ", ".join(leakage_columns)
        )

    split_strategy = (
        "group_shuffle_split"
        if (
            "template_group_id" in split.metadata_train.columns
            or "combined_text" in split.metadata_train.columns
        )
        else "stratified_random_split"
    )

    shared_template_groups = 0
    train_template_group_count = 0
    test_template_group_count = 0

    if "template_group_id" in split.metadata_train.columns:
        train_groups = set(
            split.metadata_train["template_group_id"]
            .astype(str)
            .tolist()
        )
        test_groups = set(
            split.metadata_test["template_group_id"]
            .astype(str)
            .tolist()
        )
        shared_template_groups = len(
            train_groups.intersection(test_groups)
        )
        train_template_group_count = len(train_groups)
        test_template_group_count = len(test_groups)

    return {
        "train_rows": int(
            len(split.x_train)
        ),
        "test_rows": int(
            len(split.x_test)
        ),
        "feature_count": int(
            split.x_train.shape[1]
        ),
        "train_target_distribution": {
            str(k): int(v)
            for k, v in (
                split.y_train
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "test_target_distribution": {
            str(k): int(v)
            for k, v in (
                split.y_test
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "missing_train_values": int(
            split.x_train.isna().sum().sum()
        ),
        "missing_test_values": int(
            split.x_test.isna().sum().sum()
        ),
        "split_strategy": split_strategy,
        "train_template_group_count": train_template_group_count,
        "test_template_group_count": test_template_group_count,
        "shared_template_groups": int(
            shared_template_groups
        ),
        "leakage_columns": leakage_columns,
        "feature_names": list(
            split.feature_names
        ),
    }
