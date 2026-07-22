from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE


class PreprocessingError(RuntimeError):
    """Raised when preprocessing cannot be completed safely."""


@dataclass(frozen=True)
class AwarenessSchema:
    """Expected structure of the awareness dataset."""

    target_column: str = "clicked_link"
    identifier_columns: tuple[str, ...] = (
        "user_id",
    )
    excluded_sensitive_columns: tuple[str, ...] = (
        "geo_location",
    )
    possible_leakage_columns: tuple[str, ...] = (
        "reported_email",
    )
    numeric_columns: tuple[str, ...] = (
        "hover_time_ms",
        "session_duration_sec",
    )
    categorical_columns: tuple[str, ...] = (
        "device_type",
        "browser_used",
        "email_language",
    )
    text_columns: tuple[str, ...] = (
        "email_subject",
        "sender_email_domain",
    )
    datetime_column: str = "email_received_time"

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return every raw column required by the preprocessing stage."""

        return (
            *self.identifier_columns,
            *self.excluded_sensitive_columns,
            *self.possible_leakage_columns,
            *self.numeric_columns,
            *self.categorical_columns,
            *self.text_columns,
            self.datetime_column,
            self.target_column,
        )


DEFAULT_AWARENESS_SCHEMA = AwarenessSchema()


@dataclass(frozen=True)
class AwarenessSplit:
    """Raw training and testing partitions."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ProcessedAwarenessSplit:
    """Transformed machine-learning partitions."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


BINARY_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "y",
    "clicked",
    "click",
    "positive",
}

BINARY_FALSE_VALUES = {
    "0",
    "false",
    "no",
    "n",
    "not clicked",
    "not_clicked",
    "negative",
}


def validate_awareness_columns(
    dataframe: pd.DataFrame,
    *,
    schema: AwarenessSchema = DEFAULT_AWARENESS_SCHEMA,
) -> None:
    """Confirm that all required awareness columns are present."""

    if dataframe.empty:
        raise PreprocessingError(
            "The awareness dataset is empty."
        )

    missing_columns = sorted(
        set(schema.required_columns).difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise PreprocessingError(
            "The awareness dataset is missing required columns: "
            + ", ".join(missing_columns)
        )


def encode_binary_target(
    series: pd.Series,
    *,
    column_name: str = "clicked_link",
) -> pd.Series:
    """
    Convert a binary outcome into integer values 0 and 1.

    Accepted positive examples include yes, true, clicked and 1.
    Accepted negative examples include no, false, not clicked and 0.
    """

    if series.empty:
        raise PreprocessingError(
            f"Target column '{column_name}' is empty."
        )

    encoded_values: list[int] = []
    invalid_values: set[str] = set()

    for value in series:
        if pd.isna(value):
            invalid_values.add("<missing>")
            continue

        normalized_value = str(value).strip().lower()

        if normalized_value in BINARY_TRUE_VALUES:
            encoded_values.append(1)
        elif normalized_value in BINARY_FALSE_VALUES:
            encoded_values.append(0)
        else:
            invalid_values.add(normalized_value)

    if invalid_values:
        raise PreprocessingError(
            f"Target column '{column_name}' contains unsupported "
            f"values: {sorted(invalid_values)}"
        )

    encoded = pd.Series(
        encoded_values,
        index=series.index,
        dtype="int64",
        name=column_name,
    )

    if encoded.nunique() != 2:
        raise PreprocessingError(
            f"Target column '{column_name}' must contain two classes."
        )

    return encoded


def classify_time_period(hour: int | float) -> str:
    """Convert an hour of day into a broad time period."""

    numeric_hour = int(hour)

    if 5 <= numeric_hour < 12:
        return "morning"

    if 12 <= numeric_hour < 17:
        return "afternoon"

    if 17 <= numeric_hour < 21:
        return "evening"

    return "night"


def derive_datetime_features(
    series: pd.Series,
    *,
    column_name: str = "email_received_time",
) -> pd.DataFrame:
    """Create model-friendly features from an email timestamp."""

    parsed_datetime = pd.to_datetime(
        series,
        errors="coerce",
    )

    if parsed_datetime.isna().all():
        raise PreprocessingError(
            f"Datetime column '{column_name}' could not be parsed."
        )

    derived = pd.DataFrame(
        index=series.index
    )

    derived["received_hour"] = (
        parsed_datetime.dt.hour
    )

    derived["received_day_of_week"] = (
        parsed_datetime.dt.day_name()
    )

    derived["received_weekend"] = (
        parsed_datetime.dt.dayofweek
        .isin([5, 6])
        .astype("int64")
    )

    derived["received_time_period"] = (
        derived["received_hour"]
        .map(
            lambda value: (
                classify_time_period(value)
                if pd.notna(value)
                else np.nan
            )
        )
    )

    return derived


def clean_text_column(
    series: pd.Series,
) -> pd.Series:
    """Normalize a text or categorical column without imputing it."""

    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "none": pd.NA,
                "null": pd.NA,
            }
        )
    )


def prepare_awareness_dataframe(
    dataframe: pd.DataFrame,
    *,
    schema: AwarenessSchema = DEFAULT_AWARENESS_SCHEMA,
    include_reported_email: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Perform deterministic awareness-data preparation.

    This stage:
    - validates raw columns;
    - encodes the target;
    - excludes identifiers and sensitive columns;
    - excludes reported_email unless explicitly approved;
    - derives timestamp features;
    - converts numeric columns safely;
    - retains email subject and sender domain for later transformation.

    It does not learn medians, means, scaling values or categories.
    Those operations are fitted only on the training partition.
    """

    validate_awareness_columns(
        dataframe,
        schema=schema,
    )

    prepared = dataframe.copy()

    target = encode_binary_target(
        prepared[schema.target_column],
        column_name=schema.target_column,
    )

    columns_to_drop = [
        schema.target_column,
        *schema.identifier_columns,
        *schema.excluded_sensitive_columns,
        schema.datetime_column,
    ]

    if not include_reported_email:
        columns_to_drop.extend(
            schema.possible_leakage_columns
        )

    prepared = prepared.drop(
        columns=columns_to_drop,
        errors="raise",
    )

    datetime_features = derive_datetime_features(
        dataframe[schema.datetime_column],
        column_name=schema.datetime_column,
    )

    prepared = pd.concat(
        [
            prepared,
            datetime_features,
        ],
        axis=1,
    )

    for column in schema.numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

        negative_mask = prepared[column] < 0

        if negative_mask.any():
            prepared.loc[
                negative_mask,
                column,
            ] = np.nan

    columns_to_clean = [
        column
        for column in (
            *schema.categorical_columns,
            *schema.text_columns,
            *schema.possible_leakage_columns,
        )
        if column in prepared.columns
    ]

    columns_to_clean.extend(
        [
            "received_day_of_week",
            "received_time_period",
        ]
    )

    for column in columns_to_clean:
        prepared[column] = clean_text_column(
            prepared[column]
        )

    if schema.target_column in prepared.columns:
        raise PreprocessingError(
            "Target leakage detected: target column remains in predictors."
        )

    forbidden_columns = {
        *schema.identifier_columns,
        *schema.excluded_sensitive_columns,
    }

    if not include_reported_email:
        forbidden_columns.update(
            schema.possible_leakage_columns
        )

    leaked_columns = sorted(
        forbidden_columns.intersection(
            prepared.columns
        )
    )

    if leaked_columns:
        raise PreprocessingError(
            "Excluded columns remain in predictors: "
            + ", ".join(leaked_columns)
        )

    return (
        prepared.reset_index(drop=True),
        target.reset_index(drop=True),
    )


def stratified_awareness_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> AwarenessSplit:
    """Create a reproducible stratified train/test split."""

    if features.empty:
        raise PreprocessingError(
            "Feature DataFrame is empty."
        )

    if len(features) != len(target):
        raise PreprocessingError(
            "Feature and target row counts do not match."
        )

    if target.nunique() != 2:
        raise PreprocessingError(
            "Stratified splitting requires a binary target."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between zero and one."
        )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return AwarenessSplit(
        x_train=x_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def identify_awareness_feature_types(
    dataframe: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Separate predictors into numeric and categorical columns."""

    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns
    ]

    if not numeric_columns:
        raise PreprocessingError(
            "No numeric awareness features were found."
        )

    if not categorical_columns:
        raise PreprocessingError(
            "No categorical awareness features were found."
        )

    return (
        sorted(numeric_columns),
        sorted(categorical_columns),
    )


def build_awareness_preprocessor(
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    """
    Build a leakage-safe scikit-learn preprocessing transformer.

    Numeric pipeline:
    - median imputation;
    - standard scaling.

    Categorical pipeline:
    - most-frequent imputation;
    - one-hot encoding;
    - unknown test categories ignored safely.
    """

    if not numeric_columns:
        raise PreprocessingError(
            "At least one numeric column is required."
        )

    if not categorical_columns:
        raise PreprocessingError(
            "At least one categorical column is required."
        )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(numeric_columns),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(categorical_columns),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fit_transform_awareness_split(
    split: AwarenessSplit,
) -> tuple[
    ProcessedAwarenessSplit,
    ColumnTransformer,
]:
    """
    Fit preprocessing only on training data and transform both partitions.

    This is the key protection against preprocessing leakage.
    """

    (
        numeric_columns,
        categorical_columns,
    ) = identify_awareness_feature_types(
        split.x_train
    )

    preprocessor = build_awareness_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    transformed_train = (
        preprocessor.fit_transform(
            split.x_train
        )
    )

    transformed_test = (
        preprocessor.transform(
            split.x_test
        )
    )

    feature_names = tuple(
        preprocessor.get_feature_names_out()
    )

    x_train_processed = pd.DataFrame(
        transformed_train,
        columns=feature_names,
    )

    x_test_processed = pd.DataFrame(
        transformed_test,
        columns=feature_names,
    )

    metadata = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_names": list(feature_names),
        "train_rows": int(
            len(x_train_processed)
        ),
        "test_rows": int(
            len(x_test_processed)
        ),
        "processed_feature_count": int(
            len(feature_names)
        ),
        "train_target_distribution": {
            str(key): int(value)
            for key, value in (
                split.y_train
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "test_target_distribution": {
            str(key): int(value)
            for key, value in (
                split.y_test
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "preprocessing_fit_scope": (
            "training_partition_only"
        ),
        "reported_email_included": (
            "reported_email"
            in split.x_train.columns
        ),
    }

    processed_split = ProcessedAwarenessSplit(
        x_train=x_train_processed,
        x_test=x_test_processed,
        y_train=split.y_train.copy(),
        y_test=split.y_test.copy(),
        feature_names=feature_names,
        metadata=metadata,
    )

    return processed_split, preprocessor


def preprocess_awareness_dataset(
    dataframe: pd.DataFrame,
    *,
    schema: AwarenessSchema = DEFAULT_AWARENESS_SCHEMA,
    include_reported_email: bool = False,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    ProcessedAwarenessSplit,
    ColumnTransformer,
]:
    """Run the complete leakage-safe awareness preprocessing workflow."""

    features, target = (
        prepare_awareness_dataframe(
            dataframe,
            schema=schema,
            include_reported_email=(
                include_reported_email
            ),
        )
    )

    raw_split = stratified_awareness_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    return fit_transform_awareness_split(
        raw_split
    )


def awareness_schema_to_dict(
    schema: AwarenessSchema = DEFAULT_AWARENESS_SCHEMA,
) -> Mapping[str, Any]:
    """Convert the awareness schema into serializable metadata."""

    return asdict(schema)