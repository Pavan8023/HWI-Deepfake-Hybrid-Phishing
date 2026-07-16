from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CSV_ENCODINGS_TO_TRY,
    EDA_FIGURES_DIR,
    EDA_REPORTS_DIR,
    EDA_SAMPLE_ROWS,
    EDA_STATISTICS_DIR,
    EDA_TOP_CATEGORIES,
    RANDOM_STATE,
)
from src.data_loader import load_dataset
from src.utils import get_logger

LOGGER = get_logger(__name__)


class EDAError(RuntimeError):
    """Raised when an exploratory-data-analysis operation fails."""


def sanitize_filename(value: str) -> str:
    """
    Convert text into a safe lowercase filename component.

    Example:
        "Awareness Dataset.csv" -> "awareness_dataset_csv"
    """

    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_").lower()

    return cleaned or "dataset"


def file_size_mb(path: str | Path) -> float:
    """Return the file size in megabytes."""

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    return round(dataset_path.stat().st_size / (1024**2), 3)


def _try_read_csv_preview(
    dataset_path: Path,
    number_of_rows: int,
) -> tuple[pd.DataFrame, str]:
    """
    Read a CSV preview using configured encoding fallbacks.

    Returns:
        A dataframe and the encoding that worked.
    """

    last_error: Exception | None = None

    for encoding in CSV_ENCODINGS_TO_TRY:
        try:
            dataframe = pd.read_csv(
                dataset_path,
                encoding=encoding,
                nrows=number_of_rows,
                low_memory=False,
            )
            return dataframe, encoding

        except UnicodeDecodeError as error:
            last_error = error

    raise EDAError(
        f"Could not decode CSV file using configured encodings: "
        f"{dataset_path}"
    ) from last_error


def load_csv_sample(
    path: str | Path,
    sample_rows: int = EDA_SAMPLE_ROWS,
    chunk_size: int = 100_000,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, bool, str]:
    """
    Load a complete small CSV or a reproducible sample of a large CSV.

    The function first reads ``sample_rows + 1`` rows to determine whether
    sampling is required.

    For a large CSV, it processes the file in chunks and assigns a deterministic
    random priority to every row. Only the rows with the smallest priorities are
    retained. This keeps memory usage bounded while allowing rows from the
    complete file to be represented in the final sample.

    Returns:
        dataframe:
            Complete dataset when small, otherwise a sampled DataFrame.

        sampled:
            True when sampling was required.

        encoding:
            Encoding successfully used for reading.
    """

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_path}"
        )

    if sample_rows <= 0:
        raise ValueError(
            "sample_rows must be greater than zero."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    preview, encoding = _try_read_csv_preview(
        dataset_path=dataset_path,
        number_of_rows=sample_rows + 1,
    )

    # The complete dataset fits within the requested limit.
    if len(preview) <= sample_rows:
        return (
            preview.reset_index(drop=True),
            False,
            encoding,
        )

    random_generator = np.random.default_rng(random_state)

    reservoir: pd.DataFrame | None = None

    try:
        chunk_iterator = pd.read_csv(
            dataset_path,
            encoding=encoding,
            chunksize=chunk_size,
            low_memory=False,
        )

        for chunk in chunk_iterator:
            if chunk.empty:
                continue

            working_chunk = chunk.copy()

            working_chunk["_eda_random_priority"] = (
                random_generator.random(len(working_chunk))
            )

            if reservoir is None:
                candidates = working_chunk
            else:
                candidates = pd.concat(
                    [reservoir, working_chunk],
                    ignore_index=True,
                )

            rows_to_keep = min(
                sample_rows,
                len(candidates),
            )

            reservoir = (
                candidates.nsmallest(
                    rows_to_keep,
                    "_eda_random_priority",
                )
                .reset_index(drop=True)
            )

    except Exception as error:
        raise EDAError(
            f"Failed while sampling large CSV file: "
            f"{dataset_path}"
        ) from error

    if reservoir is None or reservoir.empty:
        raise EDAError(
            f"No rows could be sampled from CSV file: "
            f"{dataset_path}"
        )

    reservoir = reservoir.drop(
        columns=["_eda_random_priority"],
        errors="ignore",
    )

    if len(reservoir) != sample_rows:
        LOGGER.warning(
            "Requested %s sampled rows from %s, but only %s "
            "rows were available.",
            sample_rows,
            dataset_path,
            len(reservoir),
        )

    return (
        reservoir.reset_index(drop=True),
        True,
        encoding,
    )


def ensure_dataframe(
    loaded_data: pd.DataFrame | Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """
    Convert a loaded dataset result into one pandas DataFrame.

    The project data loader may return either:

    - a DataFrame for normal loading; or
    - an iterable of DataFrame chunks for chunked loading.

    This helper guarantees that downstream EDA functions receive exactly
    one DataFrame.
    """

    if isinstance(loaded_data, pd.DataFrame):
        return loaded_data

    chunks: list[pd.DataFrame] = []

    for chunk in loaded_data:
        if not isinstance(chunk, pd.DataFrame):
            raise EDAError(
                "The dataset loader returned an iterable containing "
                f"an unsupported object: {type(chunk).__name__}"
            )

        chunks.append(chunk)

    if not chunks:
        raise EDAError(
            "The dataset loader returned an empty chunk iterator."
        )

    return pd.concat(
        chunks,
        ignore_index=True,
    )


def load_eda_dataframe(
    path: str | Path,
    sample_rows: int = EDA_SAMPLE_ROWS,
) -> tuple[pd.DataFrame, bool, dict[str, Any]]:
    """
    Load a dataset for exploratory analysis.

    CSV datasets are sampled safely when they are large.

    Other supported formats are loaded using the existing project loader.
    If they exceed sample_rows, a reproducible random sample is returned.

    Returns:
        dataframe:
            Loaded full dataset or sample.

        sampled:
            Whether the dataframe is a sample.

        loading_metadata:
            Information about how the dataset was loaded.
    """

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {dataset_path}"
        )

    extension = dataset_path.suffix.lower()

    if extension == ".csv":
        dataframe, sampled, encoding = load_csv_sample(
            dataset_path,
            sample_rows=sample_rows,
        )

        return dataframe, sampled, {
            "file_type": extension,
            "encoding": encoding,
            "sampling_method": (
                "chunked_reproducible_sample"
                if sampled
                else "full_dataset"
            ),
            "sample_limit": sample_rows,
        }

    loaded_data = load_dataset(dataset_path)

    dataframe = ensure_dataframe(loaded_data)

    if len(dataframe) > sample_rows:
        sampled_dataframe = dataframe.sample(
            n=sample_rows,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

        return sampled_dataframe, True, {
            "file_type": extension,
            "encoding": None,
            "sampling_method": "random_sample_after_loading",
            "sample_limit": sample_rows,
        }

    full_dataframe = dataframe.reset_index(drop=True)

    return full_dataframe, False, {
        "file_type": extension,
        "encoding": None,
        "sampling_method": "full_dataset",
        "sample_limit": sample_rows,
    }


def calculate_average_string_length(series: pd.Series) -> float:
    """Calculate the average string length of non-null values."""

    non_null = series.dropna()

    if non_null.empty:
        return 0.0

    return float(non_null.astype(str).str.len().mean())


def detect_column_groups(
    dataframe: pd.DataFrame,
    categorical_unique_limit: int = 30,
    text_average_length_threshold: int = 40,
    identifier_uniqueness_threshold: float = 0.95,
) -> dict[str, list[str]]:
    """
    Group columns into broad analytical types.

    Groups returned:
        numeric_columns
        datetime_columns
        boolean_columns
        categorical_columns
        text_columns
        possible_identifier_columns
        constant_columns
        high_cardinality_columns

    These groups are suggestions for EDA only.
    They are not final preprocessing or modelling decisions.
    """

    row_count = max(len(dataframe), 1)

    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    boolean_columns = dataframe.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    datetime_columns = dataframe.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    object_columns = dataframe.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    categorical_columns: list[str] = []
    text_columns: list[str] = []
    possible_identifier_columns: list[str] = []
    constant_columns: list[str] = []
    high_cardinality_columns: list[str] = []

    for column in dataframe.columns:
        series = dataframe[column]
        unique_count = int(series.nunique(dropna=True))

        if unique_count <= 1:
            constant_columns.append(column)

    for column in object_columns:
        series = dataframe[column]
        unique_count = int(series.nunique(dropna=True))
        uniqueness_ratio = unique_count / row_count
        average_length = calculate_average_string_length(series)

        if unique_count <= categorical_unique_limit:
            categorical_columns.append(column)
        else:
            high_cardinality_columns.append(column)

        if average_length >= text_average_length_threshold:
            text_columns.append(column)

        column_name = column.lower()

        identifier_name_pattern = (
            column_name == "id"
            or column_name.endswith("_id")
            or column_name.startswith("id_")
            or "uuid" in column_name
            or "identifier" in column_name
            or "record_number" in column_name
        )

        if (
            uniqueness_ratio >= identifier_uniqueness_threshold
            or identifier_name_pattern
        ):
            possible_identifier_columns.append(column)

    for column in numeric_columns:
        series = dataframe[column]
        unique_count = int(series.nunique(dropna=True))
        uniqueness_ratio = unique_count / row_count
        column_name = column.lower()

        identifier_name_pattern = (
            column_name == "id"
            or column_name.endswith("_id")
            or column_name.startswith("id_")
            or "identifier" in column_name
        )

        if (
            uniqueness_ratio >= identifier_uniqueness_threshold
            or identifier_name_pattern
        ):
            possible_identifier_columns.append(column)

    return {
        "numeric_columns": sorted(set(numeric_columns)),
        "datetime_columns": sorted(set(datetime_columns)),
        "boolean_columns": sorted(set(boolean_columns)),
        "categorical_columns": sorted(set(categorical_columns)),
        "text_columns": sorted(set(text_columns)),
        "possible_identifier_columns": sorted(
            set(possible_identifier_columns)
        ),
        "constant_columns": sorted(set(constant_columns)),
        "high_cardinality_columns": sorted(
            set(high_cardinality_columns)
        ),
    }


def build_column_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a detailed structural profile for every column.

    The output includes:
        dtype
        non-null count
        missing count
        missing percentage
        unique count
        uniqueness percentage
        average string length
        minimum and maximum where meaningful
        sample values
    """

    records: list[dict[str, Any]] = []
    row_count = len(dataframe)

    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()

        unique_count = int(non_null.nunique())

        sample_values = (
            non_null.astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        minimum_value: Any = None
        maximum_value: Any = None

        if not non_null.empty:
            try:
                minimum_value = non_null.min()
                maximum_value = non_null.max()
            except TypeError:
                minimum_value = None
                maximum_value = None

        average_string_length = (
            calculate_average_string_length(series)
            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
            )
            else None
        )

        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows_analysed": row_count,
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(
                    float(series.isna().mean() * 100),
                    4,
                ),
                "unique_count": unique_count,
                "uniqueness_percentage": round(
                    (unique_count / row_count * 100)
                    if row_count
                    else 0.0,
                    4,
                ),
                "average_string_length": (
                    round(average_string_length, 3)
                    if average_string_length is not None
                    else None
                ),
                "minimum_value": (
                    str(minimum_value)
                    if minimum_value is not None
                    else None
                ),
                "maximum_value": (
                    str(maximum_value)
                    if maximum_value is not None
                    else None
                ),
                "sample_values": " | ".join(sample_values),
            }
        )

    return pd.DataFrame(records)


def build_numeric_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create detailed descriptive statistics for numeric columns.

    Includes:
        count
        mean
        standard deviation
        minimum
        percentiles
        maximum
        missing percentage
        skewness
        kurtosis
        number of unique values
        zero count
        infinite-value count
    """

    numeric_dataframe = dataframe.select_dtypes(include=[np.number])

    if numeric_dataframe.empty:
        return pd.DataFrame()

    clean_numeric = numeric_dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    summary = clean_numeric.describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    ).transpose()

    summary["missing_count"] = clean_numeric.isna().sum()
    summary["missing_percentage"] = (
        clean_numeric.isna().mean() * 100
    )
    summary["skewness"] = clean_numeric.skew(numeric_only=True)
    summary["kurtosis"] = clean_numeric.kurt(numeric_only=True)
    summary["unique_count"] = clean_numeric.nunique(dropna=True)
    summary["zero_count"] = (clean_numeric == 0).sum()
    summary["infinite_count_original"] = np.isinf(
        numeric_dataframe
    ).sum()

    summary = summary.reset_index().rename(
        columns={"index": "column"}
    )

    return summary


def build_categorical_summary(
    dataframe: pd.DataFrame,
    maximum_unique_values: int = EDA_TOP_CATEGORIES,
) -> pd.DataFrame:
    """
    Create frequency summaries for low-cardinality columns.

    Only columns with no more than maximum_unique_values unique values
    are included.
    """

    records: list[dict[str, Any]] = []

    candidate_columns = dataframe.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    for column in candidate_columns:
        unique_count = dataframe[column].nunique(dropna=True)

        if unique_count > maximum_unique_values:
            continue

        counts = (
            dataframe[column]
            .fillna("<MISSING>")
            .astype(str)
            .value_counts(dropna=False)
        )

        for rank, (value, count) in enumerate(
            counts.items(),
            start=1,
        ):
            records.append(
                {
                    "column": column,
                    "rank": rank,
                    "value": value,
                    "count": int(count),
                    "percentage": round(
                        count / len(dataframe) * 100,
                        4,
                    )
                    if len(dataframe)
                    else 0.0,
                }
            )

    return pd.DataFrame(records)


def build_text_summary(
    dataframe: pd.DataFrame,
    text_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Create text-length statistics for probable text columns.

    This does not perform NLP preprocessing.
    """

    if text_columns is None:
        text_columns = detect_column_groups(
            dataframe
        )["text_columns"]

    records: list[dict[str, Any]] = []

    for column in text_columns:
        if column not in dataframe.columns:
            continue

        non_null = dataframe[column].dropna().astype(str)

        if non_null.empty:
            continue

        character_lengths = non_null.str.len()
        word_counts = non_null.str.split().str.len()

        records.append(
            {
                "column": column,
                "non_null_count": int(len(non_null)),
                "average_character_length": round(
                    float(character_lengths.mean()),
                    3,
                ),
                "median_character_length": round(
                    float(character_lengths.median()),
                    3,
                ),
                "minimum_character_length": int(
                    character_lengths.min()
                ),
                "maximum_character_length": int(
                    character_lengths.max()
                ),
                "average_word_count": round(
                    float(word_counts.mean()),
                    3,
                ),
                "median_word_count": round(
                    float(word_counts.median()),
                    3,
                ),
                "minimum_word_count": int(word_counts.min()),
                "maximum_word_count": int(word_counts.max()),
                "empty_string_count": int(
                    (non_null.str.strip() == "").sum()
                ),
            }
        )

    return pd.DataFrame(records)


def build_duplicate_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Return duplicate-row statistics for the analysed data."""

    duplicate_mask = dataframe.duplicated(keep=False)
    exact_duplicate_rows = int(dataframe.duplicated().sum())
    all_rows_in_duplicate_groups = int(duplicate_mask.sum())

    return {
        "exact_duplicate_rows_after_first_occurrence": (
            exact_duplicate_rows
        ),
        "rows_participating_in_duplicate_groups": (
            all_rows_in_duplicate_groups
        ),
        "duplicate_percentage": round(
            exact_duplicate_rows / len(dataframe) * 100,
            4,
        )
        if len(dataframe)
        else 0.0,
    }


def generate_eda_summary(
    dataframe: pd.DataFrame,
    dataset_name: str,
    dataset_path: str | Path,
    sampled: bool,
    loading_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a JSON-compatible structural EDA summary."""

    column_groups = detect_column_groups(dataframe)
    duplicate_summary = build_duplicate_summary(dataframe)

    return {
        "dataset_name": dataset_name,
        "dataset_path": str(Path(dataset_path).resolve()),
        "file_size_mb": file_size_mb(dataset_path),
        "analysed_rows": int(len(dataframe)),
        "columns": int(dataframe.shape[1]),
        "sampled": bool(sampled),
        "loading_metadata": loading_metadata or {},
        "duplicate_summary": duplicate_summary,
        "total_missing_values_in_analysed_data": int(
            dataframe.isna().sum().sum()
        ),
        "columns_with_missing_values": int(
            (dataframe.isna().sum() > 0).sum()
        ),
        "memory_usage_mb": round(
            dataframe.memory_usage(deep=True).sum() / (1024**2),
            3,
        ),
        "column_groups": column_groups,
        "column_names": dataframe.columns.tolist(),
    }


def save_json_report(
    data: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save JSON data using UTF-8 encoding."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    return path


def plot_missing_values(
    dataframe: pd.DataFrame,
    dataset_name: str,
    output_directory: str | Path = EDA_FIGURES_DIR,
    maximum_columns: int = 30,
) -> Path | None:
    """
    Save a horizontal bar chart of missing-value percentages.

    Returns None when no missing values are present.
    """

    missing_percentage = (
        dataframe.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    missing_percentage = missing_percentage[
        missing_percentage > 0
    ].head(maximum_columns)

    if missing_percentage.empty:
        return None

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(dataset_name)
    figure_path = output_path / f"{safe_name}_missing_values.png"

    figure_height = max(
        5,
        len(missing_percentage) * 0.38,
    )

    plt.figure(figsize=(12, figure_height))
    missing_percentage.sort_values().plot(kind="barh")
    plt.xlabel("Missing values (%)")
    plt.ylabel("Column")
    plt.title(f"Missing-value percentages: {dataset_name}")
    plt.tight_layout()
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return figure_path


def plot_numeric_histograms(
    dataframe: pd.DataFrame,
    dataset_name: str,
    output_directory: str | Path = EDA_FIGURES_DIR,
    maximum_columns: int = 12,
) -> list[Path]:
    """Save one histogram per selected numeric column."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_columns = numeric_columns[:maximum_columns]

    saved_paths: list[Path] = []
    safe_dataset_name = sanitize_filename(dataset_name)

    for column in numeric_columns:
        clean_values = (
            dataframe[column]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if clean_values.empty:
            continue

        if clean_values.nunique() <= 1:
            continue

        figure_path = (
            output_path
            / (
                f"{safe_dataset_name}_"
                f"{sanitize_filename(column)}_histogram.png"
            )
        )

        plt.figure(figsize=(9, 6))
        plt.hist(clean_values, bins=30)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"{column} distribution: {dataset_name}")
        plt.tight_layout()
        plt.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        saved_paths.append(figure_path)

    return saved_paths


def plot_categorical_distributions(
    dataframe: pd.DataFrame,
    dataset_name: str,
    output_directory: str | Path = EDA_FIGURES_DIR,
    maximum_columns: int = 8,
    maximum_categories: int = EDA_TOP_CATEGORIES,
) -> list[Path]:
    """Save value-count plots for selected low-cardinality columns."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    column_groups = detect_column_groups(dataframe)

    categorical_columns = column_groups[
        "categorical_columns"
    ][:maximum_columns]

    saved_paths: list[Path] = []
    safe_dataset_name = sanitize_filename(dataset_name)

    for column in categorical_columns:
        counts = (
            dataframe[column]
            .fillna("<MISSING>")
            .astype(str)
            .value_counts()
            .head(maximum_categories)
            .sort_values()
        )

        if counts.empty:
            continue

        figure_path = (
            output_path
            / (
                f"{safe_dataset_name}_"
                f"{sanitize_filename(column)}_categories.png"
            )
        )

        figure_height = max(5, len(counts) * 0.35)

        plt.figure(figsize=(11, figure_height))
        counts.plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel(column)
        plt.title(f"{column} distribution: {dataset_name}")
        plt.tight_layout()
        plt.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        saved_paths.append(figure_path)

    return saved_paths


def plot_text_lengths(
    dataframe: pd.DataFrame,
    dataset_name: str,
    output_directory: str | Path = EDA_FIGURES_DIR,
    maximum_columns: int = 5,
) -> list[Path]:
    """Save text character-length histograms."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    text_columns = detect_column_groups(
        dataframe
    )["text_columns"][:maximum_columns]

    saved_paths: list[Path] = []
    safe_dataset_name = sanitize_filename(dataset_name)

    for column in text_columns:
        lengths = (
            dataframe[column]
            .dropna()
            .astype(str)
            .str.len()
        )

        if lengths.empty or lengths.nunique() <= 1:
            continue

        figure_path = (
            output_path
            / (
                f"{safe_dataset_name}_"
                f"{sanitize_filename(column)}_text_length.png"
            )
        )

        plt.figure(figsize=(9, 6))
        plt.hist(lengths, bins=40)
        plt.xlabel("Character length")
        plt.ylabel("Frequency")
        plt.title(f"Text-length distribution: {column}")
        plt.tight_layout()
        plt.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        saved_paths.append(figure_path)

    return saved_paths


def build_numeric_correlation_matrix(
    dataframe: pd.DataFrame,
    maximum_columns: int = 50,
) -> pd.DataFrame:
    """
    Build a Pearson correlation matrix for numeric columns.

    This is an exploratory association matrix, not evidence of causation.
    """

    numeric_dataframe = dataframe.select_dtypes(
        include=[np.number]
    )

    if numeric_dataframe.empty:
        return pd.DataFrame()

    usable_columns = [
        column
        for column in numeric_dataframe.columns
        if numeric_dataframe[column].nunique(dropna=True) > 1
    ]

    usable_columns = usable_columns[:maximum_columns]

    if len(usable_columns) < 2:
        return pd.DataFrame()

    return numeric_dataframe[usable_columns].corr(method="pearson")


def save_eda_outputs(
    dataframe: pd.DataFrame,
    dataset_name: str,
    dataset_path: str | Path,
    sampled: bool,
    loading_metadata: dict[str, Any] | None = None,
    reports_directory: str | Path = EDA_REPORTS_DIR,
    statistics_directory: str | Path = EDA_STATISTICS_DIR,
) -> dict[str, Path]:
    """
    Generate and save structural EDA reports.

    This function does not clean the data and does not modify raw files.
    """

    reports_path = Path(reports_directory)
    statistics_path = Path(statistics_directory)

    reports_path.mkdir(parents=True, exist_ok=True)
    statistics_path.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(dataset_name)

    column_profile = build_column_profile(dataframe)
    numeric_summary = build_numeric_summary(dataframe)
    categorical_summary = build_categorical_summary(dataframe)
    text_summary = build_text_summary(dataframe)
    correlation_matrix = build_numeric_correlation_matrix(dataframe)

    eda_summary = generate_eda_summary(
        dataframe=dataframe,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        sampled=sampled,
        loading_metadata=loading_metadata,
    )

    column_profile_path = (
        reports_path / f"{safe_name}_column_profile.csv"
    )

    numeric_summary_path = (
        statistics_path / f"{safe_name}_numeric_summary.csv"
    )

    categorical_summary_path = (
        statistics_path
        / f"{safe_name}_categorical_summary.csv"
    )

    text_summary_path = (
        statistics_path / f"{safe_name}_text_summary.csv"
    )

    correlation_path = (
        statistics_path
        / f"{safe_name}_pearson_correlation.csv"
    )

    summary_path = (
        reports_path / f"{safe_name}_eda_summary.json"
    )

    column_profile.to_csv(
        column_profile_path,
        index=False,
        encoding="utf-8",
    )

    save_json_report(
        eda_summary,
        summary_path,
    )

    outputs: dict[str, Path] = {
        "column_profile": column_profile_path,
        "eda_summary": summary_path,
    }

    if not numeric_summary.empty:
        numeric_summary.to_csv(
            numeric_summary_path,
            index=False,
            encoding="utf-8",
        )
        outputs["numeric_summary"] = numeric_summary_path

    if not categorical_summary.empty:
        categorical_summary.to_csv(
            categorical_summary_path,
            index=False,
            encoding="utf-8",
        )
        outputs["categorical_summary"] = (
            categorical_summary_path
        )

    if not text_summary.empty:
        text_summary.to_csv(
            text_summary_path,
            index=False,
            encoding="utf-8",
        )
        outputs["text_summary"] = text_summary_path

    if not correlation_matrix.empty:
        correlation_matrix.to_csv(
            correlation_path,
            encoding="utf-8",
        )
        outputs["pearson_correlation"] = correlation_path

    return outputs


def run_dataset_eda(
    dataset_path: str | Path,
    dataset_name: str | None = None,
    sample_rows: int = EDA_SAMPLE_ROWS,
) -> dict[str, Any]:
    """
    Run the complete structural EDA workflow for one dataset.

    The function:
        loads the dataset safely;
        creates tabular reports;
        creates basic plots;
        returns a result dictionary.

    It does not:
        clean data;
        encode variables;
        construct HWI;
        train models;
        modify raw data.
    """

    path = Path(dataset_path)

    if dataset_name is None:
        dataset_name = path.stem

    LOGGER.info("Starting EDA for dataset: %s", path)

    dataframe, sampled, loading_metadata = load_eda_dataframe(
        path=path,
        sample_rows=sample_rows,
    )

    saved_reports = save_eda_outputs(
        dataframe=dataframe,
        dataset_name=dataset_name,
        dataset_path=path,
        sampled=sampled,
        loading_metadata=loading_metadata,
        reports_directory=EDA_REPORTS_DIR,
        statistics_directory=EDA_STATISTICS_DIR,
    )

    missing_plot = plot_missing_values(
        dataframe=dataframe,
        dataset_name=dataset_name,
        output_directory=EDA_FIGURES_DIR,
    )

    numeric_histograms = plot_numeric_histograms(
        dataframe=dataframe,
        dataset_name=dataset_name,
        output_directory=EDA_FIGURES_DIR,
    )

    categorical_plots = plot_categorical_distributions(
        dataframe=dataframe,
        dataset_name=dataset_name,
        output_directory=EDA_FIGURES_DIR,
    )

    text_length_plots = plot_text_lengths(
        dataframe=dataframe,
        dataset_name=dataset_name,
        output_directory=EDA_FIGURES_DIR,
    )

    column_groups = detect_column_groups(dataframe)

    result = {
        "dataset_name": dataset_name,
        "dataset_path": str(path.resolve()),
        "rows_analysed": int(len(dataframe)),
        "columns": int(dataframe.shape[1]),
        "sampled": bool(sampled),
        "loading_metadata": loading_metadata,
        "column_groups": column_groups,
        "reports": {
            key: str(value)
            for key, value in saved_reports.items()
        },
        "figures": {
            "missing_values": (
                str(missing_plot)
                if missing_plot is not None
                else None
            ),
            "numeric_histograms": [
                str(item)
                for item in numeric_histograms
            ],
            "categorical_plots": [
                str(item)
                for item in categorical_plots
            ],
            "text_length_plots": [
                str(item)
                for item in text_length_plots
            ],
        },
    }

    LOGGER.info(
        "Completed EDA for %s: %s rows, %s columns",
        dataset_name,
        len(dataframe),
        dataframe.shape[1],
    )

    return result