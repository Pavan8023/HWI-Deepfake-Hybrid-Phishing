from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config import (
    APPROVED_RAW_DATASET_CATEGORIES,
    CANDIDATE_TARGET_COLUMN_KEYWORDS,
    CSV_ENCODINGS_TO_TRY,
    DATASET_METADATA_PATH,
    DATASET_SUMMARIES_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SUPPORTED_DATASET_EXTENSIONS,
)
from src.utils import get_logger, timestamp_utc, to_jsonable, write_json


LOGGER = get_logger(__name__)

INVENTORY_COLUMNS = [
    "relative_path",
    "dataset_path",
    "file_name",
    "extension",
    "file_size_bytes",
    "dataset_category",
    "load_status",
    "load_error",
    "row_count",
    "column_count",
    "column_names",
    "dtypes",
    "missing_value_counts",
    "duplicate_row_count",
    "memory_usage_bytes",
    "candidate_target_columns",
    "unique_value_counts",
    "unit_of_analysis_note",
    "licence_note",
    "metadata_notes",
    "generated_at_utc",
]

QUALITY_COLUMNS = [
    "relative_path",
    "dataset_path",
    "dataset_category",
    "load_status",
    "load_error",
    "row_count",
    "column_count",
    "total_missing_values",
    "duplicate_row_count",
    "memory_usage_bytes",
    "candidate_target_columns",
    "unit_of_analysis_note",
    "licence_note",
    "generated_at_utc",
]

RAW_FILE_MANIFEST_COLUMNS = [
    "relative_path",
    "category",
    "filename",
    "extension",
    "size_bytes",
    "size_mb",
    "supported",
    "last_modified",
]


@dataclass
class InventoryBuildDiagnostics:
    """Capture file-discovery and load outcomes for one inventory run."""

    approved_category_status: dict[str, bool]
    unexpected_categories: list[str] = field(default_factory=list)
    supported_files: list[str] = field(default_factory=list)
    unsupported_files: list[str] = field(default_factory=list)
    successful_files: list[str] = field(default_factory=list)
    failed_files: dict[str, str] = field(default_factory=dict)


class DatasetLoaderError(RuntimeError):
    """Raised when dataset discovery or loading fails in a controlled way."""


def is_supported_dataset(path: Path | str) -> bool:
    """Return True when the file extension is supported by the loader."""

    return Path(path).suffix.lower() in SUPPORTED_DATASET_EXTENSIONS


def _relative_to_raw_dir(path: Path, raw_data_dir: Path = RAW_DATA_DIR) -> str:
    """Return a dataset path relative to the raw data directory when possible."""

    raw_dir = Path(raw_data_dir).resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(raw_dir).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def validate_dataset_path(path: Path | str) -> Path:
    """Validate that a dataset path exists and is supported."""

    dataset_path = Path(path).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise DatasetLoaderError(f"Dataset path is not a file: {dataset_path}")
    if not is_supported_dataset(dataset_path):
        raise DatasetLoaderError(
            f"Unsupported dataset extension '{dataset_path.suffix}' for {dataset_path.name}"
        )

    return dataset_path


def discover_raw_category_directories(
    raw_data_dir: Path = RAW_DATA_DIR,
) -> dict[str, Path]:
    """Return the direct child directories currently present under data/raw."""

    raw_dir = Path(raw_data_dir).resolve()
    if not raw_dir.exists():
        return {}

    return {
        child.name: child
        for child in sorted(raw_dir.iterdir(), key=lambda item: item.name.lower())
        if child.is_dir()
    }


def validate_approved_raw_categories(
    raw_data_dir: Path = RAW_DATA_DIR,
    approved_categories: Iterable[str] = APPROVED_RAW_DATASET_CATEGORIES,
) -> dict[str, bool]:
    """Return existence flags for each approved raw category."""

    raw_dir = Path(raw_data_dir).resolve()
    return {
        category: (raw_dir / category).exists() and (raw_dir / category).is_dir()
        for category in approved_categories
    }


def find_unexpected_raw_categories(
    raw_data_dir: Path = RAW_DATA_DIR,
    approved_categories: Iterable[str] = APPROVED_RAW_DATASET_CATEGORIES,
) -> list[Path]:
    """Find direct child raw-data directories that are not approved categories."""

    category_directories = discover_raw_category_directories(raw_data_dir=raw_data_dir)
    approved = set(approved_categories)
    return [
        path
        for name, path in category_directories.items()
        if name not in approved
    ]


def _approved_category_paths(
    raw_data_dir: Path = RAW_DATA_DIR,
    approved_categories: Iterable[str] = APPROVED_RAW_DATASET_CATEGORIES,
) -> list[Path]:
    raw_dir = Path(raw_data_dir).resolve()
    return [
        raw_dir / category
        for category in approved_categories
        if (raw_dir / category).exists() and (raw_dir / category).is_dir()
    ]


def discover_dataset_files(
    raw_data_dir: Path = RAW_DATA_DIR,
    approved_categories: Iterable[str] = APPROVED_RAW_DATASET_CATEGORIES,
) -> list[Path]:
    """Recursively discover supported datasets inside approved raw categories only."""

    dataset_paths: list[Path] = []
    for category_path in _approved_category_paths(
        raw_data_dir=raw_data_dir,
        approved_categories=approved_categories,
    ):
        dataset_paths.extend(
            [
                path
                for path in category_path.rglob("*")
                if path.is_file() and is_supported_dataset(path)
            ]
        )

    return sorted(dataset_paths, key=lambda item: item.as_posix().lower())


def discover_unsupported_raw_files(
    raw_data_dir: Path = RAW_DATA_DIR,
    approved_categories: Iterable[str] = APPROVED_RAW_DATASET_CATEGORIES,
) -> list[Path]:
    """List unsupported files found within approved raw categories."""

    unsupported_paths: list[Path] = []
    for category_path in _approved_category_paths(
        raw_data_dir=raw_data_dir,
        approved_categories=approved_categories,
    ):
        unsupported_paths.extend(
            [
                path
                for path in category_path.rglob("*")
                if path.is_file() and not is_supported_dataset(path)
            ]
        )

    return sorted(unsupported_paths, key=lambda item: item.as_posix().lower())


def detect_dataset_category(
    dataset_path: Path,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> str:
    """Infer the top-level raw category from a dataset path."""

    relative_path = _relative_to_raw_dir(dataset_path, raw_data_dir=raw_data_dir)
    parts = Path(relative_path).parts
    if parts:
        return parts[0]
    return "unclassified"


def create_raw_file_manifest(
    raw_data_dir: Path = RAW_DATA_DIR,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Create a manifest of all files currently present under data/raw."""

    raw_dir = Path(raw_data_dir).resolve()
    records: list[dict[str, Any]] = []

    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
            if not path.is_file():
                continue
            stat = path.stat()
            records.append(
                {
                    "relative_path": _relative_to_raw_dir(path, raw_data_dir=raw_dir),
                    "category": detect_dataset_category(path, raw_data_dir=raw_dir),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": int(stat.st_size),
                    "size_mb": round(stat.st_size / (1024 * 1024), 4),
                    "supported": is_supported_dataset(path),
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

    manifest_frame = pd.DataFrame(records, columns=RAW_FILE_MANIFEST_COLUMNS)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_frame.to_csv(output_path, index=False, encoding="utf-8")
    return manifest_frame


def _save_unsupported_raw_files_report(
    unsupported_files: list[Path],
    raw_data_dir: Path = RAW_DATA_DIR,
    output_path: Path | None = None,
) -> pd.DataFrame:
    records = [
        {
            "relative_path": _relative_to_raw_dir(path, raw_data_dir=raw_data_dir),
            "category": detect_dataset_category(path, raw_data_dir=raw_data_dir),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": int(path.stat().st_size),
        }
        for path in unsupported_files
    ]

    output_frame = pd.DataFrame(
        records,
        columns=["relative_path", "category", "filename", "extension", "size_bytes"],
    )
    if output_path is None:
        output_path = REPORTS_DIR / "unsupported_raw_files.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False, encoding="utf-8")
    return output_frame


def _load_csv(
    dataset_path: Path,
    *,
    chunksize: int | None = None,
    nrows: int | None = None,
    encoding: str | None = None,
    low_memory: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | Iterable[pd.DataFrame]:
    encodings = (encoding,) if encoding else CSV_ENCODINGS_TO_TRY
    errors: list[str] = []

    for candidate_encoding in encodings:
        try:
            return pd.read_csv(
                dataset_path,
                encoding=candidate_encoding,
                chunksize=chunksize,
                nrows=nrows,
                low_memory=low_memory,
                **kwargs,
            )
        except UnicodeDecodeError as exc:
            errors.append(f"{candidate_encoding}: {exc}")
        except Exception as exc:
            raise DatasetLoaderError(
                f"Failed to read CSV dataset '{dataset_path.name}' using "
                f"encoding '{candidate_encoding}': {exc}"
            ) from exc

    raise DatasetLoaderError(
        "Unable to decode CSV dataset with the configured encodings. "
        f"Errors: {'; '.join(errors)}"
    )


def _load_excel(
    dataset_path: Path,
    *,
    sheet_name: int | str = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    engine = "openpyxl" if dataset_path.suffix.lower() == ".xlsx" else "xlrd"
    try:
        excel_result = pd.read_excel(
            dataset_path,
            sheet_name=sheet_name,
            engine=engine,
            **kwargs,
        )
        if not isinstance(excel_result, pd.DataFrame):
            raise DatasetLoaderError(
                "Expected a single-sheet DataFrame when loading Excel data. "
                "Pass a specific sheet name or index instead of requesting all sheets."
            )
        return excel_result
    except ImportError as exc:
        raise DatasetLoaderError(
            f"Missing optional dependency '{engine}' required for {dataset_path.name}."
        ) from exc
    except Exception as exc:
        raise DatasetLoaderError(
            f"Failed to read Excel dataset '{dataset_path.name}': {exc}"
        ) from exc


def _load_json(dataset_path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_json(dataset_path, **kwargs)
    except ValueError:
        try:
            return pd.read_json(dataset_path, lines=True, **kwargs)
        except Exception as exc:
            raise DatasetLoaderError(
                f"Failed to read JSON dataset '{dataset_path.name}': {exc}"
            ) from exc
    except Exception as exc:
        raise DatasetLoaderError(
            f"Failed to read JSON dataset '{dataset_path.name}': {exc}"
        ) from exc


def _load_parquet(dataset_path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_parquet(dataset_path, **kwargs)
    except ImportError as exc:
        raise DatasetLoaderError(
            f"Missing optional dependency for parquet support when loading {dataset_path.name}."
        ) from exc
    except Exception as exc:
        raise DatasetLoaderError(
            f"Failed to read parquet dataset '{dataset_path.name}': {exc}"
        ) from exc


def load_dataset(
    path: Path | str,
    *,
    chunksize: int | None = None,
    nrows: int | None = None,
    encoding: str | None = None,
    low_memory: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | Iterable[pd.DataFrame]:
    """Load a supported dataset without mutating the raw source file."""

    dataset_path = validate_dataset_path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        return _load_csv(
            dataset_path,
            chunksize=chunksize,
            nrows=nrows,
            encoding=encoding,
            low_memory=low_memory,
            **kwargs,
        )
    if suffix in {".xlsx", ".xls"}:
        return _load_excel(dataset_path, nrows=nrows, **kwargs)
    if suffix == ".json":
        return _load_json(dataset_path, **kwargs)
    if suffix == ".parquet":
        return _load_parquet(dataset_path, **kwargs)

    raise DatasetLoaderError(f"No loader is implemented for extension '{suffix}'.")


def load_dataset_preview(
    path: Path | str,
    *,
    preview_rows: int = 5,
    encoding: str | None = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load a small preview without forcing a full read for CSV and Excel files."""

    dataset_path = validate_dataset_path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        preview = load_dataset(
            dataset_path,
            nrows=preview_rows,
            encoding=encoding,
            low_memory=low_memory,
        )
        if isinstance(preview, pd.DataFrame):
            return preview
    elif suffix in {".xlsx", ".xls"}:
        preview = load_dataset(dataset_path, nrows=preview_rows)
        if isinstance(preview, pd.DataFrame):
            return preview
    else:
        preview = load_dataset(dataset_path)
        if isinstance(preview, pd.DataFrame):
            return preview.head(preview_rows)

    raise DatasetLoaderError(f"Unable to produce a preview for {dataset_path.name}.")


def load_dataset_metadata(
    metadata_path: Path = DATASET_METADATA_PATH,
) -> dict[str, dict[str, Any]]:
    """Load optional manual dataset metadata for unit-of-analysis and licence notes."""

    resolved_path = Path(metadata_path).resolve()
    if not resolved_path.exists():
        return {}

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetLoaderError(
            f"Dataset metadata file is not valid JSON: {resolved_path}"
        ) from exc

    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise DatasetLoaderError(
            "Dataset metadata JSON must contain a top-level 'datasets' object."
        )

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in datasets.items():
        if isinstance(value, dict):
            normalized[str(key)] = value
    return normalized


def resolve_dataset_metadata(
    dataset_path: Path,
    metadata_map: dict[str, dict[str, Any]],
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> dict[str, Any]:
    """Resolve metadata using relative path, filename, or stem-based lookup."""

    candidate_keys = [
        _relative_to_raw_dir(dataset_path, raw_data_dir=raw_data_dir),
        dataset_path.name,
        dataset_path.stem,
    ]

    for key in candidate_keys:
        if key in metadata_map:
            return metadata_map[key]

    return {}


def detect_candidate_target_columns(columns: Iterable[Any]) -> list[str]:
    """Flag plausible target columns using transparent keyword matching."""

    candidates: list[str] = []
    for column in columns:
        column_name = str(column)
        lowered = column_name.lower()
        if any(keyword in lowered for keyword in CANDIDATE_TARGET_COLUMN_KEYWORDS):
            candidates.append(column_name)
    return candidates


def generate_dataset_summary(
    dataframe: pd.DataFrame,
    dataset_path: Path | str,
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a summary describing dataset structure and quality characteristics."""

    resolved_path = Path(dataset_path).resolve()
    metadata = metadata or {}

    column_names = [str(column) for column in dataframe.columns]
    missing_value_counts = {
        str(column): int(value) for column, value in dataframe.isna().sum().items()
    }
    unique_value_counts = {
        str(column): int(value)
        for column, value in dataframe.nunique(dropna=False).items()
    }
    dtypes = {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}

    summary = {
        "relative_path": _relative_to_raw_dir(resolved_path, raw_data_dir=raw_data_dir),
        "dataset_path": resolved_path.as_posix(),
        "file_name": resolved_path.name,
        "extension": resolved_path.suffix.lower(),
        "file_size_bytes": int(resolved_path.stat().st_size),
        "dataset_category": detect_dataset_category(resolved_path, raw_data_dir=raw_data_dir),
        "load_status": "success",
        "load_error": "",
        "row_count": int(len(dataframe)),
        "column_count": int(dataframe.shape[1]),
        "column_names": column_names,
        "dtypes": dtypes,
        "missing_value_counts": missing_value_counts,
        "total_missing_values": int(sum(missing_value_counts.values())),
        "duplicate_row_count": int(dataframe.duplicated().sum()) if len(dataframe) else 0,
        "memory_usage_bytes": int(dataframe.memory_usage(deep=True).sum()),
        "candidate_target_columns": detect_candidate_target_columns(column_names),
        "unique_value_counts": unique_value_counts,
        "unit_of_analysis_note": metadata.get("unit_of_analysis", ""),
        "licence_note": metadata.get("licence", ""),
        "metadata_notes": metadata.get("notes", ""),
        "generated_at_utc": timestamp_utc(),
    }
    return summary


def _failed_inventory_record(
    dataset_path: Path,
    error_message: str,
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "relative_path": _relative_to_raw_dir(dataset_path, raw_data_dir=raw_data_dir),
        "dataset_path": dataset_path.resolve().as_posix(),
        "file_name": dataset_path.name,
        "extension": dataset_path.suffix.lower(),
        "file_size_bytes": int(dataset_path.stat().st_size),
        "dataset_category": detect_dataset_category(dataset_path, raw_data_dir=raw_data_dir),
        "load_status": "failed",
        "load_error": error_message,
        "row_count": None,
        "column_count": None,
        "column_names": [],
        "dtypes": {},
        "missing_value_counts": {},
        "duplicate_row_count": None,
        "memory_usage_bytes": None,
        "candidate_target_columns": [],
        "unique_value_counts": {},
        "unit_of_analysis_note": metadata.get("unit_of_analysis", ""),
        "licence_note": metadata.get("licence", ""),
        "metadata_notes": metadata.get("notes", ""),
        "generated_at_utc": timestamp_utc(),
    }


def save_dataset_summary(
    summary: dict[str, Any],
    output_dir: Path = DATASET_SUMMARIES_DIR,
) -> Path:
    """Save a single dataset summary as JSON."""

    relative_path = Path(summary["relative_path"])
    output_name = "__".join(relative_path.parts).replace(relative_path.suffix, "")
    output_path = Path(output_dir) / f"{output_name}_summary.json"
    write_json(summary, output_path)
    return output_path


def _inventory_record_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": summary["relative_path"],
        "dataset_path": summary["dataset_path"],
        "file_name": summary["file_name"],
        "extension": summary["extension"],
        "file_size_bytes": summary["file_size_bytes"],
        "dataset_category": summary["dataset_category"],
        "load_status": summary["load_status"],
        "load_error": summary["load_error"],
        "row_count": summary["row_count"],
        "column_count": summary["column_count"],
        "column_names": summary["column_names"],
        "dtypes": summary["dtypes"],
        "missing_value_counts": summary["missing_value_counts"],
        "duplicate_row_count": summary["duplicate_row_count"],
        "memory_usage_bytes": summary["memory_usage_bytes"],
        "candidate_target_columns": summary["candidate_target_columns"],
        "unique_value_counts": summary["unique_value_counts"],
        "unit_of_analysis_note": summary["unit_of_analysis_note"],
        "licence_note": summary["licence_note"],
        "metadata_notes": summary["metadata_notes"],
        "generated_at_utc": summary["generated_at_utc"],
    }


def _quality_record_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": summary["relative_path"],
        "dataset_path": summary["dataset_path"],
        "dataset_category": summary["dataset_category"],
        "load_status": summary["load_status"],
        "load_error": summary["load_error"],
        "row_count": summary["row_count"],
        "column_count": summary["column_count"],
        "total_missing_values": summary.get("total_missing_values"),
        "duplicate_row_count": summary["duplicate_row_count"],
        "memory_usage_bytes": summary["memory_usage_bytes"],
        "candidate_target_columns": summary["candidate_target_columns"],
        "unit_of_analysis_note": summary["unit_of_analysis_note"],
        "licence_note": summary["licence_note"],
        "generated_at_utc": summary["generated_at_utc"],
    }


def _records_to_csv_ready_frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=columns)

    csv_ready_records: list[dict[str, Any]] = []
    for record in records:
        normalized: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(to_jsonable(value), ensure_ascii=False)
            else:
                normalized[key] = value
        csv_ready_records.append(normalized)
    return pd.DataFrame(csv_ready_records, columns=columns)


def save_inventory_reports(
    inventory_records: list[dict[str, Any]],
    quality_records: list[dict[str, Any]],
    output_dir: Path = REPORTS_DIR,
) -> dict[str, Path]:
    """Persist dataset inventory outputs in CSV and JSON formats."""

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    inventory_csv_path = resolved_output_dir / "dataset_inventory.csv"
    inventory_json_path = resolved_output_dir / "dataset_inventory.json"
    quality_csv_path = resolved_output_dir / "data_quality_summary.csv"

    inventory_frame = _records_to_csv_ready_frame(inventory_records, INVENTORY_COLUMNS)
    quality_frame = _records_to_csv_ready_frame(quality_records, QUALITY_COLUMNS)

    inventory_frame.to_csv(inventory_csv_path, index=False, encoding="utf-8")
    quality_frame.to_csv(quality_csv_path, index=False, encoding="utf-8")
    write_json(inventory_records, inventory_json_path)

    return {
        "inventory_csv": inventory_csv_path,
        "inventory_json": inventory_json_path,
        "quality_csv": quality_csv_path,
    }


def build_dataset_inventory(
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
    metadata_path: Path = DATASET_METADATA_PATH,
    output_dir: Path = REPORTS_DIR,
    save_summaries: bool = True,
    summaries_output_dir: Path | None = None,
    return_diagnostics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, InventoryBuildDiagnostics]:
    """Discover datasets, summarise them, save reports, and continue past per-file errors."""

    raw_dir = Path(raw_data_dir).resolve()
    metadata_map = load_dataset_metadata(metadata_path=metadata_path)

    approved_category_status = validate_approved_raw_categories(raw_data_dir=raw_dir)
    unexpected_categories = [
        path.name for path in find_unexpected_raw_categories(raw_data_dir=raw_dir)
    ]
    if unexpected_categories:
        LOGGER.warning("Unexpected raw categories found: %s", unexpected_categories)

    supported_dataset_paths = discover_dataset_files(raw_data_dir=raw_dir)
    unsupported_dataset_paths = discover_unsupported_raw_files(raw_data_dir=raw_dir)

    output_dir_path = Path(output_dir).resolve()
    manifest_output_path = output_dir_path / "raw_file_manifest.csv"
    unsupported_output_path = output_dir_path / "unsupported_raw_files.csv"

    create_raw_file_manifest(raw_data_dir=raw_dir, output_path=manifest_output_path)
    _save_unsupported_raw_files_report(
        unsupported_dataset_paths,
        raw_data_dir=raw_dir,
        output_path=unsupported_output_path,
    )

    diagnostics = InventoryBuildDiagnostics(
        approved_category_status=approved_category_status,
        unexpected_categories=unexpected_categories,
        supported_files=[_relative_to_raw_dir(path, raw_data_dir=raw_dir) for path in supported_dataset_paths],
        unsupported_files=[_relative_to_raw_dir(path, raw_data_dir=raw_dir) for path in unsupported_dataset_paths],
    )

    inventory_records: list[dict[str, Any]] = []
    quality_records: list[dict[str, Any]] = []

    summaries_dir = (
        Path(summaries_output_dir).resolve()
        if summaries_output_dir is not None
        else output_dir_path / "dataset_summaries"
    )

    for dataset_path in supported_dataset_paths:
        relative_path = _relative_to_raw_dir(dataset_path, raw_data_dir=raw_dir)
        metadata = resolve_dataset_metadata(
            dataset_path,
            metadata_map,
            raw_data_dir=raw_dir,
        )

        try:
            LOGGER.info("Profiling dataset: %s", relative_path)
            dataset = load_dataset(dataset_path)
            if not isinstance(dataset, pd.DataFrame):
                raise DatasetLoaderError(
                    "Chunked dataset iterators are not supported for inventory generation."
                )

            summary = generate_dataset_summary(
                dataset,
                dataset_path,
                raw_data_dir=raw_dir,
                metadata=metadata,
            )
            inventory_record = _inventory_record_from_summary(summary)
            quality_record = _quality_record_from_summary(summary)

            inventory_records.append(inventory_record)
            quality_records.append(quality_record)
            diagnostics.successful_files.append(relative_path)

            if save_summaries:
                save_dataset_summary(summary, output_dir=summaries_dir)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Failed to load dataset %s: %s", relative_path, error_message)
            failure_record = _failed_inventory_record(
                dataset_path,
                error_message,
                raw_data_dir=raw_dir,
                metadata=metadata,
            )
            inventory_records.append(failure_record)
            quality_records.append(_quality_record_from_summary(failure_record))
            diagnostics.failed_files[relative_path] = error_message

    save_inventory_reports(inventory_records, quality_records, output_dir=output_dir_path)

    inventory_frame = pd.DataFrame(inventory_records, columns=INVENTORY_COLUMNS)
    quality_frame = pd.DataFrame(quality_records, columns=QUALITY_COLUMNS)

    if return_diagnostics:
        return inventory_frame, quality_frame, diagnostics
    return inventory_frame, quality_frame
