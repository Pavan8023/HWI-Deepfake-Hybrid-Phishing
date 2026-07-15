from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    build_dataset_inventory,
    discover_dataset_files,
    generate_dataset_summary,
    is_supported_dataset,
    load_dataset,
    validate_dataset_path,
)


def test_supported_extension_validation() -> None:
    assert is_supported_dataset("example.csv")
    assert is_supported_dataset("example.parquet")
    assert not is_supported_dataset("example.txt")


def test_validate_dataset_path_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        validate_dataset_path(missing_path)


def test_load_dataset_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("label,value\nphishing,1\nlegitimate,2\n", encoding="utf-8")

    dataframe = load_dataset(csv_path)

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["label", "value"]


def test_generate_dataset_summary_reports_expected_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "emails.csv"
    csv_path.write_text(
        "label,subject,clicks\nphishing,urgent update,\nlegitimate,newsletter,0\n",
        encoding="utf-8",
    )
    dataframe = load_dataset(csv_path)

    summary = generate_dataset_summary(dataframe, csv_path, raw_data_dir=tmp_path)

    assert summary["row_count"] == 2
    assert summary["column_count"] == 3
    assert "label" in summary["candidate_target_columns"]
    assert summary["missing_value_counts"]["clicks"] == 1


def test_discover_dataset_files_filters_supported_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "urls.csv").write_text("url,label\nhttps://example.com,0\n", encoding="utf-8")
    (raw_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    discovered = discover_dataset_files(raw_dir)

    assert [path.name for path in discovered] == ["urls.csv"]


def test_generate_dataset_summary_handles_empty_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("label,score\n", encoding="utf-8")

    dataframe = load_dataset(csv_path)
    summary = generate_dataset_summary(dataframe, csv_path, raw_data_dir=tmp_path)

    assert summary["row_count"] == 0
    assert summary["column_count"] == 2
    assert summary["duplicate_row_count"] == 0
    assert "label" in summary["candidate_target_columns"]


def test_build_dataset_inventory_handles_empty_dataset_directory(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    reports_dir = tmp_path / "reports"
    metadata_path = tmp_path / "dataset_metadata.json"

    raw_dir.mkdir()
    metadata_path.write_text(json.dumps({"datasets": {}}), encoding="utf-8")

    inventory_frame, quality_frame = build_dataset_inventory(
        raw_data_dir=raw_dir,
        metadata_path=metadata_path,
        output_dir=reports_dir,
    )

    assert inventory_frame.empty
    assert quality_frame.empty
    assert (reports_dir / "dataset_inventory.csv").exists()
    assert (reports_dir / "dataset_inventory.json").exists()
    assert (reports_dir / "data_quality_summary.csv").exists()
