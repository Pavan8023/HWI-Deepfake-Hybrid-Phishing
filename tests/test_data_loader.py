from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    build_dataset_inventory,
    create_raw_file_manifest,
    detect_dataset_category,
    discover_dataset_files,
    discover_unsupported_raw_files,
    find_unexpected_raw_categories,
    generate_dataset_summary,
    is_supported_dataset,
    load_dataset,
    load_dataset_metadata,
    resolve_dataset_metadata,
    validate_approved_raw_categories,
    validate_dataset_path,
)


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
    assert summary["unit_of_analysis_note"] == ""


def test_discover_dataset_files_scans_only_approved_categories_recursively(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_csv(raw_dir / "awareness" / "awareness.csv", "score\n1\n")
    _write_csv(raw_dir / "ai_emails" / "nested" / "sample.csv", "text,label\nhello,0\n")
    _write_csv(raw_dir / "emails" / "mail.csv", "subject,label\nx,1\n")
    _write_csv(raw_dir / "phishing_urls" / "urls.csv", "url,type\nhttps://x,benign\n")
    _write_csv(raw_dir / "webpage" / "should_ignore.csv", "a\n1\n")

    discovered = discover_dataset_files(raw_dir)
    relative_paths = [path.relative_to(raw_dir).as_posix() for path in discovered]

    assert relative_paths == [
        "ai_emails/nested/sample.csv",
        "awareness/awareness.csv",
        "emails/mail.csv",
        "phishing_urls/urls.csv",
    ]


def test_discover_dataset_files_supports_multiple_files_per_category(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_csv(raw_dir / "emails" / "first.csv", "subject,label\nx,1\n")
    _write_csv(raw_dir / "emails" / "second.csv", "subject,label\ny,0\n")

    discovered = discover_dataset_files(raw_dir)

    assert [path.name for path in discovered] == ["first.csv", "second.csv"]


def test_discover_unsupported_raw_files_reports_zip_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_csv(raw_dir / "emails" / "mail.csv", "subject,label\nx,1\n")
    zip_path = raw_dir / "emails" / "archive.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(b"PK\x03\x04")

    unsupported = discover_unsupported_raw_files(raw_dir)

    assert [path.name for path in unsupported] == ["archive.zip"]


def test_find_unexpected_raw_categories_reports_unapproved_folders(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    (raw_dir / "awareness").mkdir(parents=True)
    (raw_dir / "webpage").mkdir(parents=True)
    (raw_dir / "breaches").mkdir(parents=True)

    unexpected = find_unexpected_raw_categories(raw_dir)

    assert [path.name for path in unexpected] == ["breaches", "webpage"]


def test_validate_approved_raw_categories_does_not_require_deleted_categories(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    (raw_dir / "awareness").mkdir(parents=True)
    (raw_dir / "ai_emails").mkdir()
    (raw_dir / "emails").mkdir()
    (raw_dir / "phishing_urls").mkdir()

    status = validate_approved_raw_categories(raw_dir)

    assert status == {
        "awareness": True,
        "ai_emails": True,
        "emails": True,
        "phishing_urls": True,
    }


@pytest.mark.parametrize(
    ("relative_path", "expected_category"),
    [
        ("awareness/data.csv", "awareness"),
        ("ai_emails/data.csv", "ai_emails"),
        ("emails/data.csv", "emails"),
        ("phishing_urls/data.csv", "phishing_urls"),
    ],
)
def test_detect_dataset_category_for_approved_tracks(
    tmp_path: Path,
    relative_path: str,
    expected_category: str,
) -> None:
    raw_dir = tmp_path / "raw"
    csv_path = raw_dir / relative_path
    _write_csv(csv_path, "value\n1\n")

    assert detect_dataset_category(csv_path, raw_data_dir=raw_dir) == expected_category


def test_metadata_lookup_by_relative_raw_path(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    csv_path = raw_dir / "emails" / "mail.csv"
    _write_csv(csv_path, "subject,label\nx,1\n")

    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "emails/mail.csv": {
                        "unit_of_analysis": "Requires verification",
                        "licence": "Requires verification",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    metadata_map = load_dataset_metadata(metadata_path)
    resolved = resolve_dataset_metadata(csv_path, metadata_map, raw_data_dir=raw_dir)

    assert resolved["unit_of_analysis"] == "Requires verification"


def test_missing_metadata_is_handled_safely(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    csv_path = raw_dir / "emails" / "mail.csv"
    _write_csv(csv_path, "subject,label\nx,1\n")

    resolved = resolve_dataset_metadata(csv_path, {}, raw_data_dir=raw_dir)

    assert resolved == {}


def test_create_raw_file_manifest_includes_supported_and_unsupported_files(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_csv(raw_dir / "emails" / "mail.csv", "subject,label\nx,1\n")
    notes_path = raw_dir / "emails" / "notes.zip"
    notes_path.write_bytes(b"PK\x03\x04")

    manifest = create_raw_file_manifest(raw_dir)

    assert list(manifest["relative_path"]) == ["emails/mail.csv", "emails/notes.zip"]
    assert list(manifest["supported"]) == [True, False]


def test_build_dataset_inventory_handles_multiple_categories_and_failures(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    reports_dir = tmp_path / "reports"
    metadata_path = tmp_path / "dataset_metadata.json"

    _write_csv(raw_dir / "awareness" / "awareness.csv", "score,label\n1,0\n2,1\n")
    _write_csv(raw_dir / "emails" / "mail.csv", "subject,label\nx,1\n")
    _write_csv(raw_dir / "ai_emails" / "bad.csv", "text,label\n\"broken,row\n")
    metadata_path.write_text(json.dumps({"datasets": {}}), encoding="utf-8")

    inventory_frame, quality_frame, diagnostics = build_dataset_inventory(
        raw_data_dir=raw_dir,
        metadata_path=metadata_path,
        output_dir=reports_dir,
        return_diagnostics=True,
    )

    assert len(diagnostics.supported_files) == 3
    assert len(diagnostics.successful_files) == 2
    assert "ai_emails/bad.csv" in diagnostics.failed_files
    assert "webpage" not in diagnostics.approved_category_status
    assert set(inventory_frame["load_status"]) == {"success", "failed"}
    assert set(quality_frame["load_status"]) == {"success", "failed"}
    assert (reports_dir / "dataset_inventory.csv").exists()
    assert (reports_dir / "dataset_inventory.json").exists()
    assert (reports_dir / "data_quality_summary.csv").exists()
    assert (reports_dir / "raw_file_manifest.csv").exists()
    assert (reports_dir / "unsupported_raw_files.csv").exists()
