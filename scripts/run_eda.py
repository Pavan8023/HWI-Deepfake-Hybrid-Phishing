from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    APPROVED_RAW_DATASET_CATEGORIES,
    EDA_REPORTS_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import discover_dataset_files  # noqa: E402
from src.eda import run_dataset_eda  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run structural EDA on one approved dataset category "
            "or on all four categories."
        )
    )

    parser.add_argument(
        "--category",
        choices=[*APPROVED_RAW_DATASET_CATEGORIES, "all"],
        default="awareness",
        help=(
            "Dataset category to process. "
            "Use 'all' to process every approved category."
        ),
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=50_000,
        help="Maximum number of rows analysed per dataset.",
    )

    return parser.parse_args()


def dataset_category(dataset_path: Path) -> str:
    """Return the first folder below data/raw."""

    relative_path = dataset_path.relative_to(RAW_DATA_DIR)

    if not relative_path.parts:
        raise ValueError(
            f"Could not determine category for {dataset_path}"
        )

    return relative_path.parts[0]


def create_dataset_name(
    dataset_path: Path,
    category: str,
) -> str:
    """Create a unique output-safe dataset name."""

    relative_path = dataset_path.relative_to(
        RAW_DATA_DIR / category
    )

    path_without_extension = relative_path.with_suffix("")

    path_parts = [
        category,
        *path_without_extension.parts,
    ]

    return "_".join(path_parts)


def select_files(
    all_files: list[Path],
    selected_category: str,
) -> list[Path]:
    """Select files belonging to the requested category."""

    if selected_category == "all":
        return all_files

    return [
        path
        for path in all_files
        if dataset_category(path) == selected_category
    ]


def create_comparison_record(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one EDA result for the comparison report."""

    groups = result["column_groups"]

    return {
        "dataset_name": result["dataset_name"],
        "dataset_path": result["dataset_path"],
        "rows_analysed": result["rows_analysed"],
        "columns": result["columns"],
        "sampled": result["sampled"],
        "sampling_method": result[
            "loading_metadata"
        ].get("sampling_method"),
        "numeric_columns": len(
            groups.get("numeric_columns", [])
        ),
        "categorical_columns": len(
            groups.get("categorical_columns", [])
        ),
        "text_columns": len(
            groups.get("text_columns", [])
        ),
        "possible_identifier_columns": len(
            groups.get("possible_identifier_columns", [])
        ),
        "constant_columns": len(
            groups.get("constant_columns", [])
        ),
        "high_cardinality_columns": len(
            groups.get("high_cardinality_columns", [])
        ),
    }


def main() -> int:
    """Run EDA and save comparison and error reports."""

    arguments = parse_arguments()

    if arguments.sample_rows <= 0:
        raise ValueError(
            "--sample-rows must be greater than zero."
        )

    EDA_REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovered_files = discover_dataset_files(
        RAW_DATA_DIR
    )

    selected_files = select_files(
        all_files=discovered_files,
        selected_category=arguments.category,
    )

    if not selected_files:
        print(
            f"No supported files found for category: "
            f"{arguments.category}"
        )
        return 1

    print("=" * 80)
    print("HWI STRUCTURAL EDA")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Selected category: {arguments.category}")
    print(f"Sample-row limit: {arguments.sample_rows:,}")
    print(f"Files selected: {len(selected_files)}")
    print()

    successful_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for position, dataset_path in enumerate(
        selected_files,
        start=1,
    ):
        category = dataset_category(dataset_path)

        dataset_name = create_dataset_name(
            dataset_path=dataset_path,
            category=category,
        )

        print("-" * 80)
        print(
            f"[{position}/{len(selected_files)}] "
            f"{dataset_path.relative_to(RAW_DATA_DIR)}"
        )
        print(f"Output name: {dataset_name}")
        print("-" * 80)

        try:
            result = run_dataset_eda(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                sample_rows=arguments.sample_rows,
            )

            successful_results.append(result)

            print(
                f"SUCCESS: {result['rows_analysed']:,} rows, "
                f"{result['columns']} columns, "
                f"sampled={result['sampled']}"
            )

        except Exception as error:
            error_record = {
                "category": category,
                "dataset_path": str(dataset_path),
                "dataset_name": dataset_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            errors.append(error_record)

            print(
                f"FAILED: {type(error).__name__}: {error}"
            )

        print()

    comparison_records = [
        create_comparison_record(result)
        for result in successful_results
    ]

    comparison_frame = pd.DataFrame(
        comparison_records
    )

    category_label = arguments.category.replace(
        "/",
        "_",
    )

    comparison_path = (
        EDA_REPORTS_DIR
        / f"{category_label}_eda_comparison.csv"
    )

    comparison_frame.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8",
    )

    results_json_path = (
        EDA_REPORTS_DIR
        / f"{category_label}_eda_results.json"
    )

    with results_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            successful_results,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    errors_path = (
        EDA_REPORTS_DIR
        / f"{category_label}_eda_errors.csv"
    )

    if errors:
        pd.DataFrame(errors).to_csv(
            errors_path,
            index=False,
            encoding="utf-8",
        )
    elif errors_path.exists():
        errors_path.unlink()

    print("=" * 80)
    print("EDA EXECUTION SUMMARY")
    print("=" * 80)
    print(
        f"Successful files: "
        f"{len(successful_results)}"
    )
    print(f"Failed files: {len(errors)}")
    print(f"Comparison report: {comparison_path}")
    print(f"Detailed results: {results_json_path}")

    if errors:
        print(f"Error report: {errors_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())