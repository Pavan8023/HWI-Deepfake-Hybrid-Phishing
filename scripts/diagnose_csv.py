from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect malformed CSV rows without modifying the source file."
    )
    parser.add_argument("file", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of malformed rows to display.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    path = arguments.file.resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.reader(file)

        header = next(reader)
        expected_fields = len(header)

        print(f"File: {path}")
        print(f"Expected fields: {expected_fields}")
        print(f"Header: {header}")
        print()

        malformed_count = 0
        total_rows = 0

        for line_number, row in enumerate(reader, start=2):
            total_rows += 1

            if len(row) == expected_fields:
                continue

            malformed_count += 1

            print("-" * 80)
            print(f"Line: {line_number}")
            print(f"Fields found: {len(row)}")
            print(f"Row preview: {row[:10]}")

            if malformed_count >= arguments.limit:
                break

    print()
    print(f"Rows inspected: {total_rows}")
    print(f"Malformed rows displayed: {malformed_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())