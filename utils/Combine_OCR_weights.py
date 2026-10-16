"""Merge OCR results and event weights from SQLite databases into a CSV file."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OCR_DATABASE = PROJECT_ROOT / "out" / "OCR_readings_2023-2025.db"
DEFAULT_WEIGHTS_DATABASE = PROJECT_ROOT / "data" / "Events23-25_weights.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "out" / "weights_ocr_2023-2025.csv"
OCR_TABLE = "final_summary"
WEIGHTS_TABLE = "event"
OCR_JOIN_COLUMN = "video_stem"
WEIGHTS_JOIN_COLUMN = "Event_ID"


def read_table(database: Path, table: str) -> pd.DataFrame:
    """Read a SQLite table and fail with a useful error when it is unavailable."""
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if table not in tables:
            available = ", ".join(sorted(tables)) or "(none)"
            raise ValueError(
                f"Table {table!r} not found in {database}. "
                f"Available tables: {available}"
            )
        return pd.read_sql_query(
            f'SELECT * FROM "{table.replace(chr(34), chr(34) * 2)}"',
            connection,
        )


def merge_databases(
    ocr_database: Path,
    weights_database: Path,
    output_file: Path,
    merge_type: str = "outer",
) -> pd.DataFrame:
    """Merge OCR summary rows with event weights and write the resulting CSV."""
    print(f"Loading OCR database: {ocr_database}")
    ocr_df = read_table(ocr_database, OCR_TABLE)
    print(f"OCR rows: {len(ocr_df)}")

    print(f"Loading weights database: {weights_database}")
    weights_df = read_table(weights_database, WEIGHTS_TABLE)
    print(f"Weight rows: {len(weights_df)}")

    for dataframe, column, source in (
        (ocr_df, OCR_JOIN_COLUMN, f"{OCR_TABLE} in {ocr_database}"),
        (weights_df, WEIGHTS_JOIN_COLUMN, f"{WEIGHTS_TABLE} in {weights_database}"),
    ):
        if column not in dataframe.columns:
            raise ValueError(f"Required column {column!r} is missing from {source}")

    print(
        f"Merging on {OCR_TABLE}.{OCR_JOIN_COLUMN} = "
        f"{WEIGHTS_TABLE}.{WEIGHTS_JOIN_COLUMN} ({merge_type} join)"
    )
    merged_df = pd.merge(
        ocr_df,
        weights_df,
        left_on=OCR_JOIN_COLUMN,
        right_on=WEIGHTS_JOIN_COLUMN,
        how=merge_type,
        suffixes=("_ocr", "_weights"),
    )
    merged_df["_ocr_row"] = merged_df[OCR_JOIN_COLUMN].notna()
    merged_df = (
        merged_df.sort_values("_ocr_row", ascending=False, kind="stable")
        .drop(columns="_ocr_row")
        .reset_index(drop=True)
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    matched = merged_df[OCR_JOIN_COLUMN].notna() & merged_df[WEIGHTS_JOIN_COLUMN].notna()
    only_ocr = merged_df[WEIGHTS_JOIN_COLUMN].isna().sum()
    only_weights = merged_df[OCR_JOIN_COLUMN].isna().sum()
    print(f"Output: {output_file}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Matched rows: {matched.sum()}")
    print(f"Rows only in OCR database: {only_ocr}")
    print(f"Rows only in weights database: {only_weights}")
    return merged_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge OCR results and event weights from SQLite databases."
    )
    parser.add_argument(
        "--ocr-database",
        type=Path,
        default=DEFAULT_OCR_DATABASE,
        help=f"OCR database (default: {DEFAULT_OCR_DATABASE})",
    )
    parser.add_argument(
        "--weights-database",
        type=Path,
        default=DEFAULT_WEIGHTS_DATABASE,
        help=f"Weights database (default: {DEFAULT_WEIGHTS_DATABASE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--merge",
        choices=("outer", "left", "inner"),
        default="outer",
        help="Merge type (default: outer)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_databases(
        ocr_database=args.ocr_database,
        weights_database=args.weights_database,
        output_file=args.output,
        merge_type=args.merge,
    )
