#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd


def truncate_numeric_columns(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    factor = 10 ** digits
    out = df.copy()
    float_cols = out.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].apply(lambda col: np.trunc(col * factor) / factor)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truncate float values in a CSV file to a fixed number of decimal places (no rounding)."
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument(
        "--output-csv",
        help="Path to output CSV file (default: overwrite input when --in-place is set, else <input>_truncated.csv)",
    )
    parser.add_argument("--digits", type=int, default=3, help="Decimal places to keep (default: 3)")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.digits < 0:
        raise ValueError("--digits must be >= 0")

    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    if args.in_place and args.output_csv:
        raise ValueError("Use either --in-place or --output-csv, not both")

    if args.in_place:
        output_csv = args.input_csv
    elif args.output_csv:
        output_csv = args.output_csv
    else:
        base, ext = os.path.splitext(args.input_csv)
        output_csv = f"{base}_truncated{ext or '.csv'}"

    df = pd.read_csv(args.input_csv)
    truncated = truncate_numeric_columns(df, args.digits)
    truncated.to_csv(output_csv, index=False)

    print(f"Saved truncated CSV to: {output_csv}")


if __name__ == "__main__":
    main()
