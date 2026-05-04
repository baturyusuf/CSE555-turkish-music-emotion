import argparse
import pandas as pd

from _common import (
    ensure_output_dirs,
    load_dataset,
    infer_target_column,
    get_numeric_feature_columns,
    REPORT_TABLE_DIR,
    REPORT_TEXT_DIR,
    save_json,
    save_text,
    print_section
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing CSV file.")
    parser.add_argument("--target", type=str, default=None, help="Target/class column name.")
    args = parser.parse_args()

    ensure_output_dirs()

    df, csv_path = load_dataset(csv_path=args.csv, data_dir=args.data_dir)
    target_col = infer_target_column(df, args.target)
    numeric_features = get_numeric_feature_columns(df, target_col)

    n_rows, n_cols = df.shape
    n_classes = df[target_col].nunique(dropna=True)
    duplicate_rows = df.duplicated().sum()

    print_section("DATASET INVENTORY")
    print(f"CSV path              : {csv_path}")
    print(f"Shape                 : {n_rows} rows x {n_cols} columns")
    print(f"Target column         : {target_col}")
    print(f"Number of classes     : {n_classes}")
    print(f"Numeric feature count : {len(numeric_features)}")
    print(f"Duplicate row count   : {duplicate_rows}")

    print_section("CLASS DISTRIBUTION")
    class_dist = (
        df[target_col]
        .value_counts(dropna=False)
        .rename_axis("class_label")
        .reset_index(name="count")
    )
    class_dist["percentage"] = class_dist["count"] / len(df) * 100
    print(class_dist.to_string(index=False))

    print_section("ASSIGNMENT REQUIREMENT CHECK")
    has_more_than_10_numeric = len(numeric_features) > 10
    has_more_than_2_classes = n_classes > 2

    print(f"More than 10 numerical features : {has_more_than_10_numeric}")
    print(f"More than 2 classes             : {has_more_than_2_classes}")

    if has_more_than_10_numeric and has_more_than_2_classes:
        print("Initial result: Dataset is compatible with the main assignment constraints.")
    else:
        print("Initial result: Dataset may not satisfy the assignment constraints.")

    dtypes_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[col].dtype) for col in df.columns],
        "n_unique": [df[col].nunique(dropna=True) for col in df.columns],
        "missing_count": [df[col].isna().sum() for col in df.columns],
        "missing_percentage": [df[col].isna().mean() * 100 for col in df.columns],
    })

    missing_df = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    missing_df["missing_percentage"] = missing_df["missing_count"] / len(df) * 100
    missing_df = missing_df.sort_values("missing_count", ascending=False)

    overview = {
        "csv_path": str(csv_path),
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "target_column": target_col,
        "n_classes": int(n_classes),
        "numeric_feature_count": int(len(numeric_features)),
        "duplicate_row_count": int(duplicate_rows),
        "more_than_10_numeric_features": bool(has_more_than_10_numeric),
        "more_than_2_classes": bool(has_more_than_2_classes),
        "numeric_features": numeric_features,
    }

    text_report = f"""
DATASET INVENTORY REPORT

CSV path: {csv_path}

Shape:
- Rows   : {n_rows}
- Columns: {n_cols}

Target column:
- {target_col}

Class count:
- {n_classes}

Numeric feature count:
- {len(numeric_features)}

Duplicate rows:
- {duplicate_rows}

Assignment compatibility:
- More than 10 numerical features: {has_more_than_10_numeric}
- More than 2 classes: {has_more_than_2_classes}

Class distribution:
{class_dist.to_string(index=False)}
"""

    dtypes_df.to_csv(REPORT_TABLE_DIR / "01_column_dtypes_and_missingness.csv", index=False)
    missing_df.to_csv(REPORT_TABLE_DIR / "01_missing_values.csv", index=False)
    class_dist.to_csv(REPORT_TABLE_DIR / "01_class_distribution.csv", index=False)
    df.head(10).to_csv(REPORT_TABLE_DIR / "01_dataset_head.csv", index=False)

    save_json(overview, REPORT_TEXT_DIR / "01_dataset_overview.json")
    save_text(text_report, REPORT_TEXT_DIR / "01_dataset_inventory_report.txt")

    print_section("FILES SAVED")
    print(REPORT_TABLE_DIR / "01_column_dtypes_and_missingness.csv")
    print(REPORT_TABLE_DIR / "01_missing_values.csv")
    print(REPORT_TABLE_DIR / "01_class_distribution.csv")
    print(REPORT_TABLE_DIR / "01_dataset_head.csv")
    print(REPORT_TEXT_DIR / "01_dataset_overview.json")
    print(REPORT_TEXT_DIR / "01_dataset_inventory_report.txt")


if __name__ == "__main__":
    main()
