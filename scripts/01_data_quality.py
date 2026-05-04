import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.common import ensure_dirs, load_dataset, infer_target_column, get_numeric_features, OUTPUT_FIG_DIR, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, DATA_INTERIM_DIR, save_text, save_json, print_section, remove_duplicates, iqr_cap_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    numeric_features = get_numeric_features(df_raw, target_col)

    df_nodup = remove_duplicates(df_raw)
    df_clean, cap_summary = iqr_cap_dataframe(df_nodup, numeric_features)

    class_dist_raw = df_raw[target_col].value_counts().rename_axis('class').reset_index(name='count')
    class_dist_clean = df_clean[target_col].value_counts().rename_axis('class').reset_index(name='count')

    overview = {
        'csv_path': str(csv_path),
        'raw_rows': int(len(df_raw)),
        'raw_columns': int(df_raw.shape[1]),
        'target_column': target_col,
        'class_count': int(df_raw[target_col].nunique()),
        'numeric_feature_count': int(len(numeric_features)),
        'missing_values_total': int(df_raw.isna().sum().sum()),
        'duplicate_rows_removed': int(len(df_raw) - len(df_nodup)),
        'rows_after_duplicate_removal': int(len(df_nodup)),
        'rows_after_outlier_capping': int(len(df_clean)),
        'total_cells_capped': int(cap_summary['total_capped_values'].sum()),
    }

    summary_df = pd.DataFrame({
        'stage': ['raw', 'after_duplicate_removal', 'after_iqr_capping'],
        'rows': [len(df_raw), len(df_nodup), len(df_clean)],
        'columns': [df_raw.shape[1], df_nodup.shape[1], df_clean.shape[1]],
    })

    duplicate_groups = df_raw[df_raw.duplicated(keep=False)].sort_values(by=df_raw.columns.tolist()).copy()
    duplicate_class_dist = duplicate_groups[target_col].value_counts().rename_axis('class').reset_index(name='duplicate_row_count') if not duplicate_groups.empty else pd.DataFrame(columns=['class', 'duplicate_row_count'])

    dtypes_df = pd.DataFrame({
        'column': df_raw.columns,
        'dtype': [str(df_raw[c].dtype) for c in df_raw.columns],
        'missing_count': [int(df_raw[c].isna().sum()) for c in df_raw.columns],
        'unique_count': [int(df_raw[c].nunique(dropna=True)) for c in df_raw.columns],
    })

    plt.figure(figsize=(14, max(8, len(numeric_features) * 0.24)))
    plt.boxplot([df_raw[c].dropna() for c in numeric_features], labels=numeric_features, vert=False, showfliers=True)
    plt.title('Boxplot of Raw Numerical Features')
    plt.xlabel('Feature Value')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '01_boxplot_raw_features.png', dpi=300)
    plt.close()

    plt.figure(figsize=(14, max(8, len(numeric_features) * 0.24)))
    plt.boxplot([df_clean[c].dropna() for c in numeric_features], labels=numeric_features, vert=False, showfliers=True)
    plt.title('Boxplot After Duplicate Removal and IQR-Based Capping')
    plt.xlabel('Feature Value')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '01_boxplot_cleaned_features.png', dpi=300)
    plt.close()

    df_clean.to_csv(DATA_INTERIM_DIR / 'analysis_ready_dataset.csv', index=False)
    summary_df.to_csv(OUTPUT_TABLE_DIR / '01_data_quality_overview.csv', index=False)
    class_dist_raw.to_csv(OUTPUT_TABLE_DIR / '01_class_distribution_raw.csv', index=False)
    class_dist_clean.to_csv(OUTPUT_TABLE_DIR / '01_class_distribution_clean.csv', index=False)
    dtypes_df.to_csv(OUTPUT_TABLE_DIR / '01_column_profile.csv', index=False)
    cap_summary.to_csv(OUTPUT_TABLE_DIR / '01_iqr_capping_summary.csv', index=False)
    duplicate_class_dist.to_csv(OUTPUT_TABLE_DIR / '01_duplicate_class_distribution.csv', index=False)
    if not duplicate_groups.empty:
        duplicate_groups.to_csv(OUTPUT_TABLE_DIR / '01_duplicate_rows.csv', index=False)

    text = f"""
Data Quality Summary

CSV Path: {csv_path}
Target Column: {target_col}
Rows: {len(df_raw)}
Columns: {df_raw.shape[1]}
Numeric Features: {len(numeric_features)}
Class Count: {df_raw[target_col].nunique()}
Missing Values (total): {df_raw.isna().sum().sum()}
Duplicate Rows Removed: {len(df_raw) - len(df_nodup)}
Rows After Duplicate Removal: {len(df_nodup)}
Rows After IQR-Based Capping: {len(df_clean)}
Total Capped Cells: {int(cap_summary['total_capped_values'].sum())}

Raw Class Distribution
{class_dist_raw.to_string(index=False)}

Clean Class Distribution
{class_dist_clean.to_string(index=False)}

Top 15 Features by Number of Capped Values
{cap_summary.sort_values('total_capped_values', ascending=False).head(15)[['feature', 'total_capped_values', 'below_lower_before', 'above_upper_before']].to_string(index=False)}
"""

    save_json(overview, OUTPUT_TEXT_DIR / '01_data_quality_overview.json')
    save_text(text.strip() + '\n', OUTPUT_TEXT_DIR / '01_data_quality_summary.txt')

    print_section('01 DATA QUALITY')
    print(text)


if __name__ == '__main__':
    main()
