import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder

from _common import (
    ensure_output_dirs,
    load_dataset,
    infer_target_column,
    get_numeric_feature_columns,
    REPORT_TABLE_DIR,
    REPORT_TEXT_DIR,
    REPORT_FIGURE_DIR,
    save_text,
    print_section
)


def compute_iqr_outliers(X):
    rows = []

    for col in X.columns:
        s = X[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (X[col] < lower) | (X[col] > upper)
        count = int(mask.sum())

        rows.append({
            "feature": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "iqr_outlier_count": count,
            "iqr_outlier_percentage": count / len(X) * 100
        })

    return pd.DataFrame(rows)


def compute_zscore_outliers(X, threshold=3.0):
    X_numeric = X.copy()
    means = X_numeric.mean()
    stds = X_numeric.std(ddof=0).replace(0, np.nan)

    Z = (X_numeric - means) / stds
    mask = Z.abs() > threshold

    rows = []

    for col in X.columns:
        count = int(mask[col].sum())
        rows.append({
            "feature": col,
            "z_threshold": threshold,
            "zscore_outlier_count": count,
            "zscore_outlier_percentage": count / len(X) * 100
        })

    row_outlier_counts = mask.sum(axis=1)

    return pd.DataFrame(rows), row_outlier_counts


def save_standardized_boxplot(X):
    Xz = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    Xz = Xz.replace([np.inf, -np.inf], np.nan)

    plt.figure(figsize=(14, max(8, len(X.columns) * 0.25)))
    plt.boxplot(
        [Xz[col].dropna() for col in Xz.columns],
        labels=Xz.columns,
        vert=False,
        showfliers=True
    )
    plt.title("Standardized Boxplot of Numerical Features")
    plt.xlabel("Z-score")
    plt.tight_layout()
    plt.savefig(REPORT_FIGURE_DIR / "03_standardized_boxplot_all_features.png", dpi=300)
    plt.close()


def save_correlation_heatmap(corr):
    plt.figure(figsize=(14, 12))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.title("Feature-Feature Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORT_FIGURE_DIR / "03_feature_correlation_heatmap.png", dpi=300)
    plt.close()


def get_top_abs_correlations(corr, top_n=30):
    corr_abs = corr.abs()
    upper_mask = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)

    pairs = (
        corr_abs.where(upper_mask)
        .stack()
        .reset_index()
    )

    pairs.columns = ["feature_1", "feature_2", "abs_correlation"]
    pairs["correlation"] = [
        corr.loc[row["feature_1"], row["feature_2"]]
        for _, row in pairs.iterrows()
    ]

    return pairs.sort_values("abs_correlation", ascending=False).head(top_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    args = parser.parse_args()

    ensure_output_dirs()

    df, csv_path = load_dataset(csv_path=args.csv, data_dir=args.data_dir)
    target_col = infer_target_column(df, args.target)
    numeric_features = get_numeric_feature_columns(df, target_col)

    X = df[numeric_features].copy()
    y = df[target_col].copy()

    print_section("OUTLIER AND CORRELATION PROBE")
    print(f"CSV path              : {csv_path}")
    print(f"Target column         : {target_col}")
    print(f"Numeric feature count : {len(numeric_features)}")

    iqr_outliers = compute_iqr_outliers(X)
    z_outliers, row_outlier_counts = compute_zscore_outliers(X, threshold=args.z_threshold)

    outlier_summary = iqr_outliers.merge(z_outliers, on="feature", how="left")
    outlier_summary = outlier_summary.sort_values(
        ["iqr_outlier_count", "zscore_outlier_count"],
        ascending=False
    )

    row_outlier_summary = pd.DataFrame({
        "row_index": np.arange(len(X)),
        "zscore_outlier_feature_count": row_outlier_counts
    }).sort_values("zscore_outlier_feature_count", ascending=False)

    corr = X.corr(method="pearson")
    top_corr_pairs = get_top_abs_correlations(corr, top_n=30)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    X_for_f = X.copy()
    X_for_f = X_for_f.fillna(X_for_f.median(numeric_only=True))

    f_values, p_values = f_classif(X_for_f, y_encoded)

    feature_label_assoc = pd.DataFrame({
        "feature": numeric_features,
        "anova_f_value": f_values,
        "anova_p_value": p_values
    }).sort_values("anova_f_value", ascending=False)

    save_standardized_boxplot(X)
    save_correlation_heatmap(corr)

    outlier_summary.to_csv(REPORT_TABLE_DIR / "03_outlier_summary_iqr_zscore.csv", index=False)
    row_outlier_summary.to_csv(REPORT_TABLE_DIR / "03_row_level_zscore_outlier_counts.csv", index=False)
    corr.to_csv(REPORT_TABLE_DIR / "03_feature_correlation_matrix.csv")
    top_corr_pairs.to_csv(REPORT_TABLE_DIR / "03_top_30_absolute_feature_correlations.csv", index=False)
    feature_label_assoc.to_csv(REPORT_TABLE_DIR / "03_feature_label_anova_scores.csv", index=False)

    print_section("TOP 15 FEATURES BY IQR OUTLIER COUNT")
    print(
        outlier_summary[
            ["feature", "iqr_outlier_count", "iqr_outlier_percentage", "zscore_outlier_count"]
        ].head(15).to_string(index=False)
    )

    print_section("TOP 15 FEATURE-CLASS ASSOCIATIONS BY ANOVA F-VALUE")
    print(
        feature_label_assoc[
            ["feature", "anova_f_value", "anova_p_value"]
        ].head(15).to_string(index=False)
    )

    print_section("TOP 15 ABSOLUTE FEATURE CORRELATIONS")
    print(top_corr_pairs.head(15).to_string(index=False))

    report = f"""
OUTLIER AND CORRELATION PROBE REPORT

CSV path:
{csv_path}

Target column:
{target_col}

Numeric feature count:
{len(numeric_features)}

Top 15 features by IQR outlier count:
{outlier_summary[["feature", "iqr_outlier_count", "iqr_outlier_percentage", "zscore_outlier_count"]].head(15).to_string(index=False)}

Top 15 feature-class associations by ANOVA F-value:
{feature_label_assoc[["feature", "anova_f_value", "anova_p_value"]].head(15).to_string(index=False)}

Top 15 absolute feature-feature correlations:
{top_corr_pairs.head(15).to_string(index=False)}
"""

    save_text(report, REPORT_TEXT_DIR / "03_outlier_and_correlation_probe_report.txt")

    print_section("FILES SAVED")
    print(REPORT_TABLE_DIR / "03_outlier_summary_iqr_zscore.csv")
    print(REPORT_TABLE_DIR / "03_row_level_zscore_outlier_counts.csv")
    print(REPORT_TABLE_DIR / "03_feature_correlation_matrix.csv")
    print(REPORT_TABLE_DIR / "03_top_30_absolute_feature_correlations.csv")
    print(REPORT_TABLE_DIR / "03_feature_label_anova_scores.csv")
    print(REPORT_FIGURE_DIR / "03_standardized_boxplot_all_features.png")
    print(REPORT_FIGURE_DIR / "03_feature_correlation_heatmap.png")
    print(REPORT_TEXT_DIR / "03_outlier_and_correlation_probe_report.txt")


if __name__ == "__main__":
    main()
