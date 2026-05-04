import argparse
import pandas as pd

from _common import (
    ensure_output_dirs,
    load_dataset,
    infer_target_column,
    get_numeric_feature_columns,
    REPORT_TABLE_DIR,
    REPORT_TEXT_DIR,
    save_text,
    print_section
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    ensure_output_dirs()

    df, csv_path = load_dataset(csv_path=args.csv, data_dir=args.data_dir)
    target_col = infer_target_column(df, args.target)
    numeric_features = get_numeric_feature_columns(df, target_col)

    X = df[numeric_features].copy()

    print_section("NUMERICAL FEATURE AUDIT")
    print(f"CSV path              : {csv_path}")
    print(f"Target column         : {target_col}")
    print(f"Numeric feature count : {len(numeric_features)}")

    summary = pd.DataFrame(index=numeric_features)
    summary["count"] = X.count()
    summary["missing_count"] = X.isna().sum()
    summary["missing_percentage"] = X.isna().mean() * 100
    summary["mean"] = X.mean()
    summary["std"] = X.std()
    summary["min"] = X.min()
    summary["q1"] = X.quantile(0.25)
    summary["median"] = X.median()
    summary["q3"] = X.quantile(0.75)
    summary["max"] = X.max()
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["skewness"] = X.skew()
    summary["kurtosis"] = X.kurtosis()
    summary["n_unique"] = X.nunique(dropna=True)
    summary["unique_ratio"] = summary["n_unique"] / len(X)

    summary["is_constant"] = summary["n_unique"] <= 1
    summary["is_quasi_constant_unique_ratio_lt_1pct"] = summary["unique_ratio"] < 0.01

    summary = summary.reset_index().rename(columns={"index": "feature"})

    classwise_mean = df.groupby(target_col)[numeric_features].mean().T
    classwise_std = df.groupby(target_col)[numeric_features].std().T

    classwise_mean.index.name = "feature"
    classwise_std.index.name = "feature"

    classwise_mean.to_csv(REPORT_TABLE_DIR / "02_classwise_feature_means.csv")
    classwise_std.to_csv(REPORT_TABLE_DIR / "02_classwise_feature_stds.csv")
    summary.to_csv(REPORT_TABLE_DIR / "02_numeric_feature_summary.csv", index=False)

    constant_features = summary.loc[summary["is_constant"], "feature"].tolist()
    quasi_constant_features = summary.loc[
        summary["is_quasi_constant_unique_ratio_lt_1pct"], "feature"
    ].tolist()

    top_skewed = summary.reindex(summary["skewness"].abs().sort_values(ascending=False).index).head(10)
    highest_variance = summary.sort_values("std", ascending=False).head(10)
    lowest_variance = summary.sort_values("std", ascending=True).head(10)

    print_section("FEATURE QUALITY WARNINGS")
    print(f"Constant features       : {constant_features}")
    print(f"Quasi-constant features : {quasi_constant_features}")

    print_section("TOP 10 MOST SKEWED FEATURES")
    print(top_skewed[["feature", "skewness", "kurtosis"]].to_string(index=False))

    print_section("TOP 10 HIGHEST STD FEATURES")
    print(highest_variance[["feature", "std", "min", "max"]].to_string(index=False))

    print_section("TOP 10 LOWEST STD FEATURES")
    print(lowest_variance[["feature", "std", "min", "max"]].to_string(index=False))

    report = f"""
NUMERICAL FEATURE AUDIT REPORT

CSV path:
{csv_path}

Target column:
{target_col}

Number of numerical features:
{len(numeric_features)}

Constant features:
{constant_features}

Quasi-constant features:
{quasi_constant_features}

Top 10 most skewed features:
{top_skewed[["feature", "skewness", "kurtosis"]].to_string(index=False)}

Top 10 highest standard deviation features:
{highest_variance[["feature", "std", "min", "max"]].to_string(index=False)}

Top 10 lowest standard deviation features:
{lowest_variance[["feature", "std", "min", "max"]].to_string(index=False)}
"""

    save_text(report, REPORT_TEXT_DIR / "02_feature_audit_report.txt")

    print_section("FILES SAVED")
    print(REPORT_TABLE_DIR / "02_numeric_feature_summary.csv")
    print(REPORT_TABLE_DIR / "02_classwise_feature_means.csv")
    print(REPORT_TABLE_DIR / "02_classwise_feature_stds.csv")
    print(REPORT_TEXT_DIR / "02_feature_audit_report.txt")


if __name__ == "__main__":
    main()
