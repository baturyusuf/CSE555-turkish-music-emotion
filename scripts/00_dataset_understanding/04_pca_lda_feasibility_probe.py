import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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


def multiclass_fisher_score(X_df, y):
    """
    Multi-class Fisher score:
    between-class scatter / within-class scatter

    For each feature:
    numerator   = sum_c n_c * (mu_c - mu)^2
    denominator = sum_c sum_i_in_c (x_i - mu_c)^2
    """

    y_series = pd.Series(y).reset_index(drop=True)
    X = X_df.reset_index(drop=True)

    scores = {}

    for feature in X.columns:
        x = X[feature]
        overall_mean = x.mean()

        numerator = 0.0
        denominator = 0.0

        for cls in y_series.unique():
            cls_mask = y_series == cls
            x_cls = x[cls_mask]

            n_c = len(x_cls)
            mean_c = x_cls.mean()

            numerator += n_c * (mean_c - overall_mean) ** 2
            denominator += ((x_cls - mean_c) ** 2).sum()

        scores[feature] = numerator / (denominator + 1e-12)

    return pd.Series(scores).sort_values(ascending=False)


def save_projection_scatter(proj_df, x_col, y_col, label_col, title, filename):
    plt.figure(figsize=(8, 6))

    for cls in sorted(proj_df[label_col].unique()):
        subset = proj_df[proj_df[label_col] == cls]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=30,
            alpha=0.75,
            label=str(cls)
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_FIGURE_DIR / filename, dpi=300)
    plt.close()


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
    y = df[target_col].copy()

    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_features)

    print_section("PCA / LDA FEASIBILITY PROBE")
    print(f"CSV path              : {csv_path}")
    print(f"Target column         : {target_col}")
    print(f"Numeric feature count : {len(numeric_features)}")
    print(f"Class count           : {y.nunique()}")

    original_fisher = multiclass_fisher_score(X_scaled_df, y)
    original_fisher_df = original_fisher.reset_index()
    original_fisher_df.columns = ["feature", "multiclass_fisher_score"]

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    pc_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pc_cols)

    pca_fisher = multiclass_fisher_score(X_pca_df, y)
    pca_fisher_df = pca_fisher.reset_index()
    pca_fisher_df.columns = ["principal_component", "multiclass_fisher_score"]

    explained_variance_df = pd.DataFrame({
        "principal_component": pc_cols,
        "eigenvalue": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_)
    })

    pca_overview = explained_variance_df.merge(
        pca_fisher_df,
        on="principal_component",
        how="left"
    )

    pearson_relation = pca_overview["eigenvalue"].corr(
        pca_overview["multiclass_fisher_score"],
        method="pearson"
    )

    spearman_relation = pca_overview["eigenvalue"].corr(
        pca_overview["multiclass_fisher_score"],
        method="spearman"
    )

    pca_projection_2d = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        target_col: y.values
    })

    pca_projection_last2 = pd.DataFrame({
        f"PC{X_pca.shape[1]-1}": X_pca[:, -2],
        f"PC{X_pca.shape[1]}": X_pca[:, -1],
        target_col: y.values
    })

    save_projection_scatter(
        pca_projection_2d,
        "PC1",
        "PC2",
        target_col,
        "PCA Projection: First Two Principal Components",
        "04_pca_first_two_components_scatter.png"
    )

    save_projection_scatter(
        pca_projection_last2,
        f"PC{X_pca.shape[1]-1}",
        f"PC{X_pca.shape[1]}",
        target_col,
        "PCA Projection: Last Two Principal Components",
        "04_pca_last_two_components_scatter.png"
    )

    n_classes = y.nunique()
    max_lda_components = min(n_classes - 1, len(numeric_features))

    lda_status = ""

    if max_lda_components >= 2:
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(X_scaled, y)

        lda_projection_2d = pd.DataFrame({
            "LD1": X_lda[:, 0],
            "LD2": X_lda[:, 1],
            target_col: y.values
        })

        lda_projection_2d.to_csv(REPORT_TABLE_DIR / "04_lda_2d_projection.csv", index=False)

        save_projection_scatter(
            lda_projection_2d,
            "LD1",
            "LD2",
            target_col,
            "LDA Projection: First Two Linear Discriminants",
            "04_lda_2d_projection_scatter.png"
        )

        lda_status = "LDA 2D projection was successfully generated."

    else:
        lda_status = (
            "LDA 2D projection could not be generated because "
            "the number of classes is not sufficient."
        )

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        marker="o"
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_FIGURE_DIR / "04_pca_cumulative_explained_variance.png", dpi=300)
    plt.close()

    original_fisher_df.to_csv(REPORT_TABLE_DIR / "04_original_feature_fisher_scores.csv", index=False)
    pca_fisher_df.to_csv(REPORT_TABLE_DIR / "04_pca_component_fisher_scores.csv", index=False)
    explained_variance_df.to_csv(REPORT_TABLE_DIR / "04_pca_explained_variance.csv", index=False)
    pca_overview.to_csv(REPORT_TABLE_DIR / "04_pca_eigenvalue_fisher_overview.csv", index=False)
    pca_projection_2d.to_csv(REPORT_TABLE_DIR / "04_pca_first_two_projection.csv", index=False)
    pca_projection_last2.to_csv(REPORT_TABLE_DIR / "04_pca_last_two_projection.csv", index=False)

    print_section("TOP 15 ORIGINAL FEATURES BY MULTI-CLASS FISHER SCORE")
    print(original_fisher_df.head(15).to_string(index=False))

    print_section("TOP 15 PCA COMPONENTS BY MULTI-CLASS FISHER SCORE")
    print(pca_fisher_df.head(15).to_string(index=False))

    print_section("PCA EIGENVALUE - FISHER SCORE RELATION")
    print(f"Pearson correlation  : {pearson_relation}")
    print(f"Spearman correlation : {spearman_relation}")

    print_section("LDA FEASIBILITY")
    print(f"Max LDA components: {max_lda_components}")
    print(lda_status)

    report = f"""
PCA / LDA FEASIBILITY PROBE REPORT

CSV path:
{csv_path}

Target column:
{target_col}

Numeric feature count:
{len(numeric_features)}

Class count:
{n_classes}

Max possible LDA components:
{max_lda_components}

LDA status:
{lda_status}

PCA eigenvalue - Fisher score relation:
- Pearson correlation : {pearson_relation}
- Spearman correlation: {spearman_relation}

Top 15 original features by multi-class Fisher score:
{original_fisher_df.head(15).to_string(index=False)}

Top 15 PCA components by multi-class Fisher score:
{pca_fisher_df.head(15).to_string(index=False)}
"""

    save_text(report, REPORT_TEXT_DIR / "04_pca_lda_feasibility_probe_report.txt")

    print_section("FILES SAVED")
    print(REPORT_TABLE_DIR / "04_original_feature_fisher_scores.csv")
    print(REPORT_TABLE_DIR / "04_pca_component_fisher_scores.csv")
    print(REPORT_TABLE_DIR / "04_pca_explained_variance.csv")
    print(REPORT_TABLE_DIR / "04_pca_eigenvalue_fisher_overview.csv")
    print(REPORT_TABLE_DIR / "04_pca_first_two_projection.csv")
    print(REPORT_TABLE_DIR / "04_pca_last_two_projection.csv")
    print(REPORT_FIGURE_DIR / "04_pca_first_two_components_scatter.png")
    print(REPORT_FIGURE_DIR / "04_pca_last_two_components_scatter.png")
    print(REPORT_FIGURE_DIR / "04_pca_cumulative_explained_variance.png")
    print(REPORT_TEXT_DIR / "04_pca_lda_feasibility_probe_report.txt")


if __name__ == "__main__":
    main()
