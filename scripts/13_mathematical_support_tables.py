import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.advanced_common import (
    ensure_dirs, configure_plots, load_dataset, infer_target_column, prepared_variant,
    multiclass_fisher_score, pairwise_fisher_table, pca_embedding, pca_summary_table,
    lda_embedding, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, save_text, print_section
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--feature", type=str, default=None)
    parser.add_argument("--n-cov-features", type=int, default=6)
    args = parser.parse_args()

    ensure_dirs()
    configure_plots()

    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    data = prepared_variant(df_raw, target_col, "duplicate_cleaned_iqr_capped")
    X_scaled = data["X_scaled"]
    X_scaled_df = data["X_scaled_df"]
    y = data["y"]

    fisher_df, fisher_details = multiclass_fisher_score(X_scaled_df, y)
    feature = args.feature or fisher_df.iloc[0]["feature"]

    feature_contrib = fisher_details[fisher_details["feature"] == feature].copy()
    feature_score = fisher_df[fisher_df["feature"] == feature]["multiclass_fisher_score"].iloc[0]
    feature_contrib["selected_feature_fisher_score"] = feature_score

    pairwise_df = pairwise_fisher_table(X_scaled_df[[feature]], y)

    top_features = fisher_df.head(args.n_cov_features)["feature"].tolist()
    covariance_subset = X_scaled_df[top_features].cov()
    correlation_subset = X_scaled_df[top_features].corr()

    pca, X_pca_df = pca_embedding(X_scaled)
    pca_summary = pca_summary_table(pca)
    pc_fisher_df, _ = multiclass_fisher_score(X_pca_df, y)
    pc_overview = pca_summary.merge(
        pc_fisher_df.rename(columns={"feature": "principal_component"}),
        on="principal_component",
        how="left"
    )
    pc_overview["eigenvalue_rank"] = pc_overview["eigenvalue"].rank(ascending=False, method="first").astype(int)
    pc_overview["fisher_rank"] = pc_overview["multiclass_fisher_score"].rank(ascending=False, method="first").astype(int)

    loading_table = pd.DataFrame(
        pca.components_.T,
        index=data["features"],
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    selected_loading_rows = []
    for pc in ["PC1", "PC2", "PC3"]:
        temp = loading_table[pc].abs().sort_values(ascending=False).head(12).reset_index()
        temp.columns = ["feature", "absolute_loading"]
        temp["principal_component"] = pc
        temp["signed_loading"] = [loading_table.loc[f, pc] for f in temp["feature"]]
        selected_loading_rows.append(temp[["principal_component", "feature", "signed_loading", "absolute_loading"]])
    selected_loadings = pd.concat(selected_loading_rows, ignore_index=True)

    lda, X_lda_df = lda_embedding(X_scaled, y, n_components=2)
    lda_df = X_lda_df.copy()
    lda_df[target_col] = y.values
    lda_centroids = lda_df.groupby(target_col)[["LD1", "LD2"]].mean().reset_index()

    if hasattr(lda, "explained_variance_ratio_"):
        lda_variance = pd.DataFrame({
            "linear_discriminant": [f"LD{i+1}" for i in range(len(lda.explained_variance_ratio_))],
            "explained_discriminant_variance_ratio": lda.explained_variance_ratio_,
            "cumulative_ratio": np.cumsum(lda.explained_variance_ratio_),
        })
    else:
        lda_variance = pd.DataFrame()

    if hasattr(lda, "scalings_"):
        lda_coefficients = pd.DataFrame(
            lda.scalings_[:, :2],
            index=data["features"],
            columns=["LD1_coefficient", "LD2_coefficient"]
        )
        lda_coefficients["LD1_abs"] = lda_coefficients["LD1_coefficient"].abs()
        lda_coefficients["LD2_abs"] = lda_coefficients["LD2_coefficient"].abs()
        lda_top = pd.concat([
            lda_coefficients.sort_values("LD1_abs", ascending=False).head(12).assign(linear_discriminant="LD1"),
            lda_coefficients.sort_values("LD2_abs", ascending=False).head(12).assign(linear_discriminant="LD2"),
        ]).reset_index().rename(columns={"index": "feature"})
    else:
        lda_coefficients = pd.DataFrame()
        lda_top = pd.DataFrame()

    feature_contrib.to_csv(OUTPUT_TABLE_DIR / "13_fisher_manual_contribution_selected_feature.csv", index=False)
    pairwise_df.to_csv(OUTPUT_TABLE_DIR / "13_pairwise_fisher_selected_feature.csv", index=False)
    covariance_subset.to_csv(OUTPUT_TABLE_DIR / "13_covariance_matrix_top_discriminative_features.csv")
    correlation_subset.to_csv(OUTPUT_TABLE_DIR / "13_correlation_matrix_top_discriminative_features.csv")
    pc_overview.to_csv(OUTPUT_TABLE_DIR / "13_pca_eigenvalue_fisher_rank_table.csv", index=False)
    selected_loadings.to_csv(OUTPUT_TABLE_DIR / "13_pca_top_loadings_for_manual_interpretation.csv", index=False)
    lda_centroids.to_csv(OUTPUT_TABLE_DIR / "13_lda_class_centroids.csv", index=False)
    if not lda_variance.empty:
        lda_variance.to_csv(OUTPUT_TABLE_DIR / "13_lda_explained_discriminant_variance.csv", index=False)
    if not lda_top.empty:
        lda_top.to_csv(OUTPUT_TABLE_DIR / "13_lda_top_coefficients.csv", index=False)

    formulas = f"""
Mathematical Support Tables

Selected feature for Fisher manual demonstration:
{feature}

Multi-class Fisher Score:
J(feature) = Σ_c n_c(μ_c - μ)^2 / Σ_c Σ_i∈c (x_i - μ_c)^2

Selected Feature Fisher Score:
{feature_score}

Per-Class Contributions:
{feature_contrib.to_string(index=False)}

Pairwise Fisher Distances for the Selected Feature:
{pairwise_df.to_string(index=False)}

PCA Eigenvalue and Fisher Rank Relationship:
{pc_overview.head(20).to_string(index=False)}

Top PCA Loadings for PC1, PC2, and PC3:
{selected_loadings.to_string(index=False)}

LDA Class Centroids:
{lda_centroids.to_string(index=False)}
"""
    if not lda_variance.empty:
        formulas += f"\nLDA Explained Discriminant Variance:\n{lda_variance.to_string(index=False)}\n"
    if not lda_top.empty:
        formulas += f"\nTop LDA Coefficients:\n{lda_top.head(24).to_string(index=False)}\n"

    save_text(formulas.strip() + "\n", OUTPUT_TEXT_DIR / "13_mathematical_support_summary.txt")

    print_section("13 MATHEMATICAL SUPPORT TABLES")
    print(formulas)

if __name__ == "__main__":
    main()
