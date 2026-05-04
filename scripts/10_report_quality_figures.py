import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.advanced_common import (
    ensure_dirs, configure_plots, load_dataset, infer_target_column, prepared_variant,
    multiclass_fisher_score, pairwise_fisher_table, pca_embedding, lda_embedding,
    pca_summary_table, save_scatter, scatter_by_class, OUTPUT_TABLE_DIR, OUTPUT_FIG_DIR,
    OUTPUT_TEXT_DIR, save_text, print_section
)

def classwise_boxplot(df, feature, target_col, ax):
    classes = sorted(df[target_col].astype(str).unique())
    values = [df[df[target_col].astype(str) == cls][feature].dropna().values for cls in classes]
    ax.boxplot(values, tick_labels=classes, showfliers=True)
    ax.set_title(feature)
    ax.set_xlabel("Class")
    ax.set_ylabel("Feature Value")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    ensure_dirs()
    configure_plots()

    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    data = prepared_variant(df_raw, target_col, "duplicate_cleaned_iqr_capped")
    df = data["df"]
    X_scaled = data["X_scaled"]
    X_scaled_df = data["X_scaled_df"]
    y = data["y"]

    fisher_df, fisher_details = multiclass_fisher_score(X_scaled_df, y)
    pairwise_df = pairwise_fisher_table(X_scaled_df, y)
    top_features = fisher_df.head(args.top_n)["feature"].tolist()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for ax, feature in zip(axes, top_features[:8]):
        classwise_boxplot(df, feature, target_col, ax)
    for ax in axes[len(top_features[:8]):]:
        ax.axis("off")
    fig.suptitle("Class-Wise Boxplots of the Most Discriminative Acoustic Features", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "10_classwise_boxplots_top_fisher_features.png")
    plt.close(fig)

    top_outliers = data["cap_summary"].sort_values("total_capped_values", ascending=False).head(10)["feature"].tolist()
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()
    for ax, feature in zip(axes, top_outliers):
        classwise_boxplot(df, feature, target_col, ax)
    fig.suptitle("Class-Wise Boxplots of Features Most Affected by IQR-Based Capping", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "10_classwise_boxplots_top_outlier_features.png")
    plt.close(fig)

    pca, X_pca_df = pca_embedding(X_scaled)
    pca_summary = pca_summary_table(pca)
    pc_fisher_df, _ = multiclass_fisher_score(X_pca_df, y)
    pc_overview = pca_summary.merge(
        pc_fisher_df.rename(columns={"feature": "principal_component"}),
        on="principal_component",
        how="left"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pc_overview["eigenvalue"], pc_overview["multiclass_fisher_score"], s=46, alpha=0.8)
    for _, row in pc_overview.head(8).iterrows():
        ax.annotate(row["principal_component"], (row["eigenvalue"], row["multiclass_fisher_score"]), fontsize=9)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Multi-Class Fisher Score of Projected Feature")
    ax.set_title("Relationship Between PCA Eigenvalues and Fisher Scores")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "10_pca_eigenvalue_vs_projected_fisher_score.png")
    plt.close(fig)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=data["features"],
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    selected_pcs = ["PC1", "PC2", "PC3", "PC4", "PC5"]
    selected_features = (
        loadings[selected_pcs]
        .abs()
        .sum(axis=1)
        .sort_values(ascending=False)
        .head(20)
        .index
        .tolist()
    )
    loading_subset = loadings.loc[selected_features, selected_pcs]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(loading_subset.values, aspect="auto", vmin=-loading_subset.abs().max().max(), vmax=loading_subset.abs().max().max())
    ax.set_xticks(range(len(selected_pcs)))
    ax.set_xticklabels(selected_pcs)
    ax.set_yticks(range(len(selected_features)))
    ax.set_yticklabels(selected_features, fontsize=8)
    ax.set_title("PCA Loading Heatmap for the First Five Components")
    fig.colorbar(im, ax=ax, label="Loading")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "10_pca_loading_heatmap_first_five_components.png")
    plt.close(fig)

    pca2_df = pd.DataFrame({"PC1": X_pca_df["PC1"], "PC2": X_pca_df["PC2"], target_col: y.values})
    lda, X_lda_df = lda_embedding(X_scaled, y, n_components=2)
    lda2_df = pd.DataFrame({"LD1": X_lda_df["LD1"], "LD2": X_lda_df["LD2"], target_col: y.values})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    scatter_by_class(axes[0], pca2_df, "PC1", "PC2", target_col, "PCA 2D Projection", "PC1", "PC2")
    scatter_by_class(axes[1], lda2_df, "LD1", "LD2", target_col, "LDA 2D Projection", "LD1", "LD2")
    fig.suptitle("PCA and LDA Projection Comparison", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "10_pca_vs_lda_side_by_side.png")
    plt.close(fig)

    focused_features = fisher_df.head(15)["feature"].tolist()
    corr = df[focused_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(focused_features)))
    ax.set_xticklabels(focused_features, rotation=90, fontsize=8)
    ax.set_yticks(range(len(focused_features)))
    ax.set_yticklabels(focused_features, fontsize=8)
    ax.set_title("Correlation Heatmap of the Most Discriminative Features")
    fig.colorbar(im, ax=ax, label="Pearson Correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "10_focused_discriminative_feature_correlation_heatmap.png")
    plt.close(fig)

    top_20 = fisher_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_20["feature"], top_20["multiclass_fisher_score"])
    ax.set_xlabel("Multi-Class Fisher Score")
    ax.set_ylabel("Feature")
    ax.set_title("Top 20 Features by Multi-Class Fisher Score")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "10_top_20_fisher_score_bar_chart.png")
    plt.close(fig)

    data["cap_summary"].to_csv(OUTPUT_TABLE_DIR / "10_iqr_capping_summary_used_for_figures.csv", index=False)
    fisher_df.to_csv(OUTPUT_TABLE_DIR / "10_fisher_scores_used_for_figures.csv", index=False)
    fisher_details.to_csv(OUTPUT_TABLE_DIR / "10_fisher_manual_contributions_all_features.csv", index=False)
    pairwise_df.to_csv(OUTPUT_TABLE_DIR / "10_pairwise_fisher_distance_table.csv", index=False)
    pc_overview.to_csv(OUTPUT_TABLE_DIR / "10_pca_eigenvalue_fisher_overview.csv", index=False)
    loadings.to_csv(OUTPUT_TABLE_DIR / "10_pca_loadings_full.csv")
    loading_subset.to_csv(OUTPUT_TABLE_DIR / "10_pca_loading_heatmap_values.csv")

    text = f"""
Report-Quality Figure Production

CSV Path: {csv_path}
Target Column: {target_col}
Rows Used: {len(df)}
Top Fisher Features Used for Class-Wise Boxplots:
{pd.Series(top_features).to_string(index=False)}

Top Outlier-Affected Features Used for Class-Wise Boxplots:
{pd.Series(top_outliers).to_string(index=False)}

PCA Eigenvalue-Fisher Overview:
{pc_overview.head(12).to_string(index=False)}
"""
    save_text(text.strip() + "\n", OUTPUT_TEXT_DIR / "10_report_quality_figures_summary.txt")

    print_section("10 REPORT-QUALITY FIGURES")
    print(text)

if __name__ == "__main__":
    main()
