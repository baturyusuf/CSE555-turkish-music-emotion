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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.advanced_common import (
    ensure_dirs, configure_plots, load_dataset, infer_target_column, prepared_variant,
    multiclass_fisher_score, pca_embedding, lda_embedding, pca_summary_table,
    components_needed, separability_metrics, OUTPUT_TABLE_DIR, OUTPUT_FIG_DIR,
    OUTPUT_TEXT_DIR, save_text, print_section
)

VARIANTS = ["raw", "duplicate_cleaned", "duplicate_cleaned_iqr_capped"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    configure_plots()

    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)

    overview_rows = []
    fisher_rows = []
    pca_rows = []
    threshold_rows = []
    embedding_rows = []

    for variant in VARIANTS:
        data = prepared_variant(df_raw, target_col, variant)
        X_scaled = data["X_scaled"]
        X_scaled_df = data["X_scaled_df"]
        y = data["y"]

        class_counts = y.value_counts().sort_index()
        overview_rows.append({
            "variant": variant,
            "rows": len(y),
            "features": X_scaled.shape[1],
            "class_count": y.nunique(),
            "min_class_count": int(class_counts.min()),
            "max_class_count": int(class_counts.max()),
            "total_capped_values": 0 if data["cap_summary"].empty else int(data["cap_summary"]["total_capped_values"].sum()),
        })

        fisher_df, _ = multiclass_fisher_score(X_scaled_df, y)
        fisher_df["rank"] = np.arange(1, len(fisher_df) + 1)
        fisher_df["variant"] = variant
        fisher_rows.append(fisher_df)

        pca, X_pca_df = pca_embedding(X_scaled)
        summary = pca_summary_table(pca)
        summary["variant"] = variant
        pca_rows.append(summary)
        needed = components_needed(summary)
        needed["variant"] = variant
        threshold_rows.append(needed)

        pca2 = X_pca_df[["PC1", "PC2"]].values
        lda, X_lda_df = lda_embedding(X_scaled, y, n_components=2)
        lda2 = X_lda_df[["LD1", "LD2"]].values

        embedding_rows.append({"variant": variant, "embedding": "PCA_2D", **separability_metrics(pca2, y)})
        embedding_rows.append({"variant": variant, "embedding": "LDA_2D", **separability_metrics(lda2, y)})

    overview_df = pd.DataFrame(overview_rows)
    fisher_all = pd.concat(fisher_rows, ignore_index=True)
    pca_all = pd.concat(pca_rows, ignore_index=True)
    thresholds_all = pd.concat(threshold_rows, ignore_index=True)
    embedding_df = pd.DataFrame(embedding_rows)

    top_features = (
        fisher_all[fisher_all["variant"] == "duplicate_cleaned_iqr_capped"]
        .head(12)["feature"]
        .tolist()
    )

    stability = (
        fisher_all[fisher_all["feature"].isin(top_features)]
        .pivot_table(index="feature", columns="variant", values=["rank", "multiclass_fisher_score"])
    )
    stability.columns = [f"{a}_{b}" for a, b in stability.columns]
    stability = stability.reset_index()

    pca_top = pca_all[pca_all["principal_component"].isin(["PC1", "PC2", "PC3", "PC4", "PC5"])].copy()

    overview_df.to_csv(OUTPUT_TABLE_DIR / "09_dataset_variant_overview.csv", index=False)
    fisher_all.to_csv(OUTPUT_TABLE_DIR / "09_fisher_scores_by_dataset_variant.csv", index=False)
    pca_all.to_csv(OUTPUT_TABLE_DIR / "09_pca_summary_by_dataset_variant.csv", index=False)
    thresholds_all.to_csv(OUTPUT_TABLE_DIR / "09_components_needed_by_dataset_variant.csv", index=False)
    embedding_df.to_csv(OUTPUT_TABLE_DIR / "09_pca_lda_separability_by_dataset_variant.csv", index=False)
    stability.to_csv(OUTPUT_TABLE_DIR / "09_fisher_rank_stability_top_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = fisher_all[fisher_all["feature"].isin(top_features)].copy()
    x = np.arange(len(top_features))
    width = 0.25
    for i, variant in enumerate(VARIANTS):
        vals = []
        for feature in top_features:
            vals.append(plot_data[(plot_data["variant"] == variant) & (plot_data["feature"] == feature)]["multiclass_fisher_score"].iloc[0])
        ax.bar(x + (i - 1) * width, vals, width=width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=75, ha="right")
    ax.set_ylabel("Multi-Class Fisher Score")
    ax.set_title("Fisher Score Stability Across Dataset Variants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "09_fisher_score_stability_across_variants.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANTS:
        part = pca_all[pca_all["variant"] == variant].head(20)
        ax.plot(
            np.arange(1, len(part) + 1),
            part["cumulative_explained_variance_ratio"],
            marker="o",
            label=variant
        )
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA Cumulative Explained Variance Across Dataset Variants")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "09_pca_cumulative_variance_variant_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    metric_name = "silhouette_score"
    order = ["PCA_2D", "LDA_2D"]
    x = np.arange(len(VARIANTS))
    width = 0.35
    for i, emb in enumerate(order):
        vals = [embedding_df[(embedding_df["variant"] == v) & (embedding_df["embedding"] == emb)][metric_name].iloc[0] for v in VARIANTS]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=emb)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, rotation=20, ha="right")
    ax.set_ylabel("Silhouette Score Based on True Class Labels")
    ax.set_title("PCA vs LDA Separability Under Data-Cleaning Variants")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "09_pca_lda_silhouette_variant_comparison.png")
    plt.close(fig)

    text = f"""
Sensitivity Analysis

CSV Path: {csv_path}
Target Column: {target_col}

Dataset Variant Overview
{overview_df.to_string(index=False)}

Components Needed for Variance Thresholds
{thresholds_all.to_string(index=False)}

PCA/LDA Separability by Dataset Variant
{embedding_df.to_string(index=False)}

Top Feature Fisher Rank Stability
{stability.head(20).to_string(index=False)}
"""
    save_text(text.strip() + "\n", OUTPUT_TEXT_DIR / "09_sensitivity_analysis_summary.txt")

    print_section("09 SENSITIVITY ANALYSIS")
    print(text)

if __name__ == "__main__":
    main()
