import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TEXT_DIR = PROJECT_ROOT / "outputs" / "text"


def ensure_dirs():
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def find_csv(csv_path=None):
    if csv_path:
        path = Path(csv_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path

    csv_files = sorted(DATA_RAW_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found under data/raw")

    return csv_files[0]


def load_dataset(csv_path=None):
    path = find_csv(csv_path)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    return df, path


def infer_target_column(df, target=None):
    if target:
        if target not in df.columns:
            raise ValueError(f"Target column not found: {target}")
        return target

    candidates = ["Class", "class", "Label", "label", "Target", "target", "Emotion", "emotion"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError("Target column could not be inferred. Use --target Class")


def get_numeric_features(df, target_col):
    return [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]


def iqr_cap_dataframe(df, feature_cols):
    capped = df.copy()

    for col in feature_cols:
        q1 = capped[col].quantile(0.25)
        q3 = capped[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        capped[col] = capped[col].clip(lower=lower, upper=upper)

    return capped


def prepare_dataset(df, target_col):
    numeric_features = get_numeric_features(df, target_col)

    df_clean = df.drop_duplicates().reset_index(drop=True)
    df_clean = iqr_cap_dataframe(df_clean, numeric_features)

    X = df_clean[numeric_features].copy()
    y = df_clean[target_col].astype(str).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return df_clean, X_scaled, y, y_encoded, numeric_features, encoder


def run_tsne(X_scaled, perplexity, random_state):
    try:
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1500,
            random_state=random_state,
        )
        return model.fit_transform(X_scaled)
    except TypeError:
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate=200,
            n_iter=1500,
            random_state=random_state,
        )
        return model.fit_transform(X_scaled)


def plot_tsne_by_class(embedding_df, target_col, perplexity):
    plt.figure(figsize=(8, 6))

    for cls in sorted(embedding_df[target_col].unique()):
        subset = embedding_df[embedding_df[target_col] == cls]
        plt.scatter(
            subset["TSNE1"],
            subset["TSNE2"],
            s=42,
            alpha=0.78,
            label=cls,
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"t-SNE Projection by True Emotion Class | Perplexity={perplexity}")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_DIR / f"14_tsne_perplexity_{perplexity}_true_classes.png", dpi=300)
    plt.close()


def plot_tsne_by_kmeans(embedding_df, perplexity):
    plt.figure(figsize=(8, 6))

    for cluster_id in sorted(embedding_df["kmeans_cluster"].unique()):
        subset = embedding_df[embedding_df["kmeans_cluster"] == cluster_id]
        plt.scatter(
            subset["TSNE1"],
            subset["TSNE2"],
            s=42,
            alpha=0.78,
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"K-Means Clusters on t-SNE Embedding | Perplexity={perplexity}")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_DIR / f"14_tsne_kmeans_perplexity_{perplexity}.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--selected-perplexity", type=int, default=30)
    args = parser.parse_args()

    ensure_dirs()

    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)

    df_clean, X_scaled, y, y_encoded, numeric_features, encoder = prepare_dataset(df_raw, target_col)

    perplexities = [5, 15, 30, 50]
    perplexities = [p for p in perplexities if p < len(df_clean)]

    class_silhouette_rows = []
    kmeans_metric_rows = []
    embeddings = {}

    n_classes = len(np.unique(y_encoded))

    for perplexity in perplexities:
        X_tsne = run_tsne(X_scaled, perplexity, args.random_state)
        embeddings[perplexity] = X_tsne

        class_silhouette = silhouette_score(X_tsne, y_encoded)

        kmeans = KMeans(
            n_clusters=n_classes,
            random_state=args.random_state,
            n_init=30,
        )
        cluster_labels = kmeans.fit_predict(X_tsne)

        cluster_silhouette = silhouette_score(X_tsne, cluster_labels)
        ari = adjusted_rand_score(y_encoded, cluster_labels)
        nmi = normalized_mutual_info_score(y_encoded, cluster_labels)
        homogeneity = homogeneity_score(y_encoded, cluster_labels)
        completeness = completeness_score(y_encoded, cluster_labels)
        v_measure = v_measure_score(y_encoded, cluster_labels)

        embedding_df = pd.DataFrame({
            "TSNE1": X_tsne[:, 0],
            "TSNE2": X_tsne[:, 1],
            target_col: y.values,
            "true_label_encoded": y_encoded,
            "kmeans_cluster": cluster_labels,
        })

        embedding_df.to_csv(
            OUTPUT_TABLE_DIR / f"14_tsne_embedding_perplexity_{perplexity}.csv",
            index=False,
        )

        class_silhouette_rows.append({
            "perplexity": perplexity,
            "class_based_silhouette_in_tsne_space": class_silhouette,
            "note": "This is not a clustering metric; it measures true-label separability in t-SNE space.",
        })

        kmeans_metric_rows.append({
            "perplexity": perplexity,
            "kmeans_k": n_classes,
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure,
            "cluster_silhouette": cluster_silhouette,
        })

        plot_tsne_by_class(embedding_df, target_col, perplexity)

        if perplexity == args.selected_perplexity:
            plot_tsne_by_kmeans(embedding_df, perplexity)

            crosstab = pd.crosstab(
                embedding_df[target_col],
                embedding_df["kmeans_cluster"],
                rownames=["true_class"],
                colnames=["kmeans_cluster"],
            )
            crosstab.to_csv(
                OUTPUT_TABLE_DIR / f"14_tsne_kmeans_crosstab_perplexity_{perplexity}.csv"
            )

    class_silhouette_df = pd.DataFrame(class_silhouette_rows)
    kmeans_metrics_df = pd.DataFrame(kmeans_metric_rows)

    class_silhouette_df.to_csv(
        OUTPUT_TABLE_DIR / "14_tsne_perplexity_class_silhouette.csv",
        index=False,
    )

    kmeans_metrics_df.to_csv(
        OUTPUT_TABLE_DIR / "14_tsne_kmeans_metrics.csv",
        index=False,
    )

    selected_metrics = kmeans_metrics_df[
        kmeans_metrics_df["perplexity"] == args.selected_perplexity
    ]

    summary = f"""
t-SNE Metric Correction Summary

CSV Path:
{csv_path}

Target Column:
{target_col}

Rows Used:
{len(df_clean)}

Numerical Features:
{len(numeric_features)}

Class Labels:
{list(encoder.classes_)}

Important Methodological Note:
t-SNE is not a clustering algorithm. Therefore, ARI and NMI should not be computed directly from true labels alone.
In this corrected analysis, two different evaluations are reported:

1. Class-based silhouette in t-SNE space:
   This measures how separated the true class labels appear after t-SNE projection.

2. K-Means on t-SNE embedding:
   K-Means is applied to the 2D t-SNE coordinates, and ARI/NMI are computed between K-Means cluster labels and true emotion labels.

Class-Based Silhouette Scores:
{class_silhouette_df.to_string(index=False)}

K-Means on t-SNE Embedding Metrics:
{kmeans_metrics_df.to_string(index=False)}

Selected Perplexity for Crosstab and Cluster Visualization:
{args.selected_perplexity}

Selected Perplexity Metrics:
{selected_metrics.to_string(index=False)}
""".strip()

    with open(OUTPUT_TEXT_DIR / "14_tsne_metric_correction_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print("\n" + "=" * 90)
    print("14 t-SNE METRIC CORRECTION")
    print("=" * 90)
    print(summary)


if __name__ == "__main__":
    main()