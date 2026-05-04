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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.advanced_common import (
    ensure_dirs, configure_plots, load_dataset, infer_target_column, prepared_variant,
    safe_cluster_silhouette, scatter_by_class, OUTPUT_TABLE_DIR, OUTPUT_FIG_DIR,
    OUTPUT_TEXT_DIR, save_text, print_section, color_map
)

def cluster_metrics(X, y_true, labels):
    labels = np.asarray(labels)
    non_noise = labels != -1
    n_clusters = len(set(labels[non_noise]))
    return {
        "adjusted_rand_index": float(adjusted_rand_score(y_true, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, labels)),
        "silhouette_score": safe_cluster_silhouette(X, labels),
        "n_clusters_detected": int(n_clusters),
        "n_noise_points": int((labels == -1).sum()),
    }

def plot_cluster_embedding(df_plot, x, y, label_col, title, filename):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    labels = sorted(df_plot[label_col].astype(str).unique())
    cmap = color_map(labels)
    for lbl in labels:
        part = df_plot[df_plot[label_col].astype(str) == lbl]
        ax.scatter(part[x], part[y], s=34, alpha=0.78, label=lbl, color=cmap.get(lbl), edgecolor="none")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / filename)
    plt.close(fig)

def run_som(X_scaled, y, grid_x, grid_y, seed, iterations):
    try:
        from minisom import MiniSom
    except ImportError:
        return None, None, "MiniSom is not installed. Run: pip install minisom"

    som = MiniSom(grid_x, grid_y, X_scaled.shape[1], sigma=1.2, learning_rate=0.5, random_seed=seed)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, iterations)

    winners = np.array([som.winner(x) for x in X_scaled])
    winner_labels = [f"{i}_{j}" for i, j in winners]

    y_series = pd.Series(y).astype(str).reset_index(drop=True)
    rows = []
    for idx, (coord, cls) in enumerate(zip(winner_labels, y_series)):
        rows.append({"sample_index": idx, "winning_neuron": coord, "class": cls})
    hit_df = pd.DataFrame(rows)

    purity_rows = []
    for neuron, group in hit_df.groupby("winning_neuron"):
        counts = group["class"].value_counts()
        purity_rows.append({
            "winning_neuron": neuron,
            "total_hits": int(counts.sum()),
            "majority_class": counts.index[0],
            "majority_count": int(counts.iloc[0]),
            "purity": float(counts.iloc[0] / counts.sum()),
        })
    purity_df = pd.DataFrame(purity_rows).sort_values(["purity", "total_hits"], ascending=[False, False])

    umat = som.distance_map().T
    class_order = sorted(y_series.unique())
    hit_matrices = {}
    for cls in class_order:
        mat = np.zeros((grid_y, grid_x))
        for _, row in hit_df[hit_df["class"] == cls].iterrows():
            i, j = map(int, row["winning_neuron"].split("_"))
            mat[j, i] += 1
        hit_matrices[cls] = mat

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(umat, origin="lower")
    ax.set_title("SOM U-Matrix")
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    fig.colorbar(im, ax=ax, label="Average Distance")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "11_som_u_matrix.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()
    for ax, cls in zip(axes, class_order):
        im = ax.imshow(hit_matrices[cls], origin="lower")
        ax.set_title(f"SOM Class Hit Map: {cls}")
        ax.set_xlabel("SOM X")
        ax.set_ylabel("SOM Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "11_som_class_hit_maps.png")
    plt.close(fig)

    return hit_df, purity_df, "SOM analysis completed."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--som-grid-x", type=int, default=10)
    parser.add_argument("--som-grid-y", type=int, default=10)
    parser.add_argument("--som-iterations", type=int, default=5000)
    args = parser.parse_args()

    ensure_dirs()
    configure_plots()

    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    data = prepared_variant(df_raw, target_col, "duplicate_cleaned_iqr_capped")
    X_scaled = data["X_scaled"]
    y = data["y"]

    pca = PCA(n_components=2, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "class": y.values})

    y_codes = pd.Categorical(y).codes

    k_rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=30, random_state=args.random_state)
        labels = km.fit_predict(X_pca)
        metrics = cluster_metrics(X_pca, y_codes, labels)
        k_rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            **metrics,
        })
        tmp = pca_df.copy()
        tmp["kmeans_cluster"] = labels.astype(str)
        if k == 4:
            plot_cluster_embedding(tmp, "PC1", "PC2", "kmeans_cluster", "K-Means Clusters on PCA 2D Space (k=4)", "11_kmeans_k4_clusters_on_pca2d.png")
            scatter_by_class(None, None, None, None, None) if False else None

    k_df = pd.DataFrame(k_rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_df["k"], k_df["inertia"], marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Curve on PCA 2D Space")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "11_kmeans_elbow_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_df["k"], k_df["silhouette_score"], marker="o", label="Silhouette")
    ax.plot(k_df["k"], k_df["adjusted_rand_index"], marker="o", label="ARI")
    ax.plot(k_df["k"], k_df["normalized_mutual_info"], marker="o", label="NMI")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Score")
    ax.set_title("K-Means Sensitivity Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "11_kmeans_metric_sensitivity.png")
    plt.close(fig)

    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_pca)
    distances, _ = neighbors_fit.kneighbors(X_pca)
    kdist = np.sort(distances[:, -1])
    eps_values = np.quantile(kdist, [0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(len(kdist)), kdist)
    ax.set_xlabel("Samples Sorted by 5-NN Distance")
    ax.set_ylabel("5-NN Distance")
    ax.set_title("DBSCAN k-Distance Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "11_dbscan_k_distance_curve.png")
    plt.close(fig)

    db_rows = []
    best_db = None
    best_key = -np.inf
    for eps in eps_values:
        db = DBSCAN(eps=float(eps), min_samples=5)
        labels = db.fit_predict(X_pca)
        metrics = cluster_metrics(X_pca, y_codes, labels)
        row = {"eps": float(eps), "min_samples": 5, **metrics}
        db_rows.append(row)
        key = metrics["normalized_mutual_info"] - 0.002 * metrics["n_noise_points"]
        if key > best_key:
            best_key = key
            best_db = (eps, labels, metrics)

    db_df = pd.DataFrame(db_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(db_df["eps"], db_df["n_clusters_detected"], marker="o", label="Detected Clusters")
    ax.plot(db_df["eps"], db_df["n_noise_points"], marker="o", label="Noise Points")
    ax.set_xlabel("eps")
    ax.set_title("DBSCAN eps Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "11_dbscan_eps_sensitivity_counts.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(db_df["eps"], db_df["adjusted_rand_index"], marker="o", label="ARI")
    ax.plot(db_df["eps"], db_df["normalized_mutual_info"], marker="o", label="NMI")
    ax.set_xlabel("eps")
    ax.set_ylabel("Score")
    ax.set_title("DBSCAN eps Sensitivity: External Cluster Validity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "11_dbscan_eps_sensitivity_validity.png")
    plt.close(fig)

    best_eps, best_labels, best_metrics = best_db
    tmp = pca_df.copy()
    tmp["dbscan_label"] = best_labels.astype(str)
    plot_cluster_embedding(tmp, "PC1", "PC2", "dbscan_label", f"DBSCAN Labels on PCA 2D Space (eps={best_eps:.3f})", "11_dbscan_best_labels_on_pca2d.png")

    tsne_rows = []
    for perplexity in [5, 15, 30, 50]:
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=args.random_state,
                max_iter=1500,
            )
        except TypeError:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=args.random_state,
                n_iter=1500,
            )
        X_tsne = tsne.fit_transform(X_scaled)
        tsne_df = pd.DataFrame({"tSNE1": X_tsne[:, 0], "tSNE2": X_tsne[:, 1], "class": y.values})
        tsne_df.to_csv(OUTPUT_TABLE_DIR / f"11_tsne_embedding_perplexity_{perplexity}.csv", index=False)
        tsne_rows.append({"perplexity": perplexity, **cluster_metrics(X_tsne, y_codes, y_codes)})
        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        scatter_by_class(ax, tsne_df, "tSNE1", "tSNE2", "class", f"t-SNE Embedding (perplexity={perplexity})", "t-SNE 1", "t-SNE 2")
        fig.tight_layout()
        fig.savefig(OUTPUT_FIG_DIR / f"11_tsne_perplexity_{perplexity}.png")
        plt.close(fig)

    tsne_metric_df = pd.DataFrame(tsne_rows)

    som_hit_df, som_purity_df, som_status = run_som(
        X_scaled,
        y,
        grid_x=args.som_grid_x,
        grid_y=args.som_grid_y,
        seed=args.random_state,
        iterations=args.som_iterations,
    )

    k_df.to_csv(OUTPUT_TABLE_DIR / "11_kmeans_sensitivity.csv", index=False)
    db_df.to_csv(OUTPUT_TABLE_DIR / "11_dbscan_eps_sensitivity.csv", index=False)
    tsne_metric_df.to_csv(OUTPUT_TABLE_DIR / "11_tsne_perplexity_separability_metrics.csv", index=False)

    if som_hit_df is not None:
        som_hit_df.to_csv(OUTPUT_TABLE_DIR / "11_som_winning_neurons.csv", index=False)
        som_purity_df.to_csv(OUTPUT_TABLE_DIR / "11_som_neuron_purity.csv", index=False)

    text = f"""
Clustering Sensitivity and Nonlinear Mapping Analysis

CSV Path: {csv_path}
Target Column: {target_col}
Rows Used: {len(y)}
PCA 2D Explained Variance Ratio: {pca.explained_variance_ratio_.sum()}

K-Means Sensitivity
{k_df.to_string(index=False)}

DBSCAN eps Sensitivity
{db_df.to_string(index=False)}

Selected DBSCAN eps: {best_eps}
Selected DBSCAN Metrics: {best_metrics}

t-SNE Perplexity Metrics Based on True Class Labels in Embedding Space
{tsne_metric_df.to_string(index=False)}

SOM Status:
{som_status}
"""
    if som_hit_df is not None:
        text += f"\nSOM Neuron Purity Summary\n{som_purity_df.head(20).to_string(index=False)}\n"

    save_text(text.strip() + "\n", OUTPUT_TEXT_DIR / "11_clustering_sensitivity_summary.txt")

    print_section("11 CLUSTERING SENSITIVITY")
    print(text)

if __name__ == "__main__":
    main()
