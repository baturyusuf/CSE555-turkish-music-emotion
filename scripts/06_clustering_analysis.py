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
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors

from src.common import (
    ensure_dirs,
    load_dataset,
    infer_target_column,
    prepare_analysis_dataset,
    OUTPUT_FIG_DIR,
    OUTPUT_TABLE_DIR,
    OUTPUT_TEXT_DIR,
    class_color_map,
    save_text,
    print_section,
)


RANDOM_STATE = 42


def safe_silhouette(X, labels, ignore_noise=False):
    labels = np.asarray(labels)

    if ignore_noise:
        mask = labels != -1
        if mask.sum() < 3:
            return np.nan
        X_eval = X[mask]
        labels_eval = labels[mask]
    else:
        X_eval = X
        labels_eval = labels

    unique_labels = np.unique(labels_eval)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels_eval):
        return np.nan

    return float(silhouette_score(X_eval, labels_eval))


def clustering_metrics(X, y_true, labels, method_name, ignore_noise_for_silhouette=False):
    labels = np.asarray(labels)
    non_noise = labels != -1

    n_clusters = len(set(labels[non_noise]))
    n_noise = int((labels == -1).sum())

    return {
        "method": method_name,
        "adjusted_rand_index": float(adjusted_rand_score(y_true, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, labels)),
        "homogeneity": float(homogeneity_score(y_true, labels)),
        "completeness": float(completeness_score(y_true, labels)),
        "v_measure": float(v_measure_score(y_true, labels)),
        "silhouette_score": safe_silhouette(
            X,
            labels,
            ignore_noise=ignore_noise_for_silhouette,
        ),
        "n_clusters_detected": int(n_clusters),
        "n_noise_points": n_noise,
    }


def save_scatter_by_class(df_plot, x_col, y_col, class_col, title, filename, xlabel=None, ylabel=None):
    cmap = class_color_map(df_plot[class_col].unique())

    plt.figure(figsize=(8, 6))
    for cls in sorted(df_plot[class_col].astype(str).unique()):
        subset = df_plot[df_plot[class_col].astype(str) == cls]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=42,
            alpha=0.78,
            label=cls,
            color=cmap[cls],
            edgecolor="none",
        )

    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / filename, dpi=300)
    plt.close()


def save_scatter_by_label(df_plot, x_col, y_col, label_col, title, filename, xlabel=None, ylabel=None):
    labels = sorted(df_plot[label_col].astype(str).unique())
    cmap = class_color_map(labels)

    plt.figure(figsize=(8, 6))
    for label in labels:
        subset = df_plot[df_plot[label_col].astype(str) == label]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=42,
            alpha=0.78,
            label=label,
            color=cmap[label],
            edgecolor="none",
        )

    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / filename, dpi=300)
    plt.close()


def run_tsne(X_scaled, perplexity, random_state):
    try:
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=random_state,
            max_iter=1500,
        )
        return model.fit_transform(X_scaled)
    except TypeError:
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=200,
            init="pca",
            random_state=random_state,
            n_iter=1500,
        )
        return model.fit_transform(X_scaled)


def k_distance_values(X_2d, min_samples):
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_2d)
    distances, _ = nn.kneighbors(X_2d)
    return np.sort(distances[:, -1])


def choose_dbscan_eps_unsupervised(kdist, quantile=0.90):
    return float(np.quantile(kdist, quantile))


def plot_kmeans_sensitivity(kmeans_sensitivity_df):
    plt.figure(figsize=(8, 5))
    plt.plot(kmeans_sensitivity_df["k"], kmeans_sensitivity_df["inertia"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Curve on PCA 2D Space")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "06_kmeans_elbow_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(kmeans_sensitivity_df["k"], kmeans_sensitivity_df["adjusted_rand_index"], marker="o", label="ARI")
    plt.plot(kmeans_sensitivity_df["k"], kmeans_sensitivity_df["normalized_mutual_info"], marker="o", label="NMI")
    plt.plot(kmeans_sensitivity_df["k"], kmeans_sensitivity_df["silhouette_score"], marker="o", label="Silhouette")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Score")
    plt.title("K-Means Sensitivity on PCA 2D Space")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "06_kmeans_metric_sensitivity.png", dpi=300)
    plt.close()


def plot_dbscan_sensitivity(dbscan_sensitivity_df):
    plt.figure(figsize=(8, 5))
    plt.plot(dbscan_sensitivity_df["eps"], dbscan_sensitivity_df["n_clusters_detected"], marker="o", label="Detected clusters")
    plt.plot(dbscan_sensitivity_df["eps"], dbscan_sensitivity_df["n_noise_points"], marker="o", label="Noise points")
    plt.xlabel("eps")
    plt.ylabel("Count")
    plt.title("DBSCAN eps Sensitivity: Cluster and Noise Counts")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "06_dbscan_eps_sensitivity_counts.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(dbscan_sensitivity_df["eps"], dbscan_sensitivity_df["adjusted_rand_index"], marker="o", label="ARI")
    plt.plot(dbscan_sensitivity_df["eps"], dbscan_sensitivity_df["normalized_mutual_info"], marker="o", label="NMI")
    plt.plot(dbscan_sensitivity_df["eps"], dbscan_sensitivity_df["silhouette_score"], marker="o", label="Silhouette")
    plt.xlabel("eps")
    plt.ylabel("Score")
    plt.title("DBSCAN eps Sensitivity: External and Internal Validity")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "06_dbscan_eps_sensitivity_validity.png", dpi=300)
    plt.close()


def run_som_analysis(X_scaled, y, n_classes, grid_size=10, iterations=5000, random_state=42):
    try:
        from minisom import MiniSom
    except ImportError as exc:
        raise ImportError("MiniSom is required for SOM analysis. Install it with: pip install minisom") from exc

    som = MiniSom(
        x=grid_size,
        y=grid_size,
        input_len=X_scaled.shape[1],
        sigma=1.2,
        learning_rate=0.5,
        random_seed=random_state,
    )
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, iterations)

    winners = np.array([som.winner(x) for x in X_scaled])
    winning_neuron = np.array([f"{i}_{j}" for i, j in winners])

    som_df = pd.DataFrame({
        "SOM_X": winners[:, 0],
        "SOM_Y": winners[:, 1],
        "winning_neuron": winning_neuron,
        "true_class": y.values,
    })

    neuron_purity_rows = []
    for neuron, group in som_df.groupby("winning_neuron"):
        counts = group["true_class"].value_counts()
        neuron_purity_rows.append({
            "winning_neuron": neuron,
            "total_hits": int(counts.sum()),
            "majority_class": counts.index[0],
            "majority_count": int(counts.iloc[0]),
            "purity": float(counts.iloc[0] / counts.sum()),
            "class_counts": "; ".join([f"{idx}:{val}" for idx, val in counts.items()]),
        })

    neuron_purity_df = pd.DataFrame(neuron_purity_rows).sort_values(
        ["total_hits", "purity"],
        ascending=[False, False],
    )

    weights = som.get_weights()
    neuron_rows = []
    neuron_vectors = []

    for i in range(grid_size):
        for j in range(grid_size):
            neuron_rows.append({"SOM_X": i, "SOM_Y": j, "winning_neuron": f"{i}_{j}"})
            neuron_vectors.append(weights[i, j, :])

    neuron_grid_df = pd.DataFrame(neuron_rows)
    neuron_vectors = np.asarray(neuron_vectors)

    neuron_kmeans = KMeans(
        n_clusters=n_classes,
        n_init=30,
        random_state=random_state,
    )
    neuron_cluster_labels = neuron_kmeans.fit_predict(neuron_vectors)
    neuron_grid_df["som_neuron_cluster"] = neuron_cluster_labels

    neuron_to_cluster = dict(zip(neuron_grid_df["winning_neuron"], neuron_grid_df["som_neuron_cluster"]))
    som_df["som_cluster"] = som_df["winning_neuron"].map(neuron_to_cluster).astype(int)

    u_matrix = som.distance_map().T

    plt.figure(figsize=(7, 6))
    plt.imshow(u_matrix, origin="lower")
    plt.colorbar(label="Average neighbor distance")
    plt.title("SOM U-Matrix")
    plt.xlabel("SOM X")
    plt.ylabel("SOM Y")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "06_som_u_matrix.png", dpi=300)
    plt.close()

    class_order = sorted(y.astype(str).unique())
    n_cols = 2
    n_rows = int(np.ceil(len(class_order) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4.5 * n_rows))
    axes = np.asarray(axes).ravel()

    for ax, cls in zip(axes, class_order):
        mat = np.zeros((grid_size, grid_size))
        part = som_df[som_df["true_class"] == cls]
        for _, row in part.iterrows():
            mat[int(row["SOM_Y"]), int(row["SOM_X"])] += 1

        im = ax.imshow(mat, origin="lower")
        ax.set_title(f"SOM Class Hit Map: {cls}")
        ax.set_xlabel("SOM X")
        ax.set_ylabel("SOM Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(class_order):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "06_som_class_hit_maps.png", dpi=300)
    plt.close(fig)

    jitter_rng = np.random.default_rng(random_state)
    som_df["SOM_X_jitter"] = som_df["SOM_X"] + jitter_rng.normal(0, 0.08, len(som_df))
    som_df["SOM_Y_jitter"] = som_df["SOM_Y"] + jitter_rng.normal(0, 0.08, len(som_df))

    save_scatter_by_class(
        som_df,
        "SOM_X_jitter",
        "SOM_Y_jitter",
        "true_class",
        "SOM Winner Map Colored by True Emotion Class",
        "06_som_winner_map_true_classes.png",
        xlabel="SOM X",
        ylabel="SOM Y",
    )

    save_scatter_by_label(
        som_df,
        "SOM_X_jitter",
        "SOM_Y_jitter",
        "som_cluster",
        "SOM Neuron Clusters",
        "06_som_neuron_clusters.png",
        xlabel="SOM X",
        ylabel="SOM Y",
    )

    som_crosstab = pd.crosstab(
        som_df["true_class"],
        som_df["som_cluster"],
        rownames=["true_class"],
        colnames=["som_cluster"],
    )

    occupied_neuron_purity = float(
        neuron_purity_df["majority_count"].sum() / neuron_purity_df["total_hits"].sum()
    )

    return som_df, neuron_grid_df, neuron_purity_df, som_crosstab, occupied_neuron_purity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--dbscan-selected-quantile", type=float, default=0.90)
    parser.add_argument("--selected-tsne-perplexity", type=int, default=30)
    parser.add_argument("--som-grid-size", type=int, default=10)
    parser.add_argument("--som-iterations", type=int, default=5000)
    args = parser.parse_args()

    ensure_dirs()

    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    prepared = prepare_analysis_dataset(df_raw, target_col)

    X_scaled = prepared["X_scaled"]
    y = prepared["y"].astype(str).reset_index(drop=True)
    y_codes = pd.Categorical(y).codes
    n_classes = y.nunique()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        target_col: y.values,
    })

    pca_df.to_csv(OUTPUT_TABLE_DIR / "06_pca_2d_for_clustering.csv", index=False)

    save_scatter_by_class(
        pca_df,
        "PC1",
        "PC2",
        target_col,
        "PCA 2D Space Colored by True Emotion Class",
        "06_pca2d_true_classes_for_clustering.png",
    )

    # -------------------------------------------------------------------------
    # K-Means on PCA 2D
    # -------------------------------------------------------------------------
    kmeans_rows = []
    selected_kmeans_labels = None
    selected_kmeans_df = None

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, n_init=30, random_state=args.random_state)
        labels = kmeans.fit_predict(X_pca)

        row = {
            "k": k,
            "inertia": float(kmeans.inertia_),
            **clustering_metrics(
                X_pca,
                y_codes,
                labels,
                method_name=f"KMeans_on_PCA2D_k{k}",
                ignore_noise_for_silhouette=False,
            ),
        }
        kmeans_rows.append(row)

        if k == n_classes:
            selected_kmeans_labels = labels
            selected_kmeans_df = pca_df.copy()
            selected_kmeans_df["kmeans_cluster"] = labels.astype(int)

    kmeans_sensitivity_df = pd.DataFrame(kmeans_rows)
    kmeans_sensitivity_df.to_csv(OUTPUT_TABLE_DIR / "06_kmeans_sensitivity_on_pca2d.csv", index=False)
    plot_kmeans_sensitivity(kmeans_sensitivity_df)

    selected_kmeans_df.to_csv(OUTPUT_TABLE_DIR / "06_kmeans_k4_results_on_pca2d.csv", index=False)

    save_scatter_by_class(
        selected_kmeans_df,
        "PC1",
        "PC2",
        target_col,
        "K-Means Input Space Colored by True Emotion Class",
        "06_kmeans_true_classes_on_pca2d.png",
    )

    save_scatter_by_label(
        selected_kmeans_df,
        "PC1",
        "PC2",
        "kmeans_cluster",
        "K-Means Cluster Labels on PCA 2D Space",
        "06_kmeans_cluster_labels_on_pca2d.png",
    )

    kmeans_crosstab = pd.crosstab(
        selected_kmeans_df[target_col],
        selected_kmeans_df["kmeans_cluster"],
        rownames=["true_class"],
        colnames=["kmeans_cluster"],
    )
    kmeans_crosstab.to_csv(OUTPUT_TABLE_DIR / "06_kmeans_class_cluster_crosstab.csv")

    # -------------------------------------------------------------------------
    # DBSCAN on PCA 2D
    # -------------------------------------------------------------------------
    kdist = k_distance_values(X_pca, min_samples=args.dbscan_min_samples)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(kdist)), kdist)
    plt.xlabel(f"Samples sorted by {args.dbscan_min_samples}-NN distance")
    plt.ylabel(f"{args.dbscan_min_samples}-NN distance")
    plt.title("DBSCAN k-Distance Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "06_dbscan_k_distance_curve.png", dpi=300)
    plt.close()

    eps_values = np.quantile(kdist, [0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975])
    selected_eps = choose_dbscan_eps_unsupervised(kdist, quantile=args.dbscan_selected_quantile)

    dbscan_rows = []
    selected_dbscan_labels = None
    selected_dbscan_df = None

    for eps in eps_values:
        dbscan = DBSCAN(eps=float(eps), min_samples=args.dbscan_min_samples)
        labels = dbscan.fit_predict(X_pca)

        row = {
            "eps": float(eps),
            "min_samples": args.dbscan_min_samples,
            "selection_note": "selected_unsupervised_eps" if np.isclose(eps, selected_eps) else "",
            **clustering_metrics(
                X_pca,
                y_codes,
                labels,
                method_name=f"DBSCAN_on_PCA2D_eps_{eps:.3f}",
                ignore_noise_for_silhouette=True,
            ),
        }
        dbscan_rows.append(row)

    dbscan_sensitivity_df = pd.DataFrame(dbscan_rows)
    dbscan_sensitivity_df.to_csv(OUTPUT_TABLE_DIR / "06_dbscan_eps_sensitivity_on_pca2d.csv", index=False)
    plot_dbscan_sensitivity(dbscan_sensitivity_df)

    dbscan = DBSCAN(eps=selected_eps, min_samples=args.dbscan_min_samples)
    selected_dbscan_labels = dbscan.fit_predict(X_pca)
    selected_dbscan_df = pca_df.copy()
    selected_dbscan_df["dbscan_label"] = selected_dbscan_labels.astype(int)
    selected_dbscan_df.to_csv(OUTPUT_TABLE_DIR / "06_dbscan_selected_results_on_pca2d.csv", index=False)

    save_scatter_by_class(
        selected_dbscan_df,
        "PC1",
        "PC2",
        target_col,
        f"DBSCAN Input Space Colored by True Emotion Class | eps={selected_eps:.3f}",
        "06_dbscan_true_classes_on_pca2d.png",
    )

    save_scatter_by_label(
        selected_dbscan_df,
        "PC1",
        "PC2",
        "dbscan_label",
        f"DBSCAN Labels on PCA 2D Space | eps={selected_eps:.3f}",
        "06_dbscan_cluster_labels_on_pca2d.png",
    )

    dbscan_crosstab = pd.crosstab(
        selected_dbscan_df[target_col],
        selected_dbscan_df["dbscan_label"],
        rownames=["true_class"],
        colnames=["dbscan_label"],
    )
    dbscan_crosstab.to_csv(OUTPUT_TABLE_DIR / "06_dbscan_class_cluster_crosstab.csv")

    # -------------------------------------------------------------------------
    # t-SNE on original normalized dataset + K-Means on t-SNE embedding
    # -------------------------------------------------------------------------
    tsne_rows = []
    selected_tsne_df = None

    perplexities = [5, 15, 30, 50]
    perplexities = [p for p in perplexities if p < len(y)]

    for perplexity in perplexities:
        X_tsne = run_tsne(X_scaled, perplexity=perplexity, random_state=args.random_state)

        tsne_df = pd.DataFrame({
            "TSNE1": X_tsne[:, 0],
            "TSNE2": X_tsne[:, 1],
            target_col: y.values,
        })

        kmeans_tsne = KMeans(
            n_clusters=n_classes,
            n_init=30,
            random_state=args.random_state,
        )
        tsne_cluster_labels = kmeans_tsne.fit_predict(X_tsne)
        tsne_df["kmeans_cluster_on_tsne"] = tsne_cluster_labels.astype(int)

        class_silhouette = safe_silhouette(X_tsne, y_codes)
        cluster_metrics = clustering_metrics(
            X_tsne,
            y_codes,
            tsne_cluster_labels,
            method_name=f"KMeans_on_tSNE_perplexity_{perplexity}",
            ignore_noise_for_silhouette=False,
        )

        tsne_rows.append({
            "perplexity": perplexity,
            "class_based_silhouette_in_tsne_space": class_silhouette,
            "class_based_silhouette_note": "Uses true labels only to measure visual class separability; this is not a clustering metric.",
            "kmeans_k": n_classes,
            **cluster_metrics,
        })

        tsne_df.to_csv(OUTPUT_TABLE_DIR / f"06_tsne_embedding_perplexity_{perplexity}.csv", index=False)

        save_scatter_by_class(
            tsne_df,
            "TSNE1",
            "TSNE2",
            target_col,
            f"t-SNE Projection Colored by True Emotion Class | Perplexity={perplexity}",
            f"06_tsne_true_classes_perplexity_{perplexity}.png",
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
        )

        if perplexity == args.selected_tsne_perplexity:
            selected_tsne_df = tsne_df.copy()

            save_scatter_by_label(
                selected_tsne_df,
                "TSNE1",
                "TSNE2",
                "kmeans_cluster_on_tsne",
                f"K-Means Clusters on t-SNE Embedding | Perplexity={perplexity}",
                f"06_tsne_kmeans_clusters_perplexity_{perplexity}.png",
                xlabel="t-SNE 1",
                ylabel="t-SNE 2",
            )

            tsne_crosstab = pd.crosstab(
                selected_tsne_df[target_col],
                selected_tsne_df["kmeans_cluster_on_tsne"],
                rownames=["true_class"],
                colnames=["kmeans_cluster_on_tsne"],
            )
            tsne_crosstab.to_csv(OUTPUT_TABLE_DIR / f"06_tsne_kmeans_crosstab_perplexity_{perplexity}.csv")

    tsne_metrics_df = pd.DataFrame(tsne_rows)
    tsne_metrics_df.to_csv(OUTPUT_TABLE_DIR / "06_tsne_metrics.csv", index=False)

    # -------------------------------------------------------------------------
    # SOM on original normalized dataset
    # -------------------------------------------------------------------------
    som_df, som_neuron_grid_df, som_neuron_purity_df, som_crosstab, occupied_neuron_purity = run_som_analysis(
        X_scaled=X_scaled,
        y=y,
        n_classes=n_classes,
        grid_size=args.som_grid_size,
        iterations=args.som_iterations,
        random_state=args.random_state,
    )

    som_df.to_csv(OUTPUT_TABLE_DIR / "06_som_sample_winning_neurons_and_clusters.csv", index=False)
    som_neuron_grid_df.to_csv(OUTPUT_TABLE_DIR / "06_som_neuron_grid_clusters.csv", index=False)
    som_neuron_purity_df.to_csv(OUTPUT_TABLE_DIR / "06_som_occupied_neuron_purity.csv", index=False)
    som_crosstab.to_csv(OUTPUT_TABLE_DIR / "06_som_class_cluster_crosstab.csv")

    som_metrics = {
        "method": "SOM_neuron_clustering",
        "adjusted_rand_index": float(adjusted_rand_score(y_codes, som_df["som_cluster"])),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_codes, som_df["som_cluster"])),
        "homogeneity": float(homogeneity_score(y_codes, som_df["som_cluster"])),
        "completeness": float(completeness_score(y_codes, som_df["som_cluster"])),
        "v_measure": float(v_measure_score(y_codes, som_df["som_cluster"])),
        "silhouette_score": safe_silhouette(
            som_df[["SOM_X", "SOM_Y"]].to_numpy(),
            som_df["som_cluster"].to_numpy(),
        ),
        "n_clusters_detected": int(som_df["som_cluster"].nunique()),
        "n_noise_points": 0,
        "occupied_neuron_majority_purity": occupied_neuron_purity,
    }

    # -------------------------------------------------------------------------
    # Combined metrics and settings
    # -------------------------------------------------------------------------
    combined_metrics_rows = []

    combined_metrics_rows.append(
        clustering_metrics(
            X_pca,
            y_codes,
            selected_kmeans_labels,
            method_name="KMeans_on_PCA2D_k4",
        )
    )

    combined_metrics_rows.append(
        clustering_metrics(
            X_pca,
            y_codes,
            selected_dbscan_labels,
            method_name=f"DBSCAN_on_PCA2D_eps_{selected_eps:.3f}",
            ignore_noise_for_silhouette=True,
        )
    )

    if selected_tsne_df is not None:
        combined_metrics_rows.append(
            clustering_metrics(
                selected_tsne_df[["TSNE1", "TSNE2"]].to_numpy(),
                y_codes,
                selected_tsne_df["kmeans_cluster_on_tsne"].to_numpy(),
                method_name=f"KMeans_on_tSNE_perplexity_{args.selected_tsne_perplexity}",
            )
        )

    combined_metrics_rows.append(som_metrics)

    combined_metrics_df = pd.DataFrame(combined_metrics_rows)
    combined_metrics_df.to_csv(OUTPUT_TABLE_DIR / "06_combined_clustering_metrics.csv", index=False)

    settings_df = pd.DataFrame([
        {
            "method": "PCA for K-Means and DBSCAN",
            "input_data": "duplicate-removed, IQR-capped, z-score normalized features",
            "parameters": "n_components=2",
            "random_state": args.random_state,
        },
        {
            "method": "K-Means on PCA 2D",
            "input_data": "first two principal components",
            "parameters": f"k={n_classes}, n_init=30",
            "random_state": args.random_state,
        },
        {
            "method": "DBSCAN on PCA 2D",
            "input_data": "first two principal components",
            "parameters": f"eps={selected_eps:.6f}, min_samples={args.dbscan_min_samples}, eps_quantile={args.dbscan_selected_quantile}",
            "random_state": "deterministic",
        },
        {
            "method": "t-SNE",
            "input_data": "original normalized feature matrix",
            "parameters": f"perplexities={perplexities}, selected_perplexity={args.selected_tsne_perplexity}, init=pca, learning_rate=auto/200, iterations=1500",
            "random_state": args.random_state,
        },
        {
            "method": "K-Means on t-SNE",
            "input_data": "2D t-SNE embedding",
            "parameters": f"k={n_classes}, n_init=30",
            "random_state": args.random_state,
        },
        {
            "method": "SOM",
            "input_data": "original normalized feature matrix",
            "parameters": f"grid={args.som_grid_size}x{args.som_grid_size}, sigma=1.2, learning_rate=0.5, iterations={args.som_iterations}",
            "random_state": args.random_state,
        },
    ])
    settings_df.to_csv(OUTPUT_TABLE_DIR / "06_clustering_experimental_settings.csv", index=False)

    summary = f"""
Clustering and Nonlinear Embedding Analysis

CSV Path:
{csv_path}

Target Column:
{target_col}

Rows Used:
{len(y)}

Number of True Classes:
{n_classes}

Main Methodological Notes:
1. K-Means and DBSCAN were applied on the first two principal components, as required.
2. t-SNE was applied on the original normalized dataset. Since t-SNE is not a clustering algorithm, K-Means was additionally applied on the t-SNE embedding for cluster-level evaluation.
3. SOM was applied on the original normalized dataset. Each sample was assigned to its best matching unit (BMU), and SOM neuron codebook vectors were clustered into {n_classes} neuron clusters for external validity evaluation.
4. Clustering metrics compare algorithm-produced labels with true emotion labels. Class-based silhouette in t-SNE space is reported separately as a visual separability measure, not as a clustering metric.

Selected DBSCAN eps:
{selected_eps:.6f}

Combined Clustering Metrics:
{combined_metrics_df.to_string(index=False)}

K-Means Sensitivity:
{kmeans_sensitivity_df.to_string(index=False)}

DBSCAN eps Sensitivity:
{dbscan_sensitivity_df.to_string(index=False)}

t-SNE Metrics:
{tsne_metrics_df.to_string(index=False)}

SOM Class vs Cluster Crosstab:
{som_crosstab.to_string()}

Experimental Settings:
{settings_df.to_string(index=False)}
""".strip()

    save_text(summary + "\n", OUTPUT_TEXT_DIR / "06_clustering_analysis_summary.txt")

    print_section("06 CLUSTERING ANALYSIS")
    print(summary)


if __name__ == "__main__":
    main()