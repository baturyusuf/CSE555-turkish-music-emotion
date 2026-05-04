import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from minisom import MiniSom
from src.common import ensure_dirs, load_dataset, infer_target_column, prepare_analysis_dataset, OUTPUT_FIG_DIR, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, scatter_by_class, class_color_map, save_text, print_section


def scatter_cluster(df_plot, x_col, y_col, label_col, title, filename):
    labels = sorted(df_plot[label_col].astype(str).unique())
    cmap = class_color_map(labels)
    plt.figure(figsize=(8, 6))
    for lbl in labels:
        subset = df_plot[df_plot[label_col].astype(str) == lbl]
        plt.scatter(subset[x_col], subset[y_col], s=42, alpha=0.75, label=lbl, color=cmap[lbl])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / filename, dpi=300)
    plt.close()


def dbscan_eps(X_2d, min_samples=5):
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_2d)
    distances, _ = nn.kneighbors(X_2d)
    kth = np.sort(distances[:, -1])
    return float(np.quantile(kth, 0.90))


def safe_silhouette(X, labels):
    labels = np.asarray(labels)
    unique = [u for u in np.unique(labels) if u != -1]
    if len(unique) < 2:
        return np.nan
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return np.nan
    return float(silhouette_score(X[mask], labels[mask]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    prepared = prepare_analysis_dataset(df_raw, target_col)
    X_scaled = prepared['X_scaled']
    y = prepared['y'].astype(str).reset_index(drop=True)
    n_classes = y.nunique()

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], target_col: y.values})

    kmeans = KMeans(n_clusters=n_classes, n_init=20, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_pca)
    kmeans_df = pca_df.copy()
    kmeans_df['cluster'] = kmeans_labels.astype(str)

    eps = dbscan_eps(X_pca, min_samples=5)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_pca)
    dbscan_df = pca_df.copy()
    dbscan_df['cluster'] = dbscan_labels.astype(str)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_df = pd.DataFrame({'TSNE1': X_tsne[:, 0], 'TSNE2': X_tsne[:, 1], target_col: y.values})

    som_size = 10
    som = MiniSom(som_size, som_size, X_scaled.shape[1], sigma=1.2, learning_rate=0.5, random_seed=42)
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, 3000)
    winners = np.array([som.winner(x) for x in X_scaled])
    som_df = pd.DataFrame({'SOM_X': winners[:, 0], 'SOM_Y': winners[:, 1], target_col: y.values})

    scatter_by_class(kmeans_df, 'PC1', 'PC2', target_col, 'K-Means Input Space (True Classes)', '06_kmeans_true_classes_on_pca2d.png')
    scatter_cluster(kmeans_df, 'PC1', 'PC2', 'cluster', 'K-Means Clusters on PCA 2D Space', '06_kmeans_cluster_labels_on_pca2d.png')
    scatter_by_class(dbscan_df, 'PC1', 'PC2', target_col, 'DBSCAN Input Space (True Classes)', '06_dbscan_true_classes_on_pca2d.png')
    scatter_cluster(dbscan_df, 'PC1', 'PC2', 'cluster', 'DBSCAN Labels on PCA 2D Space', '06_dbscan_cluster_labels_on_pca2d.png')
    scatter_by_class(tsne_df, 'TSNE1', 'TSNE2', target_col, 't-SNE Projection (True Classes)', '06_tsne_true_classes.png')
    scatter_by_class(som_df, 'SOM_X', 'SOM_Y', target_col, 'SOM Winner Map (True Classes)', '06_som_winner_map_true_classes.png')

    plt.figure(figsize=(7, 6))
    plt.imshow(som.distance_map().T, cmap='viridis', origin='lower')
    plt.colorbar(label='U-Matrix Distance')
    plt.title('SOM U-Matrix')
    plt.xlabel('SOM X')
    plt.ylabel('SOM Y')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '06_som_umatrix.png', dpi=300)
    plt.close()

    metrics = pd.DataFrame([
        {
            'method': 'KMeans_on_PCA2D',
            'adjusted_rand_index': adjusted_rand_score(y, kmeans_labels),
            'normalized_mutual_info': normalized_mutual_info_score(y, kmeans_labels),
            'silhouette_score': safe_silhouette(X_pca, kmeans_labels),
            'n_clusters_detected': int(len(np.unique(kmeans_labels))),
            'n_noise_points': 0,
        },
        {
            'method': 'DBSCAN_on_PCA2D',
            'adjusted_rand_index': adjusted_rand_score(y, dbscan_labels),
            'normalized_mutual_info': normalized_mutual_info_score(y, dbscan_labels),
            'silhouette_score': safe_silhouette(X_pca, dbscan_labels),
            'n_clusters_detected': int(len([u for u in np.unique(dbscan_labels) if u != -1])),
            'n_noise_points': int((dbscan_labels == -1).sum()),
        },
    ])

    kmeans_ct = pd.crosstab(y, kmeans_labels, rownames=['class'], colnames=['kmeans_cluster'])
    dbscan_ct = pd.crosstab(y, dbscan_labels, rownames=['class'], colnames=['dbscan_label'])

    pca_df.to_csv(OUTPUT_TABLE_DIR / '06_pca_2d_for_clustering.csv', index=False)
    kmeans_df.to_csv(OUTPUT_TABLE_DIR / '06_kmeans_results.csv', index=False)
    dbscan_df.to_csv(OUTPUT_TABLE_DIR / '06_dbscan_results.csv', index=False)
    tsne_df.to_csv(OUTPUT_TABLE_DIR / '06_tsne_embedding.csv', index=False)
    som_df.to_csv(OUTPUT_TABLE_DIR / '06_som_winner_map.csv', index=False)
    metrics.to_csv(OUTPUT_TABLE_DIR / '06_clustering_metrics.csv', index=False)
    kmeans_ct.to_csv(OUTPUT_TABLE_DIR / '06_kmeans_class_cluster_crosstab.csv')
    dbscan_ct.to_csv(OUTPUT_TABLE_DIR / '06_dbscan_class_cluster_crosstab.csv')

    text = f"""
Clustering and Nonlinear Mapping Analysis

CSV Path: {csv_path}
Target Column: {target_col}
Rows Used: {len(y)}
DBSCAN eps selected from 90th percentile of 5-NN distance: {eps}

Clustering Metrics
{metrics.to_string(index=False)}

K-Means Class vs Cluster Crosstab
{kmeans_ct.to_string()}

DBSCAN Class vs Label Crosstab
{dbscan_ct.to_string()}
"""
    save_text(text.strip() + '\n', OUTPUT_TEXT_DIR / '06_clustering_analysis_summary.txt')

    print_section('06 CLUSTERING ANALYSIS')
    print(text)


if __name__ == '__main__':
    main()
