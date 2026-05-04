import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.common import ensure_dirs, load_dataset, infer_target_column, prepare_analysis_dataset, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, scatter_by_class, save_text, print_section


def separability_metrics(X_embedded, labels):
    return {
        'silhouette_score': float(silhouette_score(X_embedded, labels)),
        'davies_bouldin_score': float(davies_bouldin_score(X_embedded, labels)),
        'calinski_harabasz_score': float(calinski_harabasz_score(X_embedded, labels)),
    }


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
    y = prepared['y']

    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_scaled)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda_2d = lda.fit_transform(X_scaled, y)

    pca_df = pd.DataFrame({'Dim1': X_pca_2d[:, 0], 'Dim2': X_pca_2d[:, 1], target_col: y.values})
    lda_df = pd.DataFrame({'Dim1': X_lda_2d[:, 0], 'Dim2': X_lda_2d[:, 1], target_col: y.values})

    scatter_by_class(pca_df, 'Dim1', 'Dim2', target_col, 'PCA-Based 2D Projection', '05_pca_2d_separability.png', xlabel='PC1', ylabel='PC2')
    scatter_by_class(lda_df, 'Dim1', 'Dim2', target_col, 'LDA-Based 2D Projection', '05_lda_2d_separability.png', xlabel='LD1', ylabel='LD2')

    metrics_df = pd.DataFrame([
        {'embedding': 'PCA_2D', **separability_metrics(X_pca_2d, y)},
        {'embedding': 'LDA_2D', **separability_metrics(X_lda_2d, y)},
    ])

    class_means_pca = pca_df.groupby(target_col)[['Dim1', 'Dim2']].mean().reset_index()
    class_means_lda = lda_df.groupby(target_col)[['Dim1', 'Dim2']].mean().reset_index()

    pca_df.to_csv(OUTPUT_TABLE_DIR / '05_pca_2d_embedding.csv', index=False)
    lda_df.to_csv(OUTPUT_TABLE_DIR / '05_lda_2d_embedding.csv', index=False)
    metrics_df.to_csv(OUTPUT_TABLE_DIR / '05_pca_vs_lda_separability_metrics.csv', index=False)
    class_means_pca.to_csv(OUTPUT_TABLE_DIR / '05_pca_2d_class_centroids.csv', index=False)
    class_means_lda.to_csv(OUTPUT_TABLE_DIR / '05_lda_2d_class_centroids.csv', index=False)

    text = f"""
PCA vs LDA Comparison

CSV Path: {csv_path}
Target Column: {target_col}
Rows Used: {len(y)}

Separability Metrics
{metrics_df.to_string(index=False)}

PCA 2D Class Centroids
{class_means_pca.to_string(index=False)}

LDA 2D Class Centroids
{class_means_lda.to_string(index=False)}
"""
    save_text(text.strip() + '\n', OUTPUT_TEXT_DIR / '05_lda_analysis_summary.txt')

    print_section('05 LDA ANALYSIS')
    print(text)


if __name__ == '__main__':
    main()
