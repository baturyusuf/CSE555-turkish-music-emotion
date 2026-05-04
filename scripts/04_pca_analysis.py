import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.common import ensure_dirs, load_dataset, infer_target_column, prepare_analysis_dataset, OUTPUT_FIG_DIR, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, multiclass_fisher_score, scatter_by_class, save_text, print_section


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
    X_scaled_df = prepared['X_scaled_df']
    y = prepared['y']
    numeric_features = prepared['numeric_features']

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    pc_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pc_cols)

    fisher_orig = multiclass_fisher_score(X_scaled_df, y).reset_index()
    fisher_orig.columns = ['feature', 'multiclass_fisher_score']

    fisher_pc = multiclass_fisher_score(X_pca_df, y).reset_index()
    fisher_pc.columns = ['principal_component', 'multiclass_fisher_score']

    explained = pd.DataFrame({
        'principal_component': pc_cols,
        'eigenvalue': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_explained_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
    })
    overview = explained.merge(fisher_pc, on='principal_component', how='left')
    pearson_corr = overview['eigenvalue'].corr(overview['multiclass_fisher_score'], method='pearson')
    spearman_corr = overview['eigenvalue'].corr(overview['multiclass_fisher_score'], method='spearman')

    loadings = pd.DataFrame(pca.components_.T, index=numeric_features, columns=pc_cols)
    top_loadings = []
    for pc in ['PC1', 'PC2', pc_cols[-2], pc_cols[-1]]:
        temp = loadings[pc].abs().sort_values(ascending=False).head(10).reset_index()
        temp.columns = ['feature', 'abs_loading']
        temp['principal_component'] = pc
        temp['loading'] = [loadings.loc[f, pc] for f in temp['feature']]
        top_loadings.append(temp[['principal_component', 'feature', 'loading', 'abs_loading']])
    top_loadings_df = pd.concat(top_loadings, ignore_index=True)

    pca12_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], target_col: y.values})
    pcalast_df = pd.DataFrame({pc_cols[-2]: X_pca[:, -2], pc_cols[-1]: X_pca[:, -1], target_col: y.values})

    scatter_by_class(pca12_df, 'PC1', 'PC2', target_col, 'PCA Projection: First Two Principal Components', '04_pca_first_two_components.png')
    scatter_by_class(pcalast_df, pc_cols[-2], pc_cols[-1], target_col, 'PCA Projection: Last Two Principal Components', '04_pca_last_two_components.png')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pc_cols) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '04_pca_cumulative_explained_variance.png', dpi=300)
    plt.close()

    top_15 = fisher_pc.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_15['principal_component'], top_15['multiclass_fisher_score'])
    plt.xlabel('Multi-Class Fisher Score')
    plt.ylabel('Principal Component')
    plt.title('Top 15 PCA Components by Fisher Score')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '04_top_pca_component_fisher_scores.png', dpi=300)
    plt.close()

    explained.to_csv(OUTPUT_TABLE_DIR / '04_pca_explained_variance.csv', index=False)
    overview.to_csv(OUTPUT_TABLE_DIR / '04_pca_eigenvalue_fisher_overview.csv', index=False)
    fisher_orig.to_csv(OUTPUT_TABLE_DIR / '04_original_feature_fisher_scores.csv', index=False)
    fisher_pc.to_csv(OUTPUT_TABLE_DIR / '04_pca_component_fisher_scores.csv', index=False)
    loadings.to_csv(OUTPUT_TABLE_DIR / '04_pca_loadings.csv')
    top_loadings_df.to_csv(OUTPUT_TABLE_DIR / '04_top_pca_loadings_selected_components.csv', index=False)
    pca12_df.to_csv(OUTPUT_TABLE_DIR / '04_pca_first_two_projection.csv', index=False)
    pcalast_df.to_csv(OUTPUT_TABLE_DIR / '04_pca_last_two_projection.csv', index=False)

    text = f"""
PCA Analysis

CSV Path: {csv_path}
Target Column: {target_col}
Rows Used: {len(X_scaled_df)}
Numerical Features: {len(numeric_features)}

Correlation Between Eigenvalues and PCA Fisher Scores
Pearson: {pearson_corr}
Spearman: {spearman_corr}

Top 15 PCA Components by Fisher Score
{fisher_pc.head(15).to_string(index=False)}

Explained Variance Overview (Top 15 PCs)
{explained.head(15).to_string(index=False)}

Top 20 Original Features by Fisher Score
{fisher_orig.head(20).to_string(index=False)}
"""
    save_text(text.strip() + '\n', OUTPUT_TEXT_DIR / '04_pca_analysis_summary.txt')

    print_section('04 PCA ANALYSIS')
    print(text)


if __name__ == '__main__':
    main()
