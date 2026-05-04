import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.common import ensure_dirs, load_dataset, infer_target_column, prepare_analysis_dataset, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, multiclass_fisher_score, pairwise_fisher_distance, cohens_d, save_text, print_section


def fisher_breakdown(feature_series, labels):
    overall_mean = feature_series.mean()
    rows = []
    between = 0.0
    within = 0.0
    for cls in sorted(labels.astype(str).unique()):
        vals = feature_series[labels.astype(str) == cls]
        n = len(vals)
        mu_c = vals.mean()
        within_c = ((vals - mu_c) ** 2).sum()
        between_c = n * (mu_c - overall_mean) ** 2
        between += between_c
        within += within_c
        rows.append({
            'class': cls,
            'n': n,
            'class_mean': mu_c,
            'overall_mean': overall_mean,
            'between_contribution': between_c,
            'within_contribution': within_c,
        })
    return pd.DataFrame(rows), between / (within + 1e-12)


def welch_components(a, b):
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / (se + 1e-12)
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = df_num / (df_den + 1e-12)
    return {
        'n1': n1,
        'n2': n2,
        'mean1': m1,
        'mean2': m2,
        'var1': v1,
        'var2': v2,
        'mean_difference': m1 - m2,
        'standard_error': se,
        't_statistic_manual': t,
        'welch_df_manual': df,
        'cohens_d': cohens_d(a, b),
        'pairwise_fisher_distance': pairwise_fisher_distance(a, b),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    rng = np.random.default_rng(args.random_state)
    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    prepared = prepare_analysis_dataset(df_raw, target_col)
    X_scaled = prepared['X_scaled']
    X_scaled_df = prepared['X_scaled_df']
    y = prepared['y'].astype(str).reset_index(drop=True)
    numeric_features = prepared['numeric_features']

    fisher_df = multiclass_fisher_score(X_scaled_df, y).reset_index()
    fisher_df.columns = ['feature', 'multiclass_fisher_score']
    chosen_feature = fisher_df.iloc[0]['feature']

    breakdown_df, fisher_value = fisher_breakdown(X_scaled_df[chosen_feature], y)

    classes = sorted(y.unique())
    pair_rows = []
    for i, class_a in enumerate(classes):
        for class_b in classes[i + 1:]:
            vals_a = X_scaled_df.loc[y == class_a, chosen_feature].to_numpy()
            vals_b = X_scaled_df.loc[y == class_b, chosen_feature].to_numpy()
            pair_rows.append({
                'class_a': class_a,
                'class_b': class_b,
                'pairwise_fisher_distance': pairwise_fisher_distance(vals_a, vals_b),
                'cohens_d': cohens_d(vals_a, vals_b),
            })
    pair_df = pd.DataFrame(pair_rows).sort_values('pairwise_fisher_distance', ascending=False)
    best_pair = pair_df.iloc[0]

    class_a = best_pair['class_a']
    class_b = best_pair['class_b']
    vals_a = X_scaled_df.loc[y == class_a, chosen_feature].to_numpy()
    vals_b = X_scaled_df.loc[y == class_b, chosen_feature].to_numpy()
    sample_a = rng.choice(vals_a, size=36, replace=False)
    sample_b = rng.choice(vals_b, size=36, replace=False)
    welch_df = pd.DataFrame([welch_components(sample_a, sample_b)])

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    pc1_scores = pd.DataFrame({'sample_index': np.arange(len(X_pca)), 'PC1_score': X_pca[:, 0]}).head(20)
    pc1_loading = pd.DataFrame({
        'feature': numeric_features,
        'PC1_loading': pca.components_[0],
        'abs_loading': np.abs(pca.components_[0]),
    }).sort_values('abs_loading', ascending=False)

    fisher_df.to_csv(OUTPUT_TABLE_DIR / '08_manual_support_feature_fisher_ranking.csv', index=False)
    breakdown_df.to_csv(OUTPUT_TABLE_DIR / '08_manual_support_fisher_breakdown.csv', index=False)
    pair_df.to_csv(OUTPUT_TABLE_DIR / '08_manual_support_best_pairs_for_top_feature.csv', index=False)
    welch_df.to_csv(OUTPUT_TABLE_DIR / '08_manual_support_welch_ttest_components.csv', index=False)
    pc1_scores.to_csv(OUTPUT_TABLE_DIR / '08_manual_support_pc1_scores_first_20_samples.csv', index=False)
    pc1_loading.to_csv(OUTPUT_TABLE_DIR / '08_manual_support_pc1_loadings.csv', index=False)

    text = f"""
Manual Calculation Support

CSV Path: {csv_path}
Target Column: {target_col}
Chosen Feature for Hand Calculation: {chosen_feature}
Multi-Class Fisher Score of Chosen Feature: {fisher_value}

Per-Class Contributions to Fisher Score
{breakdown_df.to_string(index=False)}

Best Class Pairs for Chosen Feature
{pair_df.to_string(index=False)}

Welch t-test Components for a 36-vs-36 Sample Draw
{welch_df.to_string(index=False)}

PC1 Summary
Eigenvalue: {pca.explained_variance_[0]}
Explained Variance Ratio: {pca.explained_variance_ratio_[0]}

Top 15 Absolute PC1 Loadings
{pc1_loading.head(15).to_string(index=False)}
"""
    save_text(text.strip() + '\n', OUTPUT_TEXT_DIR / '08_manual_calculation_support_summary.txt')

    print_section('08 MANUAL CALCULATION SUPPORT')
    print(text)


if __name__ == '__main__':
    main()
