import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.common import ensure_dirs, load_dataset, infer_target_column, prepare_analysis_dataset, OUTPUT_FIG_DIR, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, multiclass_fisher_score, pairwise_fisher_distance, cohens_d, class_color_map, save_text, print_section


def select_best_test_feature(X_scaled_df, y):
    fisher_df = multiclass_fisher_score(X_scaled_df, y).reset_index()
    fisher_df.columns = ['feature', 'multiclass_fisher_score']
    top_features = fisher_df.head(10)['feature'].tolist()
    classes = sorted(y.astype(str).unique())
    candidates = []
    for feature in top_features:
        for i, cls_a in enumerate(classes):
            for cls_b in classes[i + 1:]:
                a = X_scaled_df.loc[y.astype(str) == cls_a, feature]
                b = X_scaled_df.loc[y.astype(str) == cls_b, feature]
                candidates.append({
                    'feature': feature,
                    'class_a': cls_a,
                    'class_b': cls_b,
                    'pairwise_fisher_distance': pairwise_fisher_distance(a, b),
                    'cohens_d_abs': abs(cohens_d(a, b)),
                })
    cand_df = pd.DataFrame(candidates)
    return cand_df.sort_values(['pairwise_fisher_distance', 'cohens_d_abs'], ascending=False).reset_index(drop=True)


def welch_details(sample_a, sample_b):
    n1, n2 = len(sample_a), len(sample_b)
    m1, m2 = np.mean(sample_a), np.mean(sample_b)
    s1, s2 = np.var(sample_a, ddof=1), np.var(sample_b, ddof=1)
    t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    se = np.sqrt(s1 / n1 + s2 / n2)
    df_num = (s1 / n1 + s2 / n2) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s2 / n2) ** 2) / (n2 - 1)
    dof = df_num / (df_den + 1e-12)
    return {
        'n_class_a': n1,
        'n_class_b': n2,
        'mean_class_a': m1,
        'mean_class_b': m2,
        'var_class_a': s1,
        'var_class_b': s2,
        'mean_difference': m1 - m2,
        'standard_error': se,
        'welch_df': dof,
        't_statistic': t_stat,
        'p_value': p_value,
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
    X_scaled_df = prepared['X_scaled_df']
    y = prepared['y'].astype(str).reset_index(drop=True)

    candidates = select_best_test_feature(X_scaled_df, y)
    best = candidates.iloc[0]
    feature = best['feature']
    class_a = best['class_a']
    class_b = best['class_b']

    values_a = X_scaled_df.loc[y == class_a, feature].to_numpy()
    values_b = X_scaled_df.loc[y == class_b, feature].to_numpy()

    results = []
    details_rows = []
    for n in [36, 64]:
        sample_a = rng.choice(values_a, size=n, replace=False)
        sample_b = rng.choice(values_b, size=n, replace=False)
        details = welch_details(sample_a, sample_b)
        details['feature'] = feature
        details['class_a'] = class_a
        details['class_b'] = class_b
        details['sample_size_per_class'] = n
        details['cohens_d'] = cohens_d(sample_a, sample_b)
        details['decision_alpha_0_05'] = 'Reject H0' if details['p_value'] < 0.05 else 'Fail to Reject H0'
        details_rows.append(details)
        results.append({
            'sample_size_per_class': n,
            'feature': feature,
            'class_a': class_a,
            'class_b': class_b,
            't_statistic': details['t_statistic'],
            'p_value': details['p_value'],
            'cohens_d': details['cohens_d'],
            'decision_alpha_0_05': details['decision_alpha_0_05'],
        })

    details_df = pd.DataFrame(details_rows)
    results_df = pd.DataFrame(results)

    plot_df = pd.DataFrame({
        'value': np.concatenate([values_a, values_b]),
        'class': [class_a] * len(values_a) + [class_b] * len(values_b),
    })
    cmap = class_color_map([class_a, class_b])
    plt.figure(figsize=(7, 5))
    positions = [1, 2]
    grouped = [plot_df.loc[plot_df['class'] == class_a, 'value'], plot_df.loc[plot_df['class'] == class_b, 'value']]
    box = plt.boxplot(grouped, labels=[class_a, class_b], patch_artist=True)
    box['boxes'][0].set_facecolor(cmap[class_a])
    box['boxes'][1].set_facecolor(cmap[class_b])
    plt.ylabel(f'Standardized {feature}')
    plt.title(f'Class Comparison for {feature}')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '07_ttest_selected_feature_boxplot.png', dpi=300)
    plt.close()

    candidates.to_csv(OUTPUT_TABLE_DIR / '07_ttest_candidate_pairs.csv', index=False)
    results_df.to_csv(OUTPUT_TABLE_DIR / '07_ttest_results.csv', index=False)
    details_df.to_csv(OUTPUT_TABLE_DIR / '07_ttest_detailed_calculations.csv', index=False)

    text = f"""
Hypothesis Testing

CSV Path: {csv_path}
Target Column: {target_col}
Selected Feature: {feature}
Selected Class Pair: {class_a} vs {class_b}

Null Hypothesis (H0): The class means are equal.
Alternative Hypothesis (H1): The class means are different.

Selected Pair Ranking Basis
{candidates.head(15).to_string(index=False)}

T-Test Results
{results_df.to_string(index=False)}
"""
    save_text(text.strip() + '\n', OUTPUT_TEXT_DIR / '07_hypothesis_testing_summary.txt')

    print_section('07 HYPOTHESIS TESTING')
    print(text)


if __name__ == '__main__':
    main()
