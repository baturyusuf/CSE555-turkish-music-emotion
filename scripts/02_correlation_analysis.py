import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from src.common import ensure_dirs, load_dataset, infer_target_column, prepare_analysis_dataset, OUTPUT_FIG_DIR, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, correlation_ratio, save_text, print_section


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    prepared = prepare_analysis_dataset(df_raw, target_col)
    X = prepared['X_raw']
    y = prepared['y']
    numeric_features = prepared['numeric_features']

    corr = X.corr(method='pearson')
    corr_abs = corr.abs()
    upper_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    top_pairs = corr_abs.where(upper_mask).stack().reset_index()
    top_pairs.columns = ['feature_1', 'feature_2', 'abs_correlation']
    top_pairs['correlation'] = [corr.loc[r['feature_1'], r['feature_2']] for _, r in top_pairs.iterrows()]
    top_pairs = top_pairs.sort_values('abs_correlation', ascending=False)

    y_codes = pd.Categorical(y).codes
    f_vals, p_vals = f_classif(X, y_codes)
    eta_vals = [correlation_ratio(y, X[col]) for col in numeric_features]
    assoc_df = pd.DataFrame({
        'feature': numeric_features,
        'anova_f_value': f_vals,
        'anova_p_value': p_vals,
        'correlation_ratio_eta': eta_vals,
    }).sort_values(['correlation_ratio_eta', 'anova_f_value'], ascending=False)

    plt.figure(figsize=(14, 12))
    plt.imshow(corr, aspect='auto')
    plt.colorbar(label='Pearson Correlation')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.title('Feature-Feature Pearson Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '02_feature_correlation_heatmap.png', dpi=300)
    plt.close()

    top_15 = assoc_df.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_15['feature'], top_15['correlation_ratio_eta'])
    plt.xlabel('Correlation Ratio (η)')
    plt.ylabel('Feature')
    plt.title('Top 15 Feature-Class Associations')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '02_feature_class_association_barh.png', dpi=300)
    plt.close()

    corr.to_csv(OUTPUT_TABLE_DIR / '02_feature_correlation_matrix.csv')
    top_pairs.head(50).to_csv(OUTPUT_TABLE_DIR / '02_top_absolute_correlations.csv', index=False)
    assoc_df.to_csv(OUTPUT_TABLE_DIR / '02_feature_class_association.csv', index=False)

    text = f"""
Correlation Analysis

CSV Path: {csv_path}
Target Column: {target_col}
Rows Used: {len(X)}
Numerical Features: {len(numeric_features)}

Top 15 Absolute Feature-Feature Correlations
{top_pairs.head(15).to_string(index=False)}

Top 15 Feature-Class Associations
{assoc_df.head(15).to_string(index=False)}
"""
    save_text(text.strip() + '\n', OUTPUT_TEXT_DIR / '02_correlation_analysis_summary.txt')

    print_section('02 CORRELATION ANALYSIS')
    print(text)


if __name__ == '__main__':
    main()
