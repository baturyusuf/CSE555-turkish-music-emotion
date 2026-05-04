import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.common import ensure_dirs, load_dataset, infer_target_column, prepare_analysis_dataset, OUTPUT_FIG_DIR, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR, multiclass_fisher_score, pairwise_fisher_distance, save_text, print_section


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()

    ensure_dirs()
    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    prepared = prepare_analysis_dataset(df_raw, target_col)
    X_scaled_df = prepared['X_scaled_df']
    y = prepared['y']
    numeric_features = prepared['numeric_features']

    fisher_df = multiclass_fisher_score(X_scaled_df, y).reset_index()
    fisher_df.columns = ['feature', 'multiclass_fisher_score']

    classes = sorted(y.astype(str).unique())
    rows = []
    for i, cls_a in enumerate(classes):
        for cls_b in classes[i + 1:]:
            mask_a = y.astype(str) == cls_a
            mask_b = y.astype(str) == cls_b
            for feature in numeric_features:
                fd = pairwise_fisher_distance(X_scaled_df.loc[mask_a, feature], X_scaled_df.loc[mask_b, feature])
                rows.append({
                    'class_a': cls_a,
                    'class_b': cls_b,
                    'feature': feature,
                    'pairwise_fisher_distance': fd,
                })
    pairwise_df = pd.DataFrame(rows).sort_values('pairwise_fisher_distance', ascending=False)

    top_15 = fisher_df.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_15['feature'], top_15['multiclass_fisher_score'])
    plt.xlabel('Multi-Class Fisher Score')
    plt.ylabel('Feature')
    plt.title('Top 15 Discriminative Features')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / '03_top_fisher_features_barh.png', dpi=300)
    plt.close()

    fisher_df.to_csv(OUTPUT_TABLE_DIR / '03_multiclass_fisher_scores.csv', index=False)
    pairwise_df.to_csv(OUTPUT_TABLE_DIR / '03_pairwise_fisher_distances.csv', index=False)

    text = f"""
Feature Discriminability

CSV Path: {csv_path}
Target Column: {target_col}
Rows Used: {len(X_scaled_df)}
Numerical Features: {len(numeric_features)}

Top 15 Features by Multi-Class Fisher Score
{fisher_df.head(15).to_string(index=False)}

Top 20 Pairwise Fisher Distances
{pairwise_df.head(20).to_string(index=False)}
"""
    save_text(text.strip() + '\n', OUTPUT_TEXT_DIR / '03_feature_discriminability_summary.txt')

    print_section('03 FEATURE DISCRIMINABILITY')
    print(text)


if __name__ == '__main__':
    main()
