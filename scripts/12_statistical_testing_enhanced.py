import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.advanced_common import (
    ensure_dirs, configure_plots, load_dataset, infer_target_column, prepared_variant,
    pairwise_fisher_table, cohens_d, hedges_g, pairwise_fisher_distance,
    OUTPUT_TABLE_DIR, OUTPUT_FIG_DIR, OUTPUT_TEXT_DIR, save_text, print_section
)

def welch_components(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    mean1, mean2 = a.mean(), b.mean()
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean1 - mean2) / (se + 1e-12)
    numerator = (var1 / n1 + var2 / n2) ** 2
    denominator = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
    df = numerator / (denominator + 1e-12)
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    ci_low = (mean1 - mean2) - stats.t.ppf(0.975, df) * se
    ci_high = (mean1 - mean2) + stats.t.ppf(0.975, df) * se
    return {
        "n1": n1,
        "n2": n2,
        "mean1": mean1,
        "mean2": mean2,
        "var1": var1,
        "var2": var2,
        "mean_difference": mean1 - mean2,
        "standard_error": se,
        "welch_t_statistic_manual": t_stat,
        "welch_df_manual": df,
        "welch_p_value_manual": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "cohens_d": cohens_d(a, b),
        "hedges_g": hedges_g(a, b),
        "pairwise_fisher_distance": pairwise_fisher_distance(a, b),
    }

def assumption_tests(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    shapiro_a = stats.shapiro(a)
    shapiro_b = stats.shapiro(b)
    levene = stats.levene(a, b, center="median")
    return {
        "shapiro_class_a_statistic": shapiro_a.statistic,
        "shapiro_class_a_p_value": shapiro_a.pvalue,
        "shapiro_class_b_statistic": shapiro_b.statistic,
        "shapiro_class_b_p_value": shapiro_b.pvalue,
        "levene_statistic": levene.statistic,
        "levene_p_value": levene.pvalue,
    }

def test_sample(a, b, feature, class_a, class_b, sample_size, random_state):
    rng = np.random.default_rng(random_state)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if sample_size > len(a) or sample_size > len(b):
        raise ValueError(f"Sample size {sample_size} exceeds class sizes {len(a)} and {len(b)}")
    sa = rng.choice(a, size=sample_size, replace=False)
    sb = rng.choice(b, size=sample_size, replace=False)

    student = stats.ttest_ind(sa, sb, equal_var=True)
    welch = stats.ttest_ind(sa, sb, equal_var=False)
    components = welch_components(sa, sb)
    assumptions = assumption_tests(sa, sb)

    return {
        "feature": feature,
        "class_a": class_a,
        "class_b": class_b,
        "sample_size_per_class": sample_size,
        "random_state": random_state,
        "student_t_statistic": student.statistic,
        "student_p_value": student.pvalue,
        "welch_t_statistic_scipy": welch.statistic,
        "welch_p_value_scipy": welch.pvalue,
        "decision_alpha_0_05": "Reject H0" if welch.pvalue < 0.05 else "Fail to Reject H0",
        **assumptions,
        **components,
    }, sa, sb

def repeated_sampling(a, b, feature, class_a, class_b, sample_size, repetitions, seed):
    rng = np.random.default_rng(seed)
    rows = []
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    for r in range(repetitions):
        sa = rng.choice(a, size=sample_size, replace=False)
        sb = rng.choice(b, size=sample_size, replace=False)
        welch = stats.ttest_ind(sa, sb, equal_var=False)
        comp = welch_components(sa, sb)
        rows.append({
            "iteration": r + 1,
            "feature": feature,
            "class_a": class_a,
            "class_b": class_b,
            "sample_size_per_class": sample_size,
            "welch_t_statistic": welch.statistic,
            "welch_p_value": welch.pvalue,
            "reject_h0_alpha_0_05": int(welch.pvalue < 0.05),
            "mean_difference": comp["mean_difference"],
            "cohens_d": comp["cohens_d"],
            "ci_95_low": comp["ci_95_low"],
            "ci_95_high": comp["ci_95_high"],
        })
    return pd.DataFrame(rows)

def save_distribution_figure(a, b, class_a, class_b, feature, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([a, b], tick_labels=[class_a, class_b], showfliers=True)
    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(1, 0.035, len(a)), a, s=22, alpha=0.55)
    ax.scatter(rng.normal(2, 0.035, len(b)), b, s=22, alpha=0.55)
    ax.set_title(f"Sample Distribution for {feature}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Standardized Feature Value")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / filename)
    plt.close(fig)

def save_qq_figure(a, b, class_a, class_b, feature, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    stats.probplot(a, dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q Plot: {class_a}")
    stats.probplot(b, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot: {class_b}")
    fig.suptitle(f"Normality Assessment for {feature}", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / filename)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--feature", type=str, default=None)
    parser.add_argument("--class-a", type=str, default=None)
    parser.add_argument("--class-b", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()

    ensure_dirs()
    configure_plots()

    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)
    data = prepared_variant(df_raw, target_col, "duplicate_cleaned_iqr_capped")
    X = data["X_scaled_df"]
    y = data["y"]

    pairwise_df = pairwise_fisher_table(X, y)

    if args.feature and args.class_a and args.class_b:
        feature = args.feature
        class_a = args.class_a
        class_b = args.class_b
    else:
        best = pairwise_df.iloc[0]
        feature = best["feature"]
        class_a = best["class_a"]
        class_b = best["class_b"]

    a_all = X.loc[y == class_a, feature].values
    b_all = X.loc[y == class_b, feature].values

    result_rows = []
    sample_store = {}
    for i, sample_size in enumerate([36, 64]):
        row, sa, sb = test_sample(a_all, b_all, feature, class_a, class_b, sample_size, args.random_state + i)
        result_rows.append(row)
        sample_store[sample_size] = (sa, sb)

    results_df = pd.DataFrame(result_rows)

    repeated_frames = []
    stability_rows = []
    for sample_size in [36, 64]:
        reps = repeated_sampling(a_all, b_all, feature, class_a, class_b, sample_size, args.repetitions, args.random_state + sample_size)
        repeated_frames.append(reps)
        stability_rows.append({
            "sample_size_per_class": sample_size,
            "repetitions": args.repetitions,
            "rejection_rate_alpha_0_05": reps["reject_h0_alpha_0_05"].mean(),
            "median_p_value": reps["welch_p_value"].median(),
            "p_value_q1": reps["welch_p_value"].quantile(0.25),
            "p_value_q3": reps["welch_p_value"].quantile(0.75),
            "mean_cohens_d": reps["cohens_d"].mean(),
            "std_cohens_d": reps["cohens_d"].std(),
            "mean_difference_mean": reps["mean_difference"].mean(),
            "mean_difference_std": reps["mean_difference"].std(),
        })

    repeated_df = pd.concat(repeated_frames, ignore_index=True)
    stability_df = pd.DataFrame(stability_rows)

    for sample_size, (sa, sb) in sample_store.items():
        save_distribution_figure(sa, sb, class_a, class_b, feature, f"12_ttest_sample_distribution_n{sample_size}.png")
        save_qq_figure(sa, sb, class_a, class_b, feature, f"12_ttest_qq_plots_n{sample_size}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for sample_size in [36, 64]:
        part = repeated_df[repeated_df["sample_size_per_class"] == sample_size]
        ax.hist(-np.log10(part["welch_p_value"].clip(lower=1e-300)), bins=35, alpha=0.6, label=f"n={sample_size}")
    ax.set_xlabel("-log10(p-value)")
    ax.set_ylabel("Frequency")
    ax.set_title("Repeated Random Sampling Stability of Welch t-test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "12_repeated_sampling_pvalue_stability.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    repeated_df.boxplot(column="cohens_d", by="sample_size_per_class", ax=ax)
    ax.set_title("Cohen's d Stability Across Random Draws")
    ax.set_xlabel("Sample Size per Class")
    ax.set_ylabel("Cohen's d")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / "12_repeated_sampling_effect_size_stability.png")
    plt.close(fig)

    results_df.to_csv(OUTPUT_TABLE_DIR / "12_ttest_assumption_and_result_table.csv", index=False)
    pairwise_df.head(50).to_csv(OUTPUT_TABLE_DIR / "12_candidate_feature_class_pairs_for_ttest.csv", index=False)
    repeated_df.to_csv(OUTPUT_TABLE_DIR / "12_repeated_sampling_ttest_results.csv", index=False)
    stability_df.to_csv(OUTPUT_TABLE_DIR / "12_repeated_sampling_stability_summary.csv", index=False)

    equations = """
Statistical Testing and Manual Calculation Notes

Hypotheses:
H0: μ_class_a = μ_class_b
H1: μ_class_a ≠ μ_class_b

Welch t statistic:
t = (x̄1 - x̄2) / sqrt(s1²/n1 + s2²/n2)

Welch-Satterthwaite degrees of freedom:
df = (s1²/n1 + s2²/n2)² / [ (s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1) ]

95% confidence interval for mean difference:
(x̄1 - x̄2) ± t_(0.975, df) * sqrt(s1²/n1 + s2²/n2)

Cohen's d:
d = (x̄1 - x̄2) / sp
sp = sqrt(((n1-1)s1² + (n2-1)s2²) / (n1+n2-2))

Pairwise Fisher Distance:
FD = (μ1 - μ2)² / (σ1² + σ2²)
"""
    save_text(equations.strip() + "\n", OUTPUT_TEXT_DIR / "12_statistical_testing_equations.txt")

    text = f"""
Enhanced Statistical Testing

CSV Path: {csv_path}
Target Column: {target_col}
Selected Feature: {feature}
Selected Class Pair: {class_a} vs {class_b}

Top Candidate Feature-Class Pairs
{pairwise_df.head(15).to_string(index=False)}

Assumption Tests and t-test Results
{results_df.to_string(index=False)}

Repeated Sampling Stability Summary
{stability_df.to_string(index=False)}
"""
    save_text(text.strip() + "\n", OUTPUT_TEXT_DIR / "12_enhanced_hypothesis_testing_summary.txt")

    print_section("12 ENHANCED STATISTICAL TESTING")
    print(text)

if __name__ == "__main__":
    main()
