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

from src.common import (
    ensure_dirs,
    load_dataset,
    infer_target_column,
    prepare_analysis_dataset,
    OUTPUT_FIG_DIR,
    OUTPUT_TABLE_DIR,
    OUTPUT_TEXT_DIR,
    multiclass_fisher_score,
    pairwise_fisher_distance,
    cohens_d,
    class_color_map,
    save_text,
    print_section,
)


def hedges_g(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_total = len(a) + len(b)
    d = cohens_d(a, b)
    correction = 1 - (3 / (4 * n_total - 9))
    return d * correction


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted = np.empty(n, dtype=float)
    cumulative_min = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        cumulative_min = min(cumulative_min, value)
        adjusted[order[i]] = min(cumulative_min, 1.0)

    return adjusted


def select_candidate_pairs(X_scaled_df, y):
    y = pd.Series(y).astype(str).reset_index(drop=True)
    X_scaled_df = X_scaled_df.reset_index(drop=True)

    fisher_series = multiclass_fisher_score(X_scaled_df, y)
    fisher_df = fisher_series.reset_index()
    fisher_df.columns = ["feature", "multiclass_fisher_score"]

    classes = sorted(y.unique())
    rows = []

    for feature in X_scaled_df.columns:
        for i, class_a in enumerate(classes):
            for class_b in classes[i + 1:]:
                a = X_scaled_df.loc[y == class_a, feature].to_numpy()
                b = X_scaled_df.loc[y == class_b, feature].to_numpy()

                welch = stats.ttest_ind(a, b, equal_var=False)

                rows.append({
                    "feature": feature,
                    "class_a": class_a,
                    "class_b": class_b,
                    "n_class_a_full": len(a),
                    "n_class_b_full": len(b),
                    "mean_class_a_full": float(np.mean(a)),
                    "mean_class_b_full": float(np.mean(b)),
                    "mean_difference_full": float(np.mean(a) - np.mean(b)),
                    "pairwise_fisher_distance": float(pairwise_fisher_distance(a, b)),
                    "cohens_d_full": float(cohens_d(a, b)),
                    "cohens_d_abs_full": float(abs(cohens_d(a, b))),
                    "welch_t_full": float(welch.statistic),
                    "welch_p_full": float(welch.pvalue),
                })

    candidate_df = pd.DataFrame(rows)
    candidate_df["bonferroni_p_full"] = np.minimum(candidate_df["welch_p_full"] * len(candidate_df), 1.0)
    candidate_df["bh_fdr_p_full"] = benjamini_hochberg(candidate_df["welch_p_full"].values)

    candidate_df = candidate_df.merge(fisher_df, on="feature", how="left")

    candidate_df = candidate_df.sort_values(
        ["pairwise_fisher_distance", "cohens_d_abs_full", "multiclass_fisher_score"],
        ascending=False,
    ).reset_index(drop=True)

    return candidate_df, fisher_df


def welch_test_details(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n1, n2 = len(a), len(b)
    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)

    welch = stats.ttest_ind(a, b, equal_var=False)
    student = stats.ttest_ind(a, b, equal_var=True)

    se = np.sqrt(var1 / n1 + var2 / n2)
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
    welch_df = df_num / (df_den + 1e-12)

    ci_delta = stats.t.ppf(0.975, welch_df) * se
    mean_diff = mean1 - mean2

    shapiro_a = stats.shapiro(a)
    shapiro_b = stats.shapiro(b)
    levene = stats.levene(a, b, center="median")

    return {
        "n_class_a": int(n1),
        "n_class_b": int(n2),
        "mean_class_a": float(mean1),
        "mean_class_b": float(mean2),
        "var_class_a": float(var1),
        "var_class_b": float(var2),
        "mean_difference": float(mean_diff),
        "standard_error": float(se),
        "welch_df": float(welch_df),
        "student_t_statistic": float(student.statistic),
        "student_p_value": float(student.pvalue),
        "welch_t_statistic": float(welch.statistic),
        "welch_p_value": float(welch.pvalue),
        "ci_95_low": float(mean_diff - ci_delta),
        "ci_95_high": float(mean_diff + ci_delta),
        "cohens_d": float(cohens_d(a, b)),
        "hedges_g": float(hedges_g(a, b)),
        "pairwise_fisher_distance": float(pairwise_fisher_distance(a, b)),
        "shapiro_class_a_statistic": float(shapiro_a.statistic),
        "shapiro_class_a_p_value": float(shapiro_a.pvalue),
        "shapiro_class_b_statistic": float(shapiro_b.statistic),
        "shapiro_class_b_p_value": float(shapiro_b.pvalue),
        "levene_statistic": float(levene.statistic),
        "levene_p_value": float(levene.pvalue),
    }


def draw_balanced_samples(values_a, values_b, mode, n_value, seed):
    rng = np.random.default_rng(seed)

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    if mode == "per_class":
        n_a = n_value
        n_b = n_value
        interpretation = f"n={n_value} samples were drawn from each class."
    elif mode == "total_balanced":
        n_a = n_value // 2
        n_b = n_value - n_a
        interpretation = f"n={n_value} total samples were drawn as a balanced two-class sample: {n_a}+{n_b}."
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    if n_a > len(values_a) or n_b > len(values_b):
        raise ValueError(
            f"Requested sample sizes exceed available class sizes. "
            f"Requested: {n_a}, {n_b}; Available: {len(values_a)}, {len(values_b)}"
        )

    sample_a = rng.choice(values_a, size=n_a, replace=False)
    sample_b = rng.choice(values_b, size=n_b, replace=False)

    return sample_a, sample_b, interpretation


def save_sample_distribution_figure(sample_a, sample_b, class_a, class_b, feature, mode, n_value, filename):
    cmap = class_color_map([class_a, class_b])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    box = ax.boxplot(
        [sample_a, sample_b],
        tick_labels=[class_a, class_b],
        patch_artist=True,
        showfliers=True,
    )

    box["boxes"][0].set_facecolor(cmap[class_a])
    box["boxes"][1].set_facecolor(cmap[class_b])

    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(1, 0.035, len(sample_a)), sample_a, s=24, alpha=0.55, color=cmap[class_a])
    ax.scatter(rng.normal(2, 0.035, len(sample_b)), sample_b, s=24, alpha=0.55, color=cmap[class_b])

    ax.set_title(f"Sample Distribution for {feature}\n{mode}, n={n_value}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Standardized feature value")

    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / filename, dpi=300)
    plt.close(fig)


def save_qq_figure(sample_a, sample_b, class_a, class_b, feature, mode, n_value, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    stats.probplot(sample_a, dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q Plot: {class_a}")

    stats.probplot(sample_b, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot: {class_b}")

    fig.suptitle(f"Normality Check for {feature}\n{mode}, n={n_value}", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / filename, dpi=300)
    plt.close(fig)


def repeated_sampling(values_a, values_b, feature, class_a, class_b, mode, n_value, repetitions, seed):
    rows = []

    for iteration in range(repetitions):
        sample_seed = seed + iteration
        sample_a, sample_b, interpretation = draw_balanced_samples(
            values_a,
            values_b,
            mode=mode,
            n_value=n_value,
            seed=sample_seed,
        )

        details = welch_test_details(sample_a, sample_b)

        rows.append({
            "iteration": iteration + 1,
            "sampling_mode": mode,
            "sample_size_requested": n_value,
            "sampling_interpretation": interpretation,
            "feature": feature,
            "class_a": class_a,
            "class_b": class_b,
            "sample_seed": sample_seed,
            "reject_h0_alpha_0_05": int(details["welch_p_value"] < 0.05),
            **details,
        })

    return pd.DataFrame(rows)


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

    df_raw, csv_path = load_dataset(args.csv)
    target_col = infer_target_column(df_raw, args.target)

    prepared = prepare_analysis_dataset(df_raw, target_col)
    X_scaled_df = prepared["X_scaled_df"].reset_index(drop=True)
    y = prepared["y"].astype(str).reset_index(drop=True)

    candidate_df, fisher_df = select_candidate_pairs(X_scaled_df, y)

    if args.feature and args.class_a and args.class_b:
        feature = args.feature
        class_a = args.class_a
        class_b = args.class_b

        if feature not in X_scaled_df.columns:
            raise ValueError(f"Feature not found: {feature}")

        if class_a not in set(y) or class_b not in set(y):
            raise ValueError(f"Class pair not found: {class_a}, {class_b}")

        selected_source = "manual_argument"
    else:
        best = candidate_df.iloc[0]
        feature = best["feature"]
        class_a = best["class_a"]
        class_b = best["class_b"]
        selected_source = "automatic_highest_pairwise_fisher_distance"

    values_a = X_scaled_df.loc[y == class_a, feature].to_numpy()
    values_b = X_scaled_df.loc[y == class_b, feature].to_numpy()

    all_test_rows = []
    repeated_frames = []
    stability_rows = []

    sampling_plan = [
        ("per_class", 36),
        ("per_class", 64),
        ("total_balanced", 36),
        ("total_balanced", 64),
    ]

    for plan_index, (mode, n_value) in enumerate(sampling_plan):
        sample_seed = args.random_state + 1000 * (plan_index + 1)

        sample_a, sample_b, interpretation = draw_balanced_samples(
            values_a,
            values_b,
            mode=mode,
            n_value=n_value,
            seed=sample_seed,
        )

        details = welch_test_details(sample_a, sample_b)

        all_test_rows.append({
            "sampling_mode": mode,
            "sample_size_requested": n_value,
            "sampling_interpretation": interpretation,
            "feature": feature,
            "class_a": class_a,
            "class_b": class_b,
            "random_state": sample_seed,
            "decision_alpha_0_05": "Reject H0" if details["welch_p_value"] < 0.05 else "Fail to Reject H0",
            **details,
        })

        save_sample_distribution_figure(
            sample_a,
            sample_b,
            class_a,
            class_b,
            feature,
            mode,
            n_value,
            f"07_ttest_sample_distribution_{mode}_n{n_value}.png",
        )

        save_qq_figure(
            sample_a,
            sample_b,
            class_a,
            class_b,
            feature,
            mode,
            n_value,
            f"07_ttest_qq_plots_{mode}_n{n_value}.png",
        )

        repeated_df = repeated_sampling(
            values_a,
            values_b,
            feature,
            class_a,
            class_b,
            mode,
            n_value,
            repetitions=args.repetitions,
            seed=args.random_state + 10000 * (plan_index + 1),
        )

        repeated_frames.append(repeated_df)

        stability_rows.append({
            "sampling_mode": mode,
            "sample_size_requested": n_value,
            "repetitions": args.repetitions,
            "rejection_rate_alpha_0_05": float(repeated_df["reject_h0_alpha_0_05"].mean()),
            "median_p_value": float(repeated_df["welch_p_value"].median()),
            "p_value_q1": float(repeated_df["welch_p_value"].quantile(0.25)),
            "p_value_q3": float(repeated_df["welch_p_value"].quantile(0.75)),
            "mean_cohens_d": float(repeated_df["cohens_d"].mean()),
            "std_cohens_d": float(repeated_df["cohens_d"].std()),
            "mean_hedges_g": float(repeated_df["hedges_g"].mean()),
            "std_hedges_g": float(repeated_df["hedges_g"].std()),
            "mean_difference_mean": float(repeated_df["mean_difference"].mean()),
            "mean_difference_std": float(repeated_df["mean_difference"].std()),
        })

    results_df = pd.DataFrame(all_test_rows)
    repeated_all_df = pd.concat(repeated_frames, ignore_index=True)
    stability_df = pd.DataFrame(stability_rows)

    candidate_df.to_csv(OUTPUT_TABLE_DIR / "07_all_pairwise_feature_tests_with_multiple_testing.csv", index=False)
    fisher_df.to_csv(OUTPUT_TABLE_DIR / "07_multiclass_fisher_scores_for_ttest_selection.csv", index=False)
    results_df.to_csv(OUTPUT_TABLE_DIR / "07_ttest_results_both_sample_interpretations.csv", index=False)
    repeated_all_df.to_csv(OUTPUT_TABLE_DIR / "07_repeated_sampling_ttest_results.csv", index=False)
    stability_df.to_csv(OUTPUT_TABLE_DIR / "07_repeated_sampling_stability_summary.csv", index=False)

    # Separate convenient tables for paper insertion
    results_df[results_df["sampling_mode"] == "per_class"].to_csv(
        OUTPUT_TABLE_DIR / "07_ttest_results_per_class_interpretation.csv",
        index=False,
    )

    results_df[results_df["sampling_mode"] == "total_balanced"].to_csv(
        OUTPUT_TABLE_DIR / "07_ttest_results_total_sample_interpretation.csv",
        index=False,
    )

    # Stability figures
    plt.figure(figsize=(8, 5))
    for mode, n_value in sampling_plan:
        subset = repeated_all_df[
            (repeated_all_df["sampling_mode"] == mode)
            & (repeated_all_df["sample_size_requested"] == n_value)
        ]
        transformed_p = -np.log10(subset["welch_p_value"].clip(lower=1e-300))
        plt.hist(transformed_p, bins=30, alpha=0.45, label=f"{mode}, n={n_value}")

    plt.xlabel("-log10(Welch p-value)")
    plt.ylabel("Frequency")
    plt.title("Repeated Sampling Stability of Welch t-test p-values")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "07_repeated_sampling_pvalue_stability.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    repeated_all_df.boxplot(column="cohens_d", by=["sampling_mode", "sample_size_requested"], rot=25)
    plt.title("Cohen's d Stability Across Repeated Random Draws")
    plt.suptitle("")
    plt.xlabel("Sampling mode and requested n")
    plt.ylabel("Cohen's d")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "07_repeated_sampling_effect_size_stability.png", dpi=300)
    plt.close()

    selected_full_row = candidate_df[
        (candidate_df["feature"] == feature)
        & (candidate_df["class_a"] == class_a)
        & (candidate_df["class_b"] == class_b)
    ]

    if selected_full_row.empty:
        selected_full_row_text = "Selected manual pair was not found in automatic candidate table."
    else:
        selected_full_row_text = selected_full_row.head(1).to_string(index=False)

    equations = f"""
Hypothesis Testing Notes

Selected feature:
{feature}

Selected class pair:
{class_a} vs {class_b}

Selection source:
{selected_source}

Important interpretation note:
The feature and class pair were selected after inspecting separability scores. Therefore, the t-test should be interpreted as an exploratory statistical check rather than a fully pre-registered confirmatory test.

Sampling interpretation note:
Both possible interpretations of the assignment statement were evaluated:
1. per_class: n=36 and n=64 samples drawn from each class.
2. total_balanced: n=36 and n=64 total samples split as balanced two-class samples.

Hypotheses:
H0: μ_{class_a} = μ_{class_b}
H1: μ_{class_a} ≠ μ_{class_b}

Welch t statistic:
t = (x̄1 - x̄2) / sqrt(s1²/n1 + s2²/n2)

Welch-Satterthwaite degrees of freedom:
df = (s1²/n1 + s2²/n2)² / [ (s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1) ]

Pairwise Fisher Distance:
FD = (μ1 - μ2)² / (σ1² + σ2²)

Cohen's d:
d = (x̄1 - x̄2) / sp
""".strip()

    save_text(equations + "\n", OUTPUT_TEXT_DIR / "07_hypothesis_testing_equations_and_notes.txt")

    summary = f"""
Hypothesis Testing

CSV Path:
{csv_path}

Target Column:
{target_col}

Rows Used:
{len(y)}

Selected Feature:
{feature}

Selected Class Pair:
{class_a} vs {class_b}

Selection Source:
{selected_source}

Post-hoc / Exploratory Note:
The tested feature and class pair were selected based on separability analysis. Therefore, the t-test is interpreted as an exploratory statistical confirmation rather than a fully pre-registered confirmatory test.

Multiple Testing Control:
All feature-class pair Welch tests on the full cleaned normalized dataset were computed and corrected using Bonferroni and Benjamini-Hochberg FDR. The selected pair appears below.

Selected Pair Full-Data Row:
{selected_full_row_text}

Top 15 Candidate Feature-Class Pairs:
{candidate_df.head(15).to_string(index=False)}

t-test Results for Both Sampling Interpretations:
{results_df.to_string(index=False)}

Repeated Sampling Stability Summary:
{stability_df.to_string(index=False)}
""".strip()

    save_text(summary + "\n", OUTPUT_TEXT_DIR / "07_hypothesis_testing_summary.txt")

    print_section("07 HYPOTHESIS TESTING")
    print(summary)


if __name__ == "__main__":
    main()