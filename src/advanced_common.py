from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_TEXT_DIR = PROJECT_ROOT / "outputs" / "text"

DEFAULT_PALETTE = {
    "angry": "tab:blue",
    "happy": "tab:orange",
    "relax": "tab:green",
    "sad": "tab:red",
}

def ensure_dirs():
    for path in [OUTPUT_FIG_DIR, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

def configure_plots():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def print_section(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)

def find_csv(csv_path=None):
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    files = sorted(DATA_RAW_DIR.rglob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV file found under data/raw. Put Acoustic Features.csv there or pass --csv.")
    return files[0]

def load_dataset(csv_path=None):
    path = find_csv(csv_path)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    return df, path

def infer_target_column(df, target=None):
    if target:
        if target not in df.columns:
            raise ValueError(f"Target column not found: {target}")
        return target
    for col in ["Class", "class", "Label", "label", "Target", "target", "Emotion", "emotion"]:
        if col in df.columns:
            return col
    candidates = []
    for col in df.columns:
        n = df[col].nunique(dropna=True)
        if 2 < n <= 20 and n / max(len(df), 1) < 0.2:
            candidates.append(col)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Target column could not be inferred. Use --target Class.")

def numeric_features(df, target_col):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)

def iqr_capping(df, columns):
    out = df.copy()
    rows = []
    for col in columns:
        s = out[col].astype(float)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        below = int((s < lower).sum())
        above = int((s > upper).sum())
        capped = s.clip(lower=lower, upper=upper)
        changed = int((s != capped).sum())
        out[col] = capped
        rows.append({
            "feature": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "below_lower_before": below,
            "above_upper_before": above,
            "total_capped_values": changed,
        })
    return out, pd.DataFrame(rows)

def get_variant(df_raw, target_col, variant):
    cols = numeric_features(df_raw, target_col)
    if variant == "raw":
        df = df_raw.copy().reset_index(drop=True)
        cap_summary = pd.DataFrame()
    elif variant == "duplicate_cleaned":
        df = remove_duplicates(df_raw)
        cap_summary = pd.DataFrame()
    elif variant == "duplicate_cleaned_iqr_capped":
        df = remove_duplicates(df_raw)
        df, cap_summary = iqr_capping(df, cols)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return df, cap_summary

def prepared_variant(df_raw, target_col, variant="duplicate_cleaned_iqr_capped"):
    df, cap_summary = get_variant(df_raw, target_col, variant)
    cols = numeric_features(df, target_col)
    X_raw = df[cols].astype(float).copy()
    y = df[target_col].astype(str).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=cols)
    return {
        "df": df,
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "X_scaled_df": X_scaled_df,
        "y": y,
        "features": cols,
        "cap_summary": cap_summary,
    }

def multiclass_fisher_score(X_df, y):
    y = pd.Series(y).astype(str).reset_index(drop=True)
    X = X_df.reset_index(drop=True)
    scores = {}
    details = []
    for feature in X.columns:
        x = X[feature].astype(float)
        overall = x.mean()
        between = 0.0
        within = 0.0
        for cls in sorted(y.unique()):
            xc = x[y == cls]
            mu_c = xc.mean()
            bc = len(xc) * (mu_c - overall) ** 2
            wc = ((xc - mu_c) ** 2).sum()
            between += bc
            within += wc
            details.append({
                "feature": feature,
                "class": cls,
                "n": len(xc),
                "class_mean": mu_c,
                "overall_mean": overall,
                "between_contribution": bc,
                "within_contribution": wc,
            })
        scores[feature] = between / (within + 1e-12)
    score_df = pd.Series(scores).sort_values(ascending=False).reset_index()
    score_df.columns = ["feature", "multiclass_fisher_score"]
    return score_df, pd.DataFrame(details)

def pairwise_fisher_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return ((a.mean() - b.mean()) ** 2) / (a.var(ddof=1) + b.var(ddof=1) + 1e-12)

def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2 + 1e-12))
    return (a.mean() - b.mean()) / (pooled + 1e-12)

def hedges_g(a, b):
    d = cohens_d(a, b)
    n = len(a) + len(b)
    correction = 1 - (3 / (4 * n - 9))
    return d * correction

def pairwise_fisher_table(X_df, y):
    y = pd.Series(y).astype(str).reset_index(drop=True)
    classes = sorted(y.unique())
    rows = []
    for i, ca in enumerate(classes):
        for cb in classes[i + 1:]:
            ma = y == ca
            mb = y == cb
            for feature in X_df.columns:
                a = X_df.loc[ma, feature]
                b = X_df.loc[mb, feature]
                rows.append({
                    "class_a": ca,
                    "class_b": cb,
                    "feature": feature,
                    "pairwise_fisher_distance": pairwise_fisher_distance(a, b),
                    "cohens_d": cohens_d(a, b),
                    "cohens_d_abs": abs(cohens_d(a, b)),
                })
    return pd.DataFrame(rows).sort_values("pairwise_fisher_distance", ascending=False)

def separability_metrics(X, y):
    y = pd.Series(y).astype(str)
    values = {
        "silhouette_score": np.nan,
        "davies_bouldin_score": np.nan,
        "calinski_harabasz_score": np.nan,
    }
    if len(set(y)) > 1:
        values["silhouette_score"] = float(silhouette_score(X, y))
        values["davies_bouldin_score"] = float(davies_bouldin_score(X, y))
        values["calinski_harabasz_score"] = float(calinski_harabasz_score(X, y))
    return values

def pca_embedding(X_scaled, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    return pca, pd.DataFrame(X_pca, columns=cols)

def lda_embedding(X_scaled, y, n_components=2):
    n_classes = len(pd.Series(y).astype(str).unique())
    max_components = min(n_classes - 1, X_scaled.shape[1])
    n_components = min(n_components, max_components)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y)
    cols = [f"LD{i+1}" for i in range(X_lda.shape[1])]
    return lda, pd.DataFrame(X_lda, columns=cols)

def color_map(classes):
    classes = sorted(pd.Series(classes).astype(str).unique())
    palette = list(DEFAULT_PALETTE.values()) + ["tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    return {cls: DEFAULT_PALETTE.get(cls, palette[i % len(palette)]) for i, cls in enumerate(classes)}

def scatter_by_class(ax, data, x, y, label_col, title=None, xlabel=None, ylabel=None):
    cmap = color_map(data[label_col])
    for cls in sorted(data[label_col].astype(str).unique()):
        part = data[data[label_col].astype(str) == cls]
        ax.scatter(part[x], part[y], s=34, alpha=0.78, label=cls, color=cmap[cls], edgecolor="none")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.legend(frameon=True)

def save_scatter(data, x, y, label_col, title, filename, xlabel=None, ylabel=None):
    configure_plots()
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    scatter_by_class(ax, data, x, y, label_col, title, xlabel, ylabel)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_DIR / filename, dpi=300)
    plt.close(fig)

def pca_summary_table(pca):
    ratios = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    return pd.DataFrame({
        "principal_component": [f"PC{i+1}" for i in range(len(ratios))],
        "eigenvalue": eigenvalues,
        "explained_variance_ratio": ratios,
        "cumulative_explained_variance_ratio": np.cumsum(ratios),
    })

def components_needed(summary, thresholds=(0.70, 0.80, 0.90, 0.95)):
    rows = []
    cum = summary["cumulative_explained_variance_ratio"].values
    for t in thresholds:
        rows.append({
            "threshold": t,
            "components_needed": int(np.searchsorted(cum, t) + 1),
        })
    return pd.DataFrame(rows)

def safe_cluster_silhouette(X, labels):
    labels = np.asarray(labels)
    unique = set(labels)
    if len(unique) <= 1:
        return np.nan
    if len(unique) >= len(labels):
        return np.nan
    return float(silhouette_score(X, labels))
