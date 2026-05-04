from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_FIG_DIR = PROJECT_ROOT / 'outputs' / 'figures'
OUTPUT_TABLE_DIR = PROJECT_ROOT / 'outputs' / 'tables'
OUTPUT_TEXT_DIR = PROJECT_ROOT / 'outputs' / 'text'


def ensure_dirs():
    for path in [DATA_INTERIM_DIR, DATA_PROCESSED_DIR, OUTPUT_FIG_DIR, OUTPUT_TABLE_DIR, OUTPUT_TEXT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_text(text, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def print_section(title):
    print('\n' + '=' * 90)
    print(title)
    print('=' * 90)


def find_csv(csv_path=None):
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    csv_files = sorted(DATA_RAW_DIR.rglob('*.csv'))
    if not csv_files:
        raise FileNotFoundError('No CSV file found under data/raw')
    return csv_files[0]


def load_dataset(csv_path=None):
    path = find_csv(csv_path)
    try:
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(path)
    return df, path


def infer_target_column(df, target=None):
    if target:
        if target not in df.columns:
            raise ValueError(f'Target column not found: {target}')
        return target
    candidates = ['Class', 'class', 'label', 'Label', 'target', 'Target', 'emotion', 'Emotion']
    for col in candidates:
        if col in df.columns:
            return col
    possibles = []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if 2 < n_unique <= 20 and n_unique / len(df) < 0.2:
            possibles.append(col)
    if len(possibles) == 1:
        return possibles[0]
    raise ValueError('Target column could not be inferred.')


def get_numeric_features(df, target_col):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c != target_col]


def correlation_ratio(categories, values):
    categories = pd.Series(categories).astype(str)
    values = pd.Series(values).astype(float)
    grand_mean = values.mean()
    classes = categories.unique()
    numerator = 0.0
    denominator = ((values - grand_mean) ** 2).sum()
    for cls in classes:
        vals = values[categories == cls]
        numerator += len(vals) * (vals.mean() - grand_mean) ** 2
    return np.sqrt(numerator / (denominator + 1e-12))


def multiclass_fisher_score(X_df, y):
    y = pd.Series(y).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)
    scores = {}
    for feature in X_df.columns:
        x = X_df[feature]
        mu = x.mean()
        between = 0.0
        within = 0.0
        for cls in y.unique():
            x_cls = x[y == cls]
            mu_c = x_cls.mean()
            between += len(x_cls) * (mu_c - mu) ** 2
            within += ((x_cls - mu_c) ** 2).sum()
        scores[feature] = between / (within + 1e-12)
    return pd.Series(scores).sort_values(ascending=False)


def pairwise_fisher_distance(values_a, values_b):
    values_a = pd.Series(values_a).astype(float)
    values_b = pd.Series(values_b).astype(float)
    mu_a = values_a.mean()
    mu_b = values_b.mean()
    var_a = values_a.var(ddof=1)
    var_b = values_b.var(ddof=1)
    return ((mu_a - mu_b) ** 2) / (var_a + var_b + 1e-12)


def cohens_d(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n1, n2 = len(values_a), len(values_b)
    s1 = values_a.var(ddof=1)
    s2 = values_b.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2 + 1e-12))
    return (values_a.mean() - values_b.mean()) / (pooled + 1e-12)


def class_color_map(classes):
    base = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    classes = list(sorted(map(str, classes)))
    return {cls: base[i % len(base)] for i, cls in enumerate(classes)}


def scatter_by_class(df_plot, x_col, y_col, class_col, title, filename, xlabel=None, ylabel=None):
    cmap = class_color_map(df_plot[class_col].unique())
    plt.figure(figsize=(8, 6))
    for cls in sorted(df_plot[class_col].astype(str).unique()):
        subset = df_plot[df_plot[class_col].astype(str) == cls]
        plt.scatter(subset[x_col], subset[y_col], s=42, alpha=0.75, label=cls, color=cmap[cls])
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / filename, dpi=300)
    plt.close()


def scatter_by_label(df_plot, x_col, y_col, label_col, title, filename, xlabel=None, ylabel=None):
    labels = df_plot[label_col].astype(str).unique().tolist()
    cmap = class_color_map(labels)
    plt.figure(figsize=(8, 6))
    for lbl in sorted(df_plot[label_col].astype(str).unique()):
        subset = df_plot[df_plot[label_col].astype(str) == lbl]
        plt.scatter(subset[x_col], subset[y_col], s=42, alpha=0.75, label=lbl, color=cmap[lbl])
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / filename, dpi=300)
    plt.close()


def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def iqr_cap_dataframe(df, columns):
    capped = df.copy()
    rows = []
    for col in columns:
        s = capped[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before_lower = int((s < lower).sum())
        before_upper = int((s > upper).sum())
        capped[col] = s.clip(lower=lower, upper=upper)
        after_s = capped[col]
        changed = int((s != after_s).sum())
        rows.append({
            'feature': col,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower,
            'upper_bound': upper,
            'below_lower_before': before_lower,
            'above_upper_before': before_upper,
            'total_capped_values': changed,
        })
    return capped, pd.DataFrame(rows)


def prepare_analysis_dataset(df, target_col):
    numeric_features = get_numeric_features(df, target_col)
    df = remove_duplicates(df)
    df, cap_summary = iqr_cap_dataframe(df, numeric_features)
    X = df[numeric_features].copy()
    y = df[target_col].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_features, index=df.index)
    return {
        'df_clean': df,
        'X_raw': X,
        'y': y,
        'numeric_features': numeric_features,
        'X_scaled': X_scaled,
        'X_scaled_df': X_scaled_df,
        'scaler': scaler,
        'cap_summary': cap_summary,
    }
