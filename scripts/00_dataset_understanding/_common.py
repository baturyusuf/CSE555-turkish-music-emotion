from pathlib import Path
import json
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables" / "eda"
REPORT_TEXT_DIR = PROJECT_ROOT / "reports" / "text" / "eda"
REPORT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "eda"


def ensure_output_dirs():
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def find_csv_file(csv_path=None, data_dir=None):
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path

    search_dir = Path(data_dir) if data_dir is not None else DATA_RAW_DIR

    if not search_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {search_dir}")

    csv_files = list(search_dir.rglob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV file found under {search_dir}. Put the dataset CSV into data/raw/."
        )

    if len(csv_files) > 1:
        csv_files = sorted(csv_files, key=lambda p: p.stat().st_size, reverse=True)
        print("[WARNING] Multiple CSV files found. The largest one will be used:")
        for file in csv_files:
            print(f"  - {file} | size={file.stat().st_size} bytes")

    return csv_files[0]


def load_dataset(csv_path=None, data_dir=None):
    path = find_csv_file(csv_path=csv_path, data_dir=data_dir)

    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)

    return df, path


def infer_target_column(df, target_col=None):
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
            )
        return target_col

    name_candidates = [
        "class", "Class", "CLASS",
        "label", "Label", "LABEL",
        "target", "Target", "TARGET",
        "emotion", "Emotion", "EMOTION",
        "mood", "Mood", "MOOD"
    ]

    for col in name_candidates:
        if col in df.columns:
            return col

    possible_targets = []

    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        ratio_unique = n_unique / max(len(df), 1)

        if 2 < n_unique <= 20 and ratio_unique < 0.2:
            possible_targets.append((col, n_unique, str(df[col].dtype)))

    if len(possible_targets) == 1:
        return possible_targets[0][0]

    if len(possible_targets) > 1:
        print("[WARNING] Multiple possible target columns found:")
        for col, n_unique, dtype in possible_targets:
            print(f"  - {col} | unique={n_unique} | dtype={dtype}")
        print("[INFO] The first candidate will be used.")
        return possible_targets[0][0]

    raise ValueError(
        "Target column could not be inferred. Please run the scripts with --target TARGET_COLUMN_NAME."
    )


def get_numeric_feature_columns(df, target_col):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_cols if col != target_col]
    return numeric_features


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
