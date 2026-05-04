# Turkish Music Emotion Analysis

This repository provides a complete analysis pipeline for the Turkish Music Emotion dataset with publication-style outputs suitable for a formal report and presentation.

## Dataset Placement

Put the dataset CSV file into:

```text
data/raw/
```

Example:

```text
data/raw/Acoustic Features.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Entire Pipeline

```bash
python scripts/99_run_all.py --target Class
```

If the target column is inferred correctly, `--target Class` is optional.

## Run Scripts Individually

```bash
python scripts/01_data_quality.py --target Class
python scripts/02_correlation_analysis.py --target Class
python scripts/03_feature_discriminability.py --target Class
python scripts/04_pca_analysis.py --target Class
python scripts/05_lda_analysis.py --target Class
python scripts/06_clustering_analysis.py --target Class
python scripts/07_hypothesis_testing.py --target Class
python scripts/08_manual_calculation_support.py --target Class
```

## Output Structure

- `outputs/figures/`: figures for the report and presentation
- `outputs/tables/`: CSV tables for direct inclusion in the report
- `outputs/text/`: concise interpretive summaries
- `data/interim/`: cleaned analysis-ready dataset

## Analysis Coverage

1. Data quality assessment, duplicate removal, outlier handling, raw and cleaned boxplots
2. Feature-feature correlation and feature-class association analysis
3. Z-score normalization and Fisher score analysis
4. PCA transformation, Fisher scores of principal components, eigenvalue-discriminability comparison
5. PCA and LDA 2D projections with separability metrics
6. K-Means, DBSCAN, t-SNE, and SOM outputs
7. Hypothesis testing with 36 and 64 samples per class using Welch’s t-test
8. Manual-calculation support tables for Fisher score, t-test, and PCA
