# Acoustic Feature-Based Statistical Separability Analysis of Turkish Music Emotion Classes

> A statistical investigation of Turkish music emotion classes using acoustic descriptors, Fisher-based discriminability analysis, PCA, LDA, clustering, nonlinear embedding, SOM, and hypothesis testing.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Analysis](https://img.shields.io/badge/Analysis-Statistical%20Learning-informational)
![Dataset](https://img.shields.io/badge/Dataset-Turkish%20Music%20Emotion-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Overview

This project analyzes whether numerical acoustic descriptors can reveal meaningful separability among Turkish music emotion classes. The dataset contains four balanced emotion labels:

- **angry**
- **happy**
- **relax**
- **sad**

The study is designed as a **statistical data analysis project**, not only as a prediction task. The main focus is on feature discriminability, projection quality, class separability, clustering behavior, and inferential statistics.

### Main Questions

- Which acoustic features are the most discriminative?
- How does preprocessing affect separability?
- How do PCA and LDA compare in revealing emotion structure?
- Can clustering methods recover emotion-related grouping?
- Is there a statistically significant mean difference between selected emotion classes?

---

## Dataset

**Source:** [Turkish Music Emotion Dataset on Kaggle](https://www.kaggle.com/datasets/blaler/turkish-music-emotion-dataset)

Expected raw file location:

```text
data/raw/Acoustic Features.csv
```

### Dataset Summary

| Property | Value |
|---|---:|
| Number of raw samples | 400 |
| Number of columns | 51 |
| Numerical features | 50 |
| Target column | `Class` |
| Number of classes | 4 |
| Classes | angry, happy, relax, sad |
| Initial samples per class | 100 |
| Missing values | 0 |
| Duplicate rows | 12 |
| Rows after duplicate removal | 388 |
| Rows after IQR-based capping | 388 |
| Total capped feature values | 426 |

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── Acoustic Features.csv
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── text/
├── reports/
├── scripts/
│   ├── 01_data_quality.py
│   ├── 02_correlation_analysis.py
│   ├── 03_feature_discriminability.py
│   ├── 04_pca_analysis.py
│   ├── 05_lda_analysis.py
│   ├── 06_clustering_analysis.py
│   ├── 07_hypothesis_testing.py
│   ├── 08_manual_calculation_support.py
│   ├── 09_sensitivity_analysis.py
│   ├── 10_report_quality_figures.py
│   ├── 11_clustering_sensitivity.py
│   ├── 12_statistical_testing_enhanced.py
│   ├── 13_mathematical_support_tables.py
│   ├── 14_tsne_metric_correction.py
│   ├── 99_run_all.py
│   └── 99_run_stage2.py
├── src/
│   ├── common.py
│   └── advanced_common.py
├── README.md
├── requirements.txt
└── requirements_stage2.txt
```

---

## Installation

Create a virtual environment:

```bash
python -m venv myenv
```

Windows PowerShell:

```powershell
myenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_stage2.txt
```

---

## Running the Analysis

Run the first-stage analysis:

```bash
python scripts/99_run_all.py --target Class
```

Run the second-stage analysis:

```bash
python scripts/99_run_stage2.py --target Class
```

Run the corrected t-SNE analysis:

```bash
python scripts/14_tsne_metric_correction.py --target Class
```

If you want to explicitly provide the CSV path:

```bash
python scripts/99_run_all.py --csv "data/raw/Acoustic Features.csv" --target Class
python scripts/99_run_stage2.py --csv "data/raw/Acoustic Features.csv" --target Class
python scripts/14_tsne_metric_correction.py --csv "data/raw/Acoustic Features.csv" --target Class
```

---

## Methodological Pipeline

```text
Raw Dataset
   ↓
Data Quality Assessment
   ↓
Duplicate Removal + IQR-Based Capping
   ↓
Z-score Normalization
   ↓
Correlation Analysis + Fisher Score Analysis
   ↓
PCA and LDA Projection Analysis
   ↓
K-Means, DBSCAN, t-SNE, SOM
   ↓
Hypothesis Testing + Stability Analysis
```

---

## Mathematical Foundation

### Z-score Normalization

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

### IQR-Based Capping

$$
IQR = Q_3 - Q_1
$$

$$
L = Q_1 - 1.5 \times IQR
$$

$$
U = Q_3 + 1.5 \times IQR
$$

### Multi-Class Fisher Score

$$
J_j =
\frac{
\sum_{c=1}^{C} n_c(\mu_{cj}-\mu_j)^2
}{
\sum_{c=1}^{C}\sum_{i:y_i=c}(x_{ij}-\mu_{cj})^2
}
$$

### PCA Eigenvalue Problem

$$
S v_k = \lambda_k v_k
$$

### LDA Objective

$$
J(w) = \frac{w^T S_B w}{w^T S_W w}
$$

### K-Means Objective

$$
J = \sum_{k=1}^{K}\sum_{i \in C_k} \|x_i - \mu_k\|^2
$$

### Welch t-test

$$
t =
\frac{\bar{x}_1-\bar{x}_2}
{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}
$$

---

## Key Findings

- The dataset is balanced and appropriate for multivariate statistical analysis.
- Duplicate removal and IQR-based capping improve data quality without damaging class structure.
- The most discriminative feature is **`_HarmonicChangeDetectionFunction_Std`**.
- Emotion separability is mainly associated with **harmonic change**, **zero-crossing behavior**, **rhythmic pulse**, **event density**, and **spectral structure**.
- **PCA** reveals dominant variance directions but limited class separability in 2D.
- **LDA** clearly outperforms PCA for supervised class separation.
- **K-Means** partially recovers emotion-related structure.
- **DBSCAN** does not align well with the true class labels in PCA 2D.
- **t-SNE + K-Means** provides better local grouping than PCA-based clustering.
- The hypothesis test confirms a highly significant difference between **angry** and **relax** for `_HarmonicChangeDetectionFunction_Std`.

---

## Selected Visual Results

### Top Discriminative Features

<img src="outputs/figures/03_top_fisher_features_barh.png" width="850">

**Interpretation:** The strongest class separation is driven by harmonic change, zero-crossing rate, pulse clarity, event density, and spectral descriptors.

---

### Feature Correlation Heatmap

<img src="outputs/figures/02_feature_correlation_heatmap.png" width="850">

**Interpretation:** Strong correlations are visible among spectral descriptors, indicating that several variables describe related aspects of the frequency-domain structure.

---

### PCA Cumulative Explained Variance

<img src="outputs/figures/04_pca_cumulative_explained_variance.png" width="850">

**Interpretation:** The first two principal components explain only a limited portion of total variance, which is one reason why PCA 2D does not perfectly separate classes.

---

### PCA Projection: First Two Components

<img src="outputs/figures/04_pca_first_two_components.png" width="850">

**Interpretation:** Partial grouping exists, but substantial overlap remains among emotion classes.

---

### PCA Projection: Last Two Components

<img src="outputs/figures/04_pca_last_two_components.png" width="850">

**Interpretation:** The last components carry very little discriminative information, so class overlap becomes stronger.

---

### LDA Projection

> Insert your LDA projection figure here if available.
>
> Recommended file names in your repo may be one of the following:
>
> - `outputs/figures/05_lda_2d_separability.png`
> - `outputs/figures/04_lda_2d_projection_scatter.png`

**Interpretation:** LDA provides much clearer class separation than PCA because it explicitly uses class labels while constructing projection axes.

---

### t-SNE Projection by True Emotion Class

<img src="outputs/figures/14_tsne_perplexity_30_true_classes.png" width="850">

**Interpretation:** t-SNE reveals local neighborhood structure more clearly than PCA, although classes still overlap in some regions.

---

### K-Means on t-SNE Embedding

<img src="outputs/figures/14_tsne_kmeans_perplexity_30.png" width="850">

**Interpretation:** K-Means on the t-SNE embedding shows better local grouping than PCA-based clustering, but it still does not achieve perfect agreement with true labels.

---

## Core Results Summary

### Top Features by Fisher Score

| Rank | Feature | Fisher Score |
|---:|---|---:|
| 1 | `_HarmonicChangeDetectionFunction_Std` | 1.328385 |
| 2 | `_Zero-crossingrate_Mean` | 0.985252 |
| 3 | `_HarmonicChangeDetectionFunction_PeriodAmp` | 0.929150 |
| 4 | `_Pulseclarity_Mean` | 0.693210 |
| 5 | `_Eventdensity_Mean` | 0.641541 |

### PCA vs LDA Separability

| Embedding | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|
| PCA 2D | 0.037892 | 3.754672 | 71.968649 |
| LDA 2D | 0.306186 | 1.215227 | 346.499539 |

### t-SNE + K-Means Metrics

| Perplexity | ARI | NMI | Cluster Silhouette |
|---:|---:|---:|---:|
| 5 | 0.148831 | 0.188742 | 0.404948 |
| 15 | 0.207996 | 0.236232 | 0.374948 |
| 30 | 0.290140 | 0.312316 | 0.397721 |
| 50 | 0.272602 | 0.313342 | 0.401833 |

### Hypothesis Testing Result

| Feature | Classes | Sample Size | Welch t | p-value | Cohen's d | Decision |
|---|---|---:|---:|---:|---:|---|
| `_HarmonicChangeDetectionFunction_Std` | angry vs relax | 36 | -12.223350 | 5.44e-19 | -2.881071 | Reject H0 |
| `_HarmonicChangeDetectionFunction_Std` | angry vs relax | 64 | -15.363372 | 1.16e-30 | -2.715886 | Reject H0 |

---

## Output Locations

### Figures

```text
outputs/figures/
```

### Tables

```text
outputs/tables/
```

### Text Summaries

```text
outputs/text/
```

---

## References

- Turkish Music Emotion Dataset, Kaggle.
- Scikit-learn documentation for StandardScaler, PCA, LDA, K-Means, DBSCAN, and t-SNE.
- MiniSom documentation / repository.
- Standard references on Fisher discriminant analysis and statistical hypothesis testing.

---

## Notes

This README was intentionally revised to reference **only confirmed or clearly marked assets**. If you want, the next step can be one of these:

1. I can prepare a **final polished GitHub-ready README** with a stronger visual style.
2. I can write a small script to generate a **methodological pipeline diagram** automatically.
3. I can also prepare a **project poster / project summary section** for the top of the README.
