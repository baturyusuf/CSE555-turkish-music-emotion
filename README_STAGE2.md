# Turkish Music Emotion Analysis — Stage 2 Scripts

Bu paket mevcut proje klasörünün üzerine kopyalanmak üzere hazırlanmıştır. Amaç; rapor ve sunumda kullanılabilecek profesyonel tablo, grafik ve metin çıktıları üretmektir.

## Kopyalama

Bu paketteki `scripts/` ve `src/` klasörlerini mevcut repo içine kopyalayın.

Beklenen ana yapı:

```text
project-root/
├── data/
│   └── raw/
│       └── Acoustic Features.csv
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── text/
├── scripts/
└── src/
```

## Kurulum

```bash
pip install -r requirements_stage2.txt
```

`minisom` SOM analizi için gereklidir.

## Tüm ikinci aşama analizleri çalıştırma

```bash
python scripts/99_run_stage2.py --target Class
```

CSV yolunu açık vermek için:

```bash
python scripts/99_run_stage2.py --csv "data/raw/Acoustic Features.csv" --target Class
```

## Scriptler

### `09_sensitivity_analysis.py`

Veri temizleme kararlarının sonuçları ne kadar etkilediğini ölçer.

Üretilen ana çıktılar:

```text
outputs/tables/09_dataset_variant_overview.csv
outputs/tables/09_fisher_scores_by_dataset_variant.csv
outputs/tables/09_pca_summary_by_dataset_variant.csv
outputs/tables/09_pca_lda_separability_by_dataset_variant.csv
outputs/figures/09_fisher_score_stability_across_variants.png
outputs/figures/09_pca_cumulative_variance_variant_comparison.png
outputs/figures/09_pca_lda_silhouette_variant_comparison.png
outputs/text/09_sensitivity_analysis_summary.txt
```

### `10_report_quality_figures.py`

Rapor ve sunumda kullanılabilecek okunabilir figürleri üretir.

Üretilen ana çıktılar:

```text
outputs/figures/10_classwise_boxplots_top_fisher_features.png
outputs/figures/10_classwise_boxplots_top_outlier_features.png
outputs/figures/10_pca_eigenvalue_vs_projected_fisher_score.png
outputs/figures/10_pca_loading_heatmap_first_five_components.png
outputs/figures/10_pca_vs_lda_side_by_side.png
outputs/figures/10_focused_discriminative_feature_correlation_heatmap.png
outputs/figures/10_top_20_fisher_score_bar_chart.png
```

### `11_clustering_sensitivity.py`

K-Means, DBSCAN, t-SNE ve SOM analizlerini daha güçlü biçimde üretir.

Üretilen ana çıktılar:

```text
outputs/tables/11_kmeans_sensitivity.csv
outputs/tables/11_dbscan_eps_sensitivity.csv
outputs/tables/11_tsne_perplexity_separability_metrics.csv
outputs/tables/11_som_winning_neurons.csv
outputs/tables/11_som_neuron_purity.csv
outputs/figures/11_kmeans_elbow_curve.png
outputs/figures/11_kmeans_metric_sensitivity.png
outputs/figures/11_dbscan_k_distance_curve.png
outputs/figures/11_dbscan_eps_sensitivity_counts.png
outputs/figures/11_dbscan_eps_sensitivity_validity.png
outputs/figures/11_tsne_perplexity_5.png
outputs/figures/11_tsne_perplexity_15.png
outputs/figures/11_tsne_perplexity_30.png
outputs/figures/11_tsne_perplexity_50.png
outputs/figures/11_som_u_matrix.png
outputs/figures/11_som_class_hit_maps.png
```

### `12_statistical_testing_enhanced.py`

36 ve 64 örnek için hipotez testini varsayım testleri, manuel Welch bileşenleri, confidence interval ve repeated sampling stability ile güçlendirir.

Üretilen ana çıktılar:

```text
outputs/tables/12_ttest_assumption_and_result_table.csv
outputs/tables/12_candidate_feature_class_pairs_for_ttest.csv
outputs/tables/12_repeated_sampling_ttest_results.csv
outputs/tables/12_repeated_sampling_stability_summary.csv
outputs/figures/12_ttest_sample_distribution_n36.png
outputs/figures/12_ttest_sample_distribution_n64.png
outputs/figures/12_ttest_qq_plots_n36.png
outputs/figures/12_ttest_qq_plots_n64.png
outputs/figures/12_repeated_sampling_pvalue_stability.png
outputs/figures/12_repeated_sampling_effect_size_stability.png
outputs/text/12_statistical_testing_equations.txt
outputs/text/12_enhanced_hypothesis_testing_summary.txt
```

### `13_mathematical_support_tables.py`

Raporda elle gösterilebilecek matematiksel ara çıktıları üretir.

Üretilen ana çıktılar:

```text
outputs/tables/13_fisher_manual_contribution_selected_feature.csv
outputs/tables/13_pairwise_fisher_selected_feature.csv
outputs/tables/13_covariance_matrix_top_discriminative_features.csv
outputs/tables/13_pca_eigenvalue_fisher_rank_table.csv
outputs/tables/13_pca_top_loadings_for_manual_interpretation.csv
outputs/tables/13_lda_class_centroids.csv
outputs/tables/13_lda_explained_discriminant_variance.csv
outputs/tables/13_lda_top_coefficients.csv
outputs/text/13_mathematical_support_summary.txt
```

## Not

Bu paket rapor veya sunum dosyası oluşturmaz. Çalıştırıldığında rapor/sunum yazımında kullanılacak tablo, grafik ve metin çıktıları üretir.


## Import Path Note

The scripts add the project root to `sys.path` automatically. `99_run_stage2.py` also sets `PYTHONPATH` for subprocess calls, so it can be run directly from the project root on Windows PowerShell.
