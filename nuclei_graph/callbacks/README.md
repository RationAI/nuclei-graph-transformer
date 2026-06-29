<a id="callbacks-overview"></a>
## Callbacks Overview

Most callbacks accept an optional `mlflow_run_id` constructor argument: when set, results are logged to that specific run; when left unset, they fall back to whatever run is currently active.

1. **Predictions** (`predictions.py`, [output structure](#predictions-output))  
   Saves per-nucleus or per-tile model predictions (and, for crops, attention scores and graph-level predictions) as Parquet files.

2. **Masks** (`masks.py`, [output structure](#masks-output))  
   Renders prediction or attention scores as polygon or tile-grid heatmaps on whole-slide masks.

3. **ROC/PR Curves** (`plot_curves.py`, [output structure](#curves-output))  
   Computes and logs ROC and Precision-Recall curves, plus optimal-threshold metrics, for graph- and/or nuclei-level validation predictions.

4. **Permutation Feature Importance** (`feature_importances.py`, [output structure](#feature-importances-output))  
   Measures the AUROC drop from shuffling each EFD feature group on the test set.

5. **Spatial Permutation Importance** (`spatial_permutation_importance.py`, [output structure](#spatial-permutation-output))  
   Measures the metric drop from shuffling nuclei spatial positions on the test set.

6. **Parent Run Tagging** (`parent_run.py`)  
   Tags the active MLflow run with `mlflow.parentRunId`, nesting it under an existing run (e.g. the training run a checkpoint came from) in the MLflow UI. 

## Output Structure Overview

<a id="predictions-output"></a>
### Predictions: `predictions.py`

**Location**: MLflow artifacts

**Output layout**:
```text
<mlflow_artifact_path>/   (default: "predictions")
  <SLIDE_NAME>.parquet
```

**Parquet row schema**:
- `NucleiPredictionCallback` (one row = one nucleus):
  - `id` (`str`)
  - `nuclei_prediction` (`float`)
- `CropPredictionCallback` (one row = one nucleus):
  - `id` (`str`)
  - `nuclei_prediction` (`float`)
  - `attention_score` (`float`)
  - `graph_prediction` (`float`): repeated for every row of the slide.
- `TilePredictionCallback` (one row = one tile):
  - `x` (`int`), `y` (`int`)
  - `tile_prediction` (`float`): graph-level logit (`TilePredictionCallback`).

<p align="right"><a href="#callbacks-overview">↑ back</a></p>

---

<a id="masks-output"></a>
### Masks: `masks.py`

**Location**: MLflow artifacts

**Output layout**:
```text
<mlflow_artifact_path>/
  <SLIDE_NAME>.tiff
```

- `TileHeatmapMasksCallback` (default path: `tile_heatmaps`): per-tile graph-level prediction heatmap.
- `NucleiPredictionMasksCallback` (default path: `prediction_masks`): per-nucleus prediction score painted onto its polygon.
- `AttentionMasksCallback` (default path: `attention_masks`): per-nucleus attention score (normalized to its slide's max) painted onto its polygon.

<p align="right"><a href="#callbacks-overview">↑ back</a></p>

---

<a id="curves-output"></a>
### ROC/PR Curves: `plot_curves.py`

**Location**: MLflow artifacts and metrics

**Output layout**:
```text
curves/
  <level_name>_roc.png
  <level_name>_precision_recall.png
```
where `<level_name>` is `val_graph` and/or `val_nuclei` (`CropCurvesCallback` logs both; `NucleiCurvesCallback` logs nuclei only).

**Metrics logged**:
- `thresholds/<level_name>_tpr`: threshold achieving TPR ≈ 1.0 with the lowest FPR.
- `thresholds/<level_name>_j`: threshold maximizing Youden's J statistic (TPR − FPR).
- `thresholds/<level_name>_f1`: threshold maximizing F1 score.

<p align="right"><a href="#callbacks-overview">↑ back</a></p>

---

<a id="feature-importances-output"></a>
### Permutation Feature Importance: `feature_importances.py`

**Location**: MLflow artifacts

**Output layout**:
```text
feature_importances.png
```
Bar chart of AUROC decrease per shuffled EFD feature group (EFD orders, scale, rotation).

<p align="right"><a href="#callbacks-overview">↑ back</a></p>

---

<a id="spatial-permutation-output"></a>
### Spatial Permutation Importance: `spatial_permutation_importance.py`

**Location**: MLflow artifacts

**Output layout**:
```text
spatial_permutation/
  metrics.png (bar chart of metric decrease after permuting nuclei positions)
  pr_curve.png (baseline vs. permuted Precision-Recall curve)
```

<p align="right"><a href="#callbacks-overview">↑ back</a></p>
