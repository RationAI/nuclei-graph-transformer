## Preprocessing Workflow

### MMCI Tile Level Annotations Data

1. **Nuclei Segmentation** (`nuclei_segmentation.py`)  
   Segments nuclei in whole-slide images and stores the outputs as partitioned Parquet tables.

2. **Annotation Masks** (`annotation_masks.py`)  
   Generates binary masks for annotated carcinoma regions using XML annotation files by expert pathologists. 

3. **Unipolar Heatmap-based Nuclei Labeling** (`unipolar_heatmap_labels.py`)  
   Assigns labels to segmented nuclei by checking polygon overlap with the provided (thresholded) unipolar heatmap.

4. **CAM Masks Preparation** (`merge_cam_masks.py`)  
   Aggregates generated CAM masks from multiple MLflow runs into a single location for convenience.

5. **CAM-based Nuclei Labeling** (`cam_labels.py`)  
   Computes CAM pseudo labels by thresholding positive/negative regions and storing the average CAM intensity for each nucleus (for loss weighting, etc.).

6. **Map Slides to Nuclei** (`metadata_mapping/prostate_cancer_mmci_tl.py`)  
   Creates a mapping of slides' metadata necessary for downstream modeling.

### PANDA Challenge Dataset

1. **Nuclei Data Standardization** (`nuclei_standardization.py`)  
   Standardizes nuclei segmentation files provided by a different project to match the expected structure.

2. **Annotation-based Nuclei Labeling** (`annotation_labels.py`)  
   Assigns labels to segmented nuclei by checking polygon overlap with label masks.

3. **Train-Test Split** (`data_split.py`)  
   Performs train-test split stratified by gleason scores.

4. **Map Slides to Nuclei** (`metadata_mapping/panda.py`)  
   Creates a mapping of slides' metadata necessary for downstream modeling.

## Output Structure Overview

### Nuclei Segmentation: `nuclei_segmentation.py`

**Location**: Disk

**Output layout**
```text
<DATASET_NAME>/
   slide_id=<SLIDE_NAME>/
      *.parquet (segmented nuclei)
```
**Parquet row schema (one row = one nucleus)**
- `id` (`str`): Unique nucleus hash ID.
- `polygon` (`np.ndarray[float]`): Flattened polygon coordinates (64 points × 2 coordinates).
- `centroid` (`np.ndarray[float]`): Nucleus centroid `(x, y)`. 

### Annotation Masks: `annotation_masks.py`

**Location**: MLflow artifacts

**Output layout**
```text
annotation_masks/
  <SLIDE_NAME>.tiff (single-channel binary mask for carcinoma regions)
missing_annotations.csv (slide paths of positive slides without annotation files)
```

### Unipolar Heatmap-based Nuclei Labels: `unipolar_heatmap_labels.py`

**Location**: MLflow artifacts

**Output layout**
```text
<MLFLOW_ARTIFACT_PATH>/
  <SLIDE_NAME>.parquet
```

**Parquet row schema (one row = one nucleus)**
- `slide_id` (`str`)
- `id` (`str`): Nucleus identifier.
- `<LABEL_COLUMN>` (`int`): Binary label produced from overlap with thresholded mask.


### CAM Masks Preparation: `merge_cam_masks.py`

**Location**: MLflow artifacts

**Output layout**
```text
cam_masks/
  <SLIDE_NAME>.tiff (bipolar heatmap of CAM intensities in [0, 255])
missing_cam_masks.csv (slide paths of positive slides without a CAM mask)
```

### CAM-based Nuclei Labels: `cam_labels.py`

**Location**: MLflow artifacts

**Output layout**
```text
cam_labels/
  <SLIDE_NAME>.parquet 
```

**Parquet row schema (one row = one nucleus)**
- `slide_id` (`str`)
- `id` (`str`)
- `cam_label` (`int`):
  - `1` = positive CAM region overlap above positive threshold,
  - `0` = negative CAM region overlap below negative threshold,
  - `-1` = uncertain.
- `cam_score` (`float`): Mean CAM intensity sampled over nucleus polygon vertices and centroid.

### Metadata Mapping: `metadata_mapping.py`

**Location**: MLflow artifacts

**Output layout**
```text
<DATASET_NAME>/
  slides_mapping.parquet
```

**Parquet row schema (one row = one slide)**
- `slide_id` (`str`)
- `patient_id` (`str`)
- `slide_path` (`str`)
- `slide_nuclei_path` (`str`): Path to partitioned nuclei parquet slide folder.
- `nuclei_count` (`int`)
- `is_carcinoma` (`bool`)
- `mpp_x` (`float`)
- `mpp_y` (`float`)

### Nuclei Standardization: `nuclei_standardization.py`

**Location**: Disk

**Output layout**
```text
<DATASET_NAME>/
   slide_id=<SLIDE_NAME>/ (renamed folder, matches the original slide name)
      *.parquet (segmented nuclei)
```

**Parquet row schema**
- columns of the input dataset 
- `id` (`str`): Newly generated unique nucleus hash ID.

### Train-Test Split: `data_split.py`

**Location**: MLflow artifacts

**Output layout**
```text
<DATASET_NAME>_split/
  split.csv ()
  summary.csv (table with aggregate statistics)
  total_counts.csv (table with slide counts for each set)
```

**Parquet row schema of `split.csv`**
- `slide_id` (`str`)
- `set` (`str`): "train" or "test

### Annotation-based Nuclei Labels: `annotation_labels.py`

**Location**: MLflow artifacts

**Output layout**
```text
annotation_labels/
  <SLIDE_NAME>.parquet 
```

**Parquet row schema (one row = one nucleus)**
- `slide_id` (`str`)
- `id` (`str`): Nucleus identifier.
- `annot_label` (`int`): Binary label produced from overlap with thresholded label mask.

