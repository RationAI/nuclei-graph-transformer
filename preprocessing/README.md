## Preprocessing Workflow

<a id="mmci-workflow"></a>
### MMCI Tile Level Annotations Data
<a id="mmci-workflow"></a>
### MMCI Tile Level Annotations Data

1. **Nuclei Segmentation** (`nuclei_segmentation.py`, [output structure](#nuclei-segmentation-output))  
1. **Nuclei Segmentation** (`nuclei_segmentation.py`, [output structure](#nuclei-segmentation-output))  
   Segments nuclei in whole-slide images and stores the outputs as partitioned Parquet tables.

2. **Annotation Masks** (`annotation_masks/prostate_cancer_mmci_tl.py`, [output structure](#annotation-masks-output))  
   Generates binary masks for annotated carcinoma regions using XML annotation files by expert pathologists. 

3. **CAM Masks Preparation** (`merge_cam_masks.py`, [output structure](#cam-masks-output))   
3. **CAM Masks Preparation** (`merge_cam_masks.py`, [output structure](#cam-masks-output))   
   Aggregates generated CAM masks from multiple MLflow runs into a single location for convenience.

4. **Map Slides to Nuclei** (`metadata_mapping/prostate_cancer_mmci_tl.py`, [output structure](#metadata-mapping-mmci-output))   
   Creates a mapping of slides' metadata necessary for downstream modeling.

5. **Unipolar Heatmap-based Nuclei Labeling** (`unipolar_heatmap_labels.py`, [output structure](#unipolar-labels-output))  
   Assigns labels to segmented nuclei by checking polygon overlap with the provided (thresholded) unipolar heatmap.

6. **CAM-based Nuclei Labeling** (`cam_labels.py`, [output structure](#cam-labels-output))  
   Computes CAM pseudo labels by thresholding positive/negative regions and storing the average CAM intensity for each nucleus.

<a id="panda-workflow"></a>
### PANDA Challenge Dataset

1. **Nuclei Data Standardization** (`nuclei_standardization.py`, [output structure](#nuclei-standardization-output))  
   Standardizes nuclei segmentation files provided by a different project to match the expected structure.

2. **Train-Test Split** (`data_split.py`, [output structure](#data-split-output))  
   Performs train-test split stratified by gleason scores.
4. **Map Slides to Nuclei** (`metadata_mapping/prostate_cancer_mmci_tl.py`, [output structure](#metadata-mapping-mmci-output))   
   Creates a mapping of slides' metadata necessary for downstream modeling.

5. **Unipolar Heatmap-based Nuclei Labeling** (`unipolar_heatmap_labels.py`, [output structure](#unipolar-labels-output))  
   Assigns labels to segmented nuclei by checking polygon overlap with the provided (thresholded) unipolar heatmap.

6. **CAM-based Nuclei Labeling** (`cam_labels.py`, [output structure](#cam-labels-output))  
   Computes CAM pseudo labels by thresholding positive/negative regions and storing the average CAM intensity for each nucleus.

<a id="panda-workflow"></a>
### PANDA Challenge Dataset

1. **Nuclei Data Standardization** (`nuclei_standardization.py`, [output structure](#nuclei-standardization-output))  
   Standardizes nuclei segmentation files provided by a different project to match the expected structure.

2. **Train-Test Split** (`data_split.py`, [output structure](#data-split-output))  
   Performs train-test split stratified by gleason scores.

3. **Map Slides to Nuclei** (`metadata_mapping/panda.py`, [output structure](#metadata-mapping-panda-output))  
3. **Map Slides to Nuclei** (`metadata_mapping/panda.py`, [output structure](#metadata-mapping-panda-output))  
   Creates a mapping of slides' metadata necessary for downstream modeling.

<a id="icaird-cervix-workflow"></a>
### iCAIRD Cervix

1. **iSyntax to .TIF Conversion** (`isyntax2tif.py`)  
   Converts iSyntax slides to pyramidal OpenSlide-compatible TIFF.

2. **Annotation Masks** (`annotation_masks/icaird_cervix.py`, [output structure](#icaird-annotation-masks-output))  
   Generates per-slide masks from QuPath GeoJSON annotations, encoding the most severe classification (low grade / high grade / malignant) covering each pixel.

3. **ROI Sampling** (`roi_sampling/icaird_cervix.py`, [output structure](#icaird-roi-sampling-output))  
   Randomly samples rectangular ROIs inside the positive (high grade / malignant) annotations of the test split, for precise re-annotation by a pathologist.

## Output Structure Overview

<a id="nuclei-segmentation-output"></a>
### Nuclei Segmentation: `nuclei_segmentation.py`

**Location**: Disk

**Output layout**:
```text
<DATASET_NAME>/
   <BATCH_NAME>/
      slide_id=<SLIDE_NAME>/
         *.parquet (segmented nuclei)
```
`<BATCH_NAME>` is the `batch_name` config value, or by default the folder holding the metadata CSV (a CSV at the root of a run's artifacts has to set `batch_name`).
**Parquet row schema (one row = one nucleus)**:
- `id` (`str`): Unique nucleus hash ID.
- `polygon` (`np.ndarray[float]`): Flattened polygon coordinates (64 points × 2 coordinates).
- `centroid` (`np.ndarray[float]`): Nucleus centroid `(x, y)`. 
<p align="right"><a href="#mmci-workflow">↑ back</a></p>

---

<a id="annotation-masks-output"></a>
### Annotation Masks: `annotation_masks/prostate_cancer_mmci_tl.py`

**Location**: MLflow artifacts

**Output layout**:
```text
annotation_masks/
  <SLIDE_NAME>.tiff (single-channel binary mask for carcinoma regions)
missing_annotations.csv (slide paths of positive slides without annotation files)
```
<p align="right"><a href="#mmci-workflow">↑ back</a></p>

---

<a id="cam-masks-output"></a>
### CAM Masks Preparation: `merge_cam_masks.py`

**Location**: MLflow artifacts

**Output layout**:
```text
cam_masks/
  <SLIDE_NAME>.tiff (bipolar heatmap of CAM intensities in [0, 255])
missing_cam_masks.csv (slide paths of positive slides without a CAM mask)
```
<p align="right"><a href="#mmci-workflow">↑ back</a></p>

---

<a id="icaird-annotation-masks-output"></a>
### Annotation Masks: `annotation_masks/icaird_cervix.py`

**Location**: MLflow artifacts

**Output layout**:
```text
annotation_masks/
  <SLIDE_NAME>.tiff (single-channel mask, one file per slide with lesion annotations)
```

**Mask pixel values**:
- `0`: background / unannotated
- `1`: low grade (CIN1, HPV)
- `2`: high grade (CIN2, CIN3)
- `3`: malignant (squamous carcinoma, adenocarcinoma, ...)

Slides in the "normal_inflammation" category (see the dataset's `index.csv`) have no lesion annotations and are skipped.
<p align="right"><a href="#icaird-cervix-workflow">↑ back</a></p>

---

<a id="icaird-roi-sampling-output"></a>
### ROI Sampling: `roi_sampling/icaird_cervix.py`

**Location**: MLflow artifacts

**Output layout**:
```text
roi_masks/
  <SLIDE_NAME>.tiff (binary mask, one file per slide with at least one ROI)
rois.csv (one row per ROI)
slides_summary.csv (one row per slide)
```

**Mask pixel values** (level `level`, 1 µm/px by default):
- `0`: background
- `255`: ROI

**Sampling** (parameters under `roi` in `configs/preprocessing/roi_sampling/icaird_cervix.yaml`):
- The positive region of a slide is the high grade ∪ malignant annotation **restricted to tissue** (the tissue mask logged by `tissue_masks/icaird_cervix.py`, `tissue_uri`).
- Per slide, rectangular ROIs are drawn until their part inside the positive tissue reaches `target_fraction` (10 %) of the slide's positive tissue area; the last ROI is kept only if it brings the total closer to the target.
- ROI area is uniform in `min_area_mm2`–`max_area_mm2` (0.25–1 mm²), the aspect ratio is log-uniform in 1–`max_aspect_ratio` with a random orientation.
- At least `coverage_steps[0]` (80 %) of each ROI has to lie inside the positive tissue. A slide for which no such ROI exists (small or fragmented lesions) is not left out: the requirement is relaxed to the next value of `coverage_steps` (60 %, 40 %, 20 %), and if even the last fails, the slide gets a single minimum-sized ROI where the coverage is the highest (`best_effort`).
- If 10 % of a slide's positive area is smaller than a minimum-sized ROI, the slide still gets one minimum-sized ROI.
- ROIs do not overlap and are `min_gap_um` apart (so they stay separate objects in the binary mask).
- Sampling is reproducible: the RNG is seeded from `seed` and the slide name. Coverage is evaluated on a raster at `tissue_level` (level 3), i.e. to ~2 µm.

**`rois.csv` columns**: `slide_id`, `roi_id`, `x0`, `y0`, `x1`, `y1` (mask pixel coordinates, `x1`/`y1` exclusive), `area_mm2`, `coverage` (fraction of the ROI inside the positive annotation on tissue), `tissue_fraction` (fraction of the ROI on tissue), `coverage_high_grade`, `coverage_malignant` (fractions inside each annotation class, regardless of tissue).

**`slides_summary.csv` columns**: `slide_id`, `category`, `subcategory`, `status` (`ok`, `relaxed_coverage`, `best_effort`, `no_positive_annotation`, `no_feasible_roi`), `min_coverage_used` (the `coverage_steps` value the ROIs were sampled at; empty for `best_effort`), `tissue_mask` (`used`, or `missing` / `ignored` if the tissue mask could not be used), `positive_area_mm2`, `positive_tissue_area_mm2`, `n_rois`, `roi_area_mm2`, `sampled_positive_area_mm2`, `sampled_fraction` (of the positive tissue area).
<p align="right"><a href="#icaird-cervix-workflow">↑ back</a></p>

---

<a id="metadata-mapping-mmci-output"></a>
### Metadata Mapping: `metadata_mapping/prostate_cancer_mmci_tl.py`

**Location**: MLflow artifacts

**Output layout**:
```text
<DATASET_NAME>/
  slides_mapping.parquet
```

**Parquet row schema (one row = one slide)**:
- `slide_id` (`str`)
- `patient_id` (`str`): 4-digit unique patient identifier.
- `slide_path` (`str`)
- `slide_nuclei_path` (`str`): Path to partitioned nuclei parquet slide folder.
- `nuclei_count` (`int`)
- `is_carcinoma` (`bool`)
- `mpp_x` (`float`)
- `mpp_y` (`float`)
<p align="right"><a href="#mmci-workflow">↑ back</a></p>

---

<a id="unipolar-labels-output"></a>
### Unipolar Heatmap-based Nuclei Labels: `unipolar_heatmap_labels.py`

**Location**: Disk

**Output layout**:
```text
<OUTPUT_PATH>/
  <SLIDE_NAME>.parquet
```

**Parquet row schema (one row = one nucleus)**:
- `slide_id` (`str`)
- `id` (`str`): Nucleus identifier.
- `<LABEL_COLUMN>` (`int`): Binary label produced from overlap with thresholded mask.
<p align="right"><a href="#mmci-workflow">↑ back</a></p>

---

<a id="cam-labels-output"></a>
### CAM-based Nuclei Labels: `cam_labels.py`

**Location**: Disk

**Output layout**:
```text
<OUTPUT_PATH>/
  <SLIDE_NAME>.parquet 
```

**Parquet row schema (one row = one nucleus)**:
- `slide_id` (`str`)
- `id` (`str`)
- `cam_label` (`int`):
  - `1` = positive CAM region overlap above positive threshold,
  - `0` = negative CAM region overlap below negative threshold,
  - `-1` = uncertain.
- `cam_score` (`float`): Mean CAM intensity sampled over nucleus polygon vertices and centroid.

<p align="right"><a href="#mmci-workflow">↑ back</a></p>

---

<a id="nuclei-standardization-output"></a>
### Nuclei Standardization: `nuclei_standardization.py`

**Location**: Disk

**Output layout**:
```text
<DATASET_NAME>/
   slide_id=<SLIDE_NAME>/ (renamed folder, matches the original slide name)
      *.parquet (segmented nuclei)
```

**Parquet row schema**:
- columns of the input dataset 
- `id` (`str`): Newly generated unique nucleus hash ID.
<p align="right"><a href="#panda-workflow">↑ back</a></p>

---

<a id="data-split-output"></a>
### Train-Test Split: `data_split.py`

**Location**: MLflow artifacts

**Output layout**:
```text
<DATASET_NAME>_split/
  split.csv
  summary.csv (table with aggregate statistics)
  total_counts.csv (table with slide counts for each set)
```

**CSV row schema of `split.csv`**:
- `slide_id` (`str`)
- `set` (`str`): "train" or "test"
<p align="right"><a href="#panda-workflow">↑ back</a></p>

---

<a id="metadata-mapping-panda-output"></a>
### Metadata Mapping: `metadata_mapping/panda.py`

**Location**: MLflow artifacts

**Output layout**:
```text
panda/
  slides_mapping_test.parquet
  slides_mapping_train.parquet
```

**Parquet row schema (one row = one slide)**:
- `slide_id` (`str`)
- `slide_path` (`str`)
- `slide_nuclei_path` (`str`): Path to partitioned nuclei parquet slide folder.
- `nuclei_count` (`int`)
- `is_carcinoma` (`bool`): True if ISUP grade is > 0.
- `mpp_x` (`float`)
- `mpp_y` (`float`)
<p align="right"><a href="#panda-workflow">↑ back</a></p>

---