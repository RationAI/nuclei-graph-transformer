# Data Exploration

## Output Structure Overview

### MMCI Tile-Level Annotations Data

**Location**: MLflow artifacts

**Output layout**:
```text
<DATASET_NAME>/
  slides_metadata.csv
  summary.csv (aggregate statistics)
```

**CSV metadata row schema (one row = one slide)**:
- `slide_path` (`str`)
- `is_carcinoma` (`bool`)
- `has_annotation` (`bool`): True if the annotation mask exists.
- `patient_id` (`str`): 4-digit unique patient identifier.
- `case_id` (`str`): A combination of year and patient_id.

### PANDA Challenge Dataset

**Location**: MLflow artifacts

**Output layout**:  
```text
panda/
  errors.log (invalid slides or label masks — empty, corrupted encoding...)
  slides_metadata.csv (metadata for the valid slides)
  summary.csv (aggregate statistics)
```

**CSV metadata row schema (one row = one slide)**:  
- `slide_id` (`str`): 32-character hex string identifier for each slide.
- `slide_path` (`str`)
- `segmentation_id` (`str`): Identifier of the slide in the parquet dataset with segmented nuclei.
- `data_provider` (`str`): "radboud" or "karolinska".
- `isup_grade` (`int`)
- `gleason_score` (`str`)
- `has_segmentation` (`bool`): True if the segmentation file exists.
- `has_annotation` (`bool`): True if the annotation mask exists.
- `extent_x` (`float`)
- `extent_y` (`float`)
- `mpp_x` (`float`)
- `mpp_y` (`float`)

### BEETLE Dataset

**Location**: MLflow artifacts

**Output layout**:
```text
beetle/
  errors.log (slides listed in data_overview.csv but missing on disk)
  slides_metadata.csv (metadata for the valid slides)
  summary.csv (aggregate statistics)
```

**CSV metadata row schema (one row = one slide)**:
- `slide_id` (`str`): matches the `name` column of the dataset's `data_overview.csv`.
- `slide_path` (`str`)
- `mask_path` (`str`): empty if no annotation mask exists (e.g. held-out evaluation slides).
- `has_annotation` (`bool`): True if the annotation mask exists.
- `has_annotation_xml` (`bool`): True if the XML annotation exists.
- `has_annotation_json` (`bool`): True if the JSON annotation exists.
- `patient_id` (`str`)
- `source` (`str`): originating institution/collection (e.g. "rumc", "tcga", "nki").
- `specimen_type` (`str`): "resection" or "biopsy".
- `scanner` (`str`)
- `split` (`str`): "development" or "evaluation".
- `validation_fold` (`str`): cross-validation fold within the development split; `None` for evaluation slides.

### iCAIRD Cervix Dataset

**Location**: MLflow artifacts

**Output layout**:
```text
icaird_cervix/
  errors.log (slides listed in index.csv but missing on disk)
  slides_metadata.csv (metadata for the valid slides)
  summary.csv (aggregate statistics)
```

**CSV metadata row schema (one row = one slide)**:
- `slide_id` (`str`): matches the `slide` column of the dataset's `index.csv`.
- `slide_path` (`str`)
- `category` (`str`): e.g. "normal_inflammation", "low_grade", "high_grade", "malignant".
- `subcategory` (`str`): dataset-provided subcategory of `category`.
- `split` (`str`): the dataset-provided train/valid split.
- `has_annotation` (`bool`): True if the slide's GeoJSON annotation file exists.