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
- `has_annotation` (`bool`)
- `patient_id` (`str`): 4-digit unique patient identifier.
- `case_id` (`str`): A combination of year and patient_id.

### PANDA Challenge Dataset

**Location**: MLflow artifacts

**Output layout**:  
```text
panda/
  errors.log (invalid slides — empty, corrupted encoding...)
  slides_metadata.csv (metadata for the valid slides)
  summary.csv (aggregate statistics)
```

**CSV metadata row schema (one row = one slide)**:  
- `slide_id` (`str`): 2-character hex string identifier for each slide.
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