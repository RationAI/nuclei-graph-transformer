## Preprocessing Workflow

The preprocessing pipeline follows these steps:

1. **Nuclei Segmentation** (`nuclei_segmentation.py`)  
   Segments nuclei in whole-slide images and stores the outputs as partitioned Parquet tables.

2. **Annotation Masks** (`annotation_masks.py`)  
   Generates binary masks for annotated carcinoma regions using XML annotation files by expert pathologists. 

3. **CAM Masks Preparation** (`merge_cam_masks.py`)  
   Aggregates generated CAM masks from multiple MLflow runs into a single location for convenience.

4. **Label Nuclei** (`label_nuclei.py`)  
   Assigns labels to segmented nuclei by checking polygon overlap with expert annotation masks.

5. **Compute CAM Label Indicators** (`compute_cam_indicators.py`)  
   Computes label indicators for positive slides by intersecting annotation labels with CAM masks.

6. **Map Slides to Nuclei** (`metadata_mapping.py`)  
   Creates a mapping between slide IDs, patient IDs, Mirax files, carcinoma status and nuclei segmentation folders.
