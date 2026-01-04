## Preprocessing Workflow

The preprocessing pipeline follows these steps:

1. **Nuclei Segmentation** (`nuclei_segmentation.py`)  
   Segments nuclei in whole-slide images and stores the outputs as partitioned Parquet tables.

2. **Annotation Masks** (`annotation_masks.py`)  
   Generates binary masks for annotated carcinoma regions using XML annotation files by expert pathologists. 

3. **Label Nuclei** (`annotation_labels.py`)  
   Assigns labels to segmented nuclei by checking polygon overlap with expert annotation masks.

4. **CAM Masks Preparation** (`merge_cam_masks.py`)  
   Aggregates generated CAM masks from multiple MLflow runs into a single location for convenience.

5. **CAM-based Label Refinement** (`cam_refinement.py`)  
   Computes CAM refinements by thresholding positive/negative regions and storing the average CAM intensity for each nucleus for soft labeling.

6. **Map Slides to Nuclei** (`metadata_mapping.py`)  
   Creates a mapping between slide IDs, patient IDs, Mirax files, carcinoma status, and the nuclei segmentation folders.
