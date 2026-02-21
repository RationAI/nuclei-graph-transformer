## Preprocessing Workflow

The preprocessing pipeline follows these steps:

1. **Nuclei Segmentation** (`nuclei_segmentation.py`)  
   Segments nuclei in whole-slide images and stores the outputs as partitioned Parquet tables.

2. **Annotation Masks** (`annotation_masks.py`)  
   Generates binary masks for annotated carcinoma regions using XML annotation files by expert pathologists. 

3. **Annotation-based Nuclei Labeling** (`annotation_labels.py`)  
   Assigns labels to segmented nuclei by checking polygon overlap with expert annotation masks.

4. **CAM Masks Preparation** (`merge_cam_masks.py`)  
   Aggregates generated CAM masks from multiple MLflow runs into a single location for convenience.

5. **CAM-based Nuclei Labeling** (`cam_labels.py`)  
   Computes CAM pseudo labels by thresholding positive/negative regions and storing the average CAM intensity for each nucleus (for loss weighting, etc.).

6. **Map Slides to Nuclei** (`metadata_mapping.py`)  
   Creates a mapping between slide IDs, patient IDs, Mirax files, carcinoma status, and the nuclei segmentation folders.

### AMACR Ground Truth

7. **Tissue Masks** (`tissue_masks.py`, `tissue_masks_erode_edges.py`)  
   Computes tissue masks for filtering out the staining at the borders of tissue sections.
 
8. **AMACR Mask** (`amacr_masks.py`)  
   Computes AMACR mask by isolating the stain using adaptive color deconvolution, hysteresis thresholding, and morphological cleaning.

9. **AMACR Mask Refinement** (`amacr_masks_refinement.py`)  
   Filters out artifacts and dilates the mask to obtain AMACR-based annotation masks.

