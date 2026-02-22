import os
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@with_cli_args(["+preprocessing=upload_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(_: DictConfig, logger: MLFlowLogger) -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        mask_raw = tmp_path / "mask_raw"
        mask_closed = tmp_path / "mask"
        tissue = tmp_path / "tissue_mask"
        tissue_eroded = tmp_path / "tissue_mask_eroded"
        mask_raw.mkdir()
        mask_closed.mkdir()
        tissue.mkdir()
        tissue_eroded.mkdir()

        slide_id = "2025_09852-01-02-05-AMACR"
        raw_path = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_raw/2025_09852-01-02-05-AMACR.tiff"
        refined_path = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/amacr_mask_refined/2025_09852-01-02-05-AMACR.tiff"
        tissue_mask_path = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask/2025_09852-01-02-05-AMACR.tiff"
        tissue_mask_eroded_path = "/mnt/projects/nuclei_based_wsi_analysis/amacr_ground_truth_test/tissue_mask_eroded/2025_09852-01-02-05-AMACR.tiff"

        dst_raw = mask_raw / f"{slide_id}.tiff"
        os.symlink(raw_path, dst_raw)
        dst_closed = mask_closed / f"{slide_id}.tiff"
        os.symlink(refined_path, dst_closed)
        dst_tissue = tissue / f"{slide_id}.tiff"
        os.symlink(tissue_mask_path, dst_tissue)
        dst_tissue_eroded = tissue_eroded / f"{slide_id}.tiff"
        os.symlink(tissue_mask_eroded_path, dst_tissue_eroded)

        print("Uploading to MLflow...")
        logger.log_artifacts(local_dir=tmp_dir)
        print("Upload complete.")


if __name__ == "__main__":
    main()
