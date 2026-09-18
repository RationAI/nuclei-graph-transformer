"""Build a local metadata CSV listing WSI and annotation mask locations for BEETLE.

Scans the dataset's flat directory of WSI files and matches each one to an annotation
mask of the same stem (if any) in the masks directory. There is no overview CSV
available for this dataset. Writes a clean CSV meant to be consumed by the BEETLE
report config (`configs/visualization/report/beetle.yaml`).

Note: `--masks-dir` defaults to the *rescaled* masks (see
`scripts/reporting/rescale_beetle_masks.py`), not the raw label masks at
`annotations/masks/` -- the raw label values (0-3) are indistinguishable from
black under xOpat's shaders (`edge`/`classify` render off 0-255 channel intensity).

Usage:
    python -m scripts.reporting.build_beetle_slides_metadata
    python -m scripts.reporting.build_beetle_slides_metadata --wsis-dir /mnt/projects/nuclei_based_wsi_analysis/BEETLE/data/images/development/wsis --masks-dir /mnt/projects/nuclei_based_wsi_analysis/BEETLE/annotations/masks_rescaled
"""

import argparse
import csv
from pathlib import Path


DEFAULT_WSIS_DIR = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/BEETLE/data/images/development/wsis"
)
DEFAULT_MASKS_DIR = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/BEETLE/annotations/masks_rescaled"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "data" / "beetle" / "slides_metadata.csv"
)

WSI_EXTENSIONS = {".tif", ".tiff", ".svs"}
MASK_EXTENSIONS = {".tif", ".tiff"}


def build_metadata(wsis_dir: Path, masks_dir: Path) -> list[dict[str, str]]:
    masks_by_stem = {
        p.stem: p
        for p in masks_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS
    }

    rows_out = []
    for wsi_path in sorted(wsis_dir.iterdir()):
        if wsi_path.suffix.lower() not in WSI_EXTENSIONS:
            continue

        mask_path = masks_by_stem.get(wsi_path.stem)

        rows_out.append(
            {
                "item_name": wsi_path.stem,
                "wsi_path": str(wsi_path),
                "mask_path": str(mask_path) if mask_path else "",
            }
        )
    return rows_out


def write_metadata(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item_name", "wsi_path", "mask_path"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsis-dir", type=Path, default=DEFAULT_WSIS_DIR)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_metadata(args.wsis_dir, args.masks_dir)
    write_metadata(rows, args.output)

    n_with_mask = sum(1 for row in rows if row["mask_path"])
    print(
        f"Wrote {len(rows)} slides ({n_with_mask} with annotation masks) to {args.output}"
    )


if __name__ == "__main__":
    main()
