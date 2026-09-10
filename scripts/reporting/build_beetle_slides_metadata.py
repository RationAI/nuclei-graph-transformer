"""Build a local metadata CSV listing WSI and annotation mask locations for BEETLE.

Reads the dataset's own `data_overview.csv` (paths there are relative: `wsi_path`
to `<root>/data/`, `annotation_mask_path` to `<root>/`), resolves them to absolute
paths and writes a clean CSV meant to be consumed by the BEETLE report config
(`configs/visualization/report/beetle.yaml`).

Usage:
    python -m scripts.reporting.build_beetle_slides_metadata
    python -m scripts.reporting.build_beetle_slides_metadata --root /mnt/projects/nuclei_based_wsi_analysis/BEETLE
"""

import argparse
import csv
from pathlib import Path


DEFAULT_ROOT = Path("/mnt/projects/nuclei_based_wsi_analysis/BEETLE")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "beetle" / "slides_metadata.csv"

METADATA_COLUMNS = ["source", "specimen_type", "scanner", "split", "validation_fold"]


def build_metadata(root: Path) -> list[dict[str, str]]:
    overview_path = root / "data_overview.csv"
    rows_out = []
    with overview_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("name"):
                continue

            wsi_path = root / "data" / row["wsi_path"]
            mask_path = root / row["annotation_mask_path"] if row["annotation_mask_path"] else None

            if not wsi_path.exists():
                print(f"WARNING: WSI not found, skipping {row['name']}: {wsi_path}")
                continue

            rows_out.append(
                {
                    "item_name": row["name"],
                    "wsi_path": str(wsi_path),
                    "mask_path": str(mask_path) if mask_path and mask_path.exists() else "",
                    **{col: row[col] for col in METADATA_COLUMNS},
                }
            )
    return rows_out


def write_metadata(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item_name", "wsi_path", "mask_path", *METADATA_COLUMNS]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_metadata(args.root)
    write_metadata(rows, args.output)

    n_with_mask = sum(1 for row in rows if row["mask_path"])
    print(f"Wrote {len(rows)} slides ({n_with_mask} with annotation masks) to {args.output}")


if __name__ == "__main__":
    main()
