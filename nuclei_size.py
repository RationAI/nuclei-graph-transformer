import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts


def main() -> None:
    slides_df = pd.read_parquet(
        download_artifacts(
            "mlflow-artifacts:/97/28a05b3cbc2a434eae6221f103d56020/artifacts/tile_level_annotations/slides_mapping.parquet"
        )
    )

    pos_slides = slides_df[slides_df["is_carcinoma"]].sample(5)
    neg_slides = slides_df[slides_df["is_carcinoma"] == False].sample(5)
    sample_slides = pd.concat([pos_slides, neg_slides])

    widths = []
    heights = []

    for nuclei_path in sample_slides["slide_nuclei_path"]:
        df = pd.read_parquet(nuclei_path)

        for poly in df["polygon"]:
            poly = np.array(poly)
            poly = poly.reshape(-1, 2)

            min_x, min_y = poly.min(axis=0)
            max_x, max_y = poly.max(axis=0)

            widths.append(max_x - min_x)
            heights.append(max_y - min_y)

    widths = np.array(widths)
    heights = np.array(heights)

    print(f"Sampled {len(widths):,} nuclei from 10 slides.")
    print(f"Average Nucleus: {widths.mean():.1f} x {heights.mean():.1f} pixels")
    print(
        f"95th Percentile: {np.percentile(widths, 95):.0f} x {np.percentile(heights, 95):.0f} pixels"
    )
    print(
        f"99th Percentile: {np.percentile(widths, 99):.0f} x {np.percentile(heights, 99):.0f} pixels"
    )
    print(f"Absolute Max:    {widths.max():.0f} x {heights.max():.0f} pixels")


if __name__ == "__main__":
    main()
