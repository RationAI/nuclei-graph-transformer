"""Polygon mask for visualising nuclei predictions.

The code was adapted by Lukáš Hudec from the Nuclei Segmentation repository of Matěj Pekár.
"""

import json
from collections.abc import Iterable

from PIL import Image, ImageDraw
from rationai.masks.annotations import PolygonMask
from torch import Tensor

from nuclei_graph.typing import Region


class NucleiMask(PolygonMask[Region]):
    def __init__(
        self,
        base_mpp_x: float,
        base_mpp_y: float,
        mask_size: tuple[int, int],
        mask_mpp_x: float,
        mask_mpp_y: float,
        raw_cells_path: str,
        labels: Tensor | None = None,
    ) -> None:
        """Nuclei mask for predictions.

        Args:
            base_mpp_x (float): WSI microns per pixel in the x-axis.
            base_mpp_y (float): WSI microns per pixel in the y-axis.
            mask_size (tuple[int, int]): Mask dimensions.
            mask_mpp_x (float): Mask microns per pixel in the x-axis.
            mask_mpp_y (float): Mask microns per pixel in the y-axis.
            raw_cells_path (str): Path to the raw cells JSON file.
            labels (Tensor | None, optional): Prediction labels, if None, every cell is labeled with 255. Defaults to None.
        """
        self.base_mpp_x = base_mpp_x
        self.base_mpp_y = base_mpp_y
        with open(raw_cells_path) as json_data:
            self.cells = json.load(json_data)
        self.labels = labels
        super().__init__(
            mode="L",
            mask_size=mask_size,
            mask_mpp_x=mask_mpp_x,
            mask_mpp_y=mask_mpp_y,
        )

    @property
    def regions(self) -> Iterable[tuple[Region, int]]:
        for i, cell in enumerate(self.cells["cells"]):
            if self.labels is not None:
                yield cell["contour"], int(self.labels[i].item() * 255)
            else:
                yield cell["contour"], 255

    def get_region_coordinates(self, region: Region) -> Iterable[tuple[float, float]]:
        yield from region

    @property
    def annotation_mpp_x(self) -> float:
        return self.base_mpp_x

    @property
    def annotation_mpp_y(self) -> float:
        return self.base_mpp_y

    def __call__(self) -> Image.Image:
        """Generates the polygon mask as an image.

        This method reads the regions and draws them as polygons on an image canvas.
        Each region is drawn with its corresponding label as the outline and fill
        color.

        Returns:
            The generated mask with the polygons drawn on it.
        """
        scale_factor_x = self.annotation_mpp_x / self.mask_mpp_x
        scale_factor_y = self.annotation_mpp_y / self.mask_mpp_y

        mask = Image.new(self.mode, size=self.mask_size)
        canvas = ImageDraw.Draw(mask)

        for region, label in self.regions:
            polygon = [
                (x * scale_factor_x, y * scale_factor_y)
                for x, y in self.get_region_coordinates(region)
            ]
            if self.labels is not None:
                canvas.polygon(xy=polygon, outline=label, fill=label)
            else:
                canvas.polygon(xy=polygon, outline=label)

        return mask
