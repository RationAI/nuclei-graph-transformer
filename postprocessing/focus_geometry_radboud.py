# ruff: noqa: RUF002  (µ, ², minus signs in text are intended)
"""Per-nucleus annotations and carcinoma-focus geometry of the PANDA Radboud test slides.

Used by ``postprocessing.focus_size_analysis_radboud`` (stage ``geometry``); one worker per slide.

From the Radboud label mask at level 0 (0 background, 1 stroma, 2 benign epithelium, 3–5 Gleason
pattern), channel 0:

- Vertex rule of the thesis labels (``preprocessing/unipolar_heatmap_labels.py``): the polygon
  vertices, scaled by mask / slide extent and rounded, sample the mask; carcinoma = ≥
  ``overlap_threshold`` of the vertices with value ≥ 3. It must reproduce ``annot_label``.
- Class (``cls``): carcinoma nuclei 3 / 4 / 5 = the pattern with most vertices (ties → higher);
  others 2 (benign epithelium) / 1 (stroma) when ≥ ``overlap_threshold`` of the vertices have that
  value, else 0 (other: background or mixed).
- Geometry on the carcinoma mask (value ≥ 3) subsampled by ``stride`` (pixel = stride × mpp µm):
  for each closing diameter d (disk of radius d / 2; d = 0: no closing), foci = 8-connected
  components of the closed mask (the slide border counts as outside). Per nucleus (centroid
  pixel): ``sd_<d>`` signed distance in µm to the closed mask's boundary (+ inside, − outside;
  (EDT − ½ pixel) × pixel size), ``focus_<d>`` the focus it lies in or, outside, the nearest one
  (−1 without carcinoma). Per focus: closed area, carcinoma (pattern ≥ 3) area and the area of each
  pattern, mm².
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from einops import rearrange
from numpy.typing import NDArray
from openslide import OpenSlide
from scipy import ndimage


CONNECTIVITY = np.ones((3, 3), bool)


def nucleus_keys(
    ids: pd.Series,
) -> NDArray[np.uint64]:  # as bootstrap_sources.nucleus_keys
    assert (ids.str.len() == 64).all(), "unexpected nucleus id format"
    prefix = "".join(ids.str.slice(0, 16))
    return np.frombuffer(bytes.fromhex(prefix), dtype=">u8").astype(np.uint64)


def read_mask(path: Path) -> NDArray[np.uint8]:
    """Channel 0 of level 0, decoded tile by tile (no full RGB copy)."""
    with tifffile.TiffFile(path) as tf:
        z = zarr.open(tf.aszarr(level=0), mode="r")
        if z.ndim == 3:
            return np.asarray(z[:, :, 0])
        return np.asarray(z[:])


def closing(mask: NDArray[np.bool_], radius: float) -> NDArray[np.bool_]:
    """Binary closing with a Euclidean disk of `radius` pixels (exact, via distance transforms)."""
    if radius <= 0:
        return mask.copy()
    pad = int(np.ceil(radius)) + 2
    m = np.pad(mask, pad)
    dilated = ndimage.distance_transform_edt(~m) <= radius
    closed = ndimage.distance_transform_edt(dilated) > radius
    return closed[pad:-pad, pad:-pad] | mask


def vertex_classes(
    polygons: list, mask: NDArray[np.uint8], scale: tuple[float, float], thr: float
) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
    """Vertex rule (as unipolar_heatmap_labels.label_slide) and the class of every nucleus.

    `polygons` are flat (x0, y0, x1, y1, …) level-0 vertices, `scale` = mask / slide extent.
    Returns y_rule (≥ `thr` of the vertices with value ≥ 3) and cls: the Gleason pattern with
    most vertices (ties → higher) for carcinoma nuclei, else 2 / 1 when ≥ `thr` of the vertices
    are benign epithelium / stroma, else 0.
    """
    my, mx = mask.shape
    poly = rearrange(polygons, "b (v d) -> b v d", d=2)
    c = np.round(poly * np.array(scale)).astype(int)
    v = mask[np.clip(c[..., 1], 0, my - 1), np.clip(c[..., 0], 0, mx - 1)]
    y_rule = (np.mean(v >= 3, axis=1) >= thr).astype(np.int8)
    counts = np.stack([(v == k).sum(1) for k in range(6)], 1)  # (n, 6)
    pattern = 3 + (2 - np.argmax(counts[:, 5:2:-1], axis=1))  # ties -> higher pattern
    frac = counts / v.shape[1]
    cls = np.where(
        y_rule == 1,
        pattern,
        np.where(frac[:, 2] >= thr, 2, np.where(frac[:, 1] >= thr, 1, 0)),
    ).astype(np.int8)
    return y_rule, cls


def slide_geometry(job: dict) -> dict:
    """Writes the slide's nucleus and focus tables; returns its checks."""
    s, sid = int(job["slide"]), job["slide_id"]
    out_n = Path(job["out_dir"]) / "nuclei" / f"{sid}.parquet"
    out_f = Path(job["out_dir"]) / "foci" / f"{sid}.parquet"
    out_c = Path(job["out_dir"]) / "checks" / f"{sid}.json"
    if out_n.exists() and out_f.exists() and out_c.exists():
        return json.loads(out_c.read_text())

    nuclei = pd.read_parquet(
        job["nuclei_path"], columns=["id", "polygon", "centroid"]
    ).sort_values("id", ignore_index=True)
    keys = nucleus_keys(nuclei["id"])
    mask = read_mask(Path(job["mask_path"]))
    my, mx = mask.shape
    with OpenSlide(job["slide_path"]) as slide:
        wx, wy = slide.dimensions
    sx, sy = mx / wx, my / wy

    y_rule, cls = vertex_classes(
        nuclei["polygon"].tolist(), mask, (sx, sy), float(job["overlap_threshold"])
    )

    checks = {
        "slide": s,
        "nuclei": len(nuclei),
        "mask_shape": [my, mx],
        "scale": [sx, sy],
        "rule_carcinoma": int(y_rule.sum()),
    }
    if job["label_path"]:
        lab = pd.read_parquet(job["label_path"], columns=["id", job["label_column"]])
        lab = lab.sort_values("id", ignore_index=True)
        assert np.array_equal(lab["id"].to_numpy(), nuclei["id"].to_numpy()), (
            f"{sid}: label ids != nuclei ids"
        )
        checks["label_mismatch"] = int(
            (lab[job["label_column"]].to_numpy() != y_rule).sum()
        )
    else:
        checks["label_mismatch"] = (
            None  # labels are 0 by the rule (is_carcinoma = False)
        )

    # geometry on the subsampled carcinoma mask
    st = int(job["stride"])
    sub = mask[::st, ::st]
    del mask
    carc = sub >= 3
    px_um = st * float(job["mpp"])
    px_mm2 = (px_um / 1000.0) ** 2
    cent = np.stack(nuclei["centroid"].to_numpy())  # (x, y), level 0
    cy = np.clip((cent[:, 1] * sy / st).astype(int), 0, sub.shape[0] - 1)
    cx = np.clip((cent[:, 0] * sx / st).astype(int), 0, sub.shape[1] - 1)

    table = {"slide": np.full(len(keys), s, np.int16), "key": keys, "cls": cls}
    table["y_rule"] = y_rule
    foci = []
    for d in job["closings_um"]:
        d = int(d)
        if not carc.any():
            table[f"sd_{d}"] = np.full(len(keys), np.nan, np.float32)
            table[f"focus_{d}"] = np.full(len(keys), -1, np.int32)
            continue
        closed = closing(carc, d / 2 / px_um)
        lab, n = ndimage.label(closed, structure=CONNECTIVITY)
        inside_d = ndimage.distance_transform_edt(np.pad(closed, 1))[1:-1, 1:-1]
        outside_d, idx = ndimage.distance_transform_edt(~closed, return_indices=True)
        inside = closed[cy, cx]
        sd = np.where(inside, inside_d[cy, cx] - 0.5, -(outside_d[cy, cx] - 0.5))
        table[f"sd_{d}"] = (sd * px_um).astype(np.float32)
        table[f"focus_{d}"] = lab[idx[0][cy, cx], idx[1][cy, cx]].astype(np.int32)
        del inside_d, outside_d, idx
        flat = lab.ravel()
        area = np.bincount(flat, minlength=n + 1)
        by_val = np.bincount(flat * 6 + sub.ravel().clip(0, 5), minlength=(n + 1) * 6)
        by_val = by_val.reshape(n + 1, 6)
        foci.append(
            pd.DataFrame(
                {
                    "slide": s,
                    "closing_um": d,
                    "focus": np.arange(1, n + 1, dtype=np.int32),
                    "area_mm2": area[1:] * px_mm2,
                    "carcinoma_area_mm2": by_val[1:, 3:].sum(1) * px_mm2,
                    "g3_area_mm2": by_val[1:, 3] * px_mm2,
                    "g4_area_mm2": by_val[1:, 4] * px_mm2,
                    "g5_area_mm2": by_val[1:, 5] * px_mm2,
                }
            )
        )
        checks[f"foci_{d}"] = n
    checks["carcinoma_area_mm2"] = float(carc.sum() * px_mm2)
    checks["pixel_um"] = px_um

    for p in (out_n, out_f, out_c):
        p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_parquet(out_n, index=False)
    (
        pd.concat(foci, ignore_index=True)
        if foci
        else pd.DataFrame(
            columns=[
                "slide",
                "closing_um",
                "focus",
                "area_mm2",
                "carcinoma_area_mm2",
                "g3_area_mm2",
                "g4_area_mm2",
                "g5_area_mm2",
            ]
        )
    ).to_parquet(out_f, index=False)
    out_c.write_text(json.dumps(checks))
    return checks
