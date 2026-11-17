"""Carcinoma foci of the training slides (study small_foci).

A focus is a connected component of a slide's Gleason 3-5 mask after a 50 µm closing
(`postprocessing.small_foci_radboud`, stage `foci_table`). The focus table lists the carcinoma
nuclei of each slide with their position in the id-sorted nuclei (`index`, the order of
`BaseCropDataset.get_nuclei`), the 64-bit prefix of their id (`key`), their focus and its size in
carcinoma nuclei (`focus_nuclei`). Slides are keyed by `slide_key` (a hash of the slide id).

Used by `NucleiClassificationDataset` for focus-balanced seeding (`pos_seed_mode="focus"`) and to
return each nucleus's focus size (the focus-size loss weight of `NucleiModelMetaArch`).
"""

import hashlib
from random import choice, randrange
from typing import NamedTuple

import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray


FOCI_COLUMNS = ("slide_key", "slide_nuclei", "index", "key", "focus", "focus_nuclei")


def slide_key(slide_id: str) -> int:
    """The first 64 bits of the SHA-256 of the slide id."""
    return int.from_bytes(hashlib.sha256(slide_id.encode()).digest()[:8], "big")


def nucleus_keys(ids: NDArray[np.generic]) -> NDArray[np.uint64]:
    """The 64-bit prefix of each (64-hex-character) nucleus id."""
    ids_str = np.asarray(ids).astype(str)
    assert np.char.str_len(ids_str).min(initial=64) == 64, (
        "unexpected nucleus id format"
    )
    prefix = "".join(s[:16] for s in ids_str)
    return np.frombuffer(bytes.fromhex(prefix), dtype=">u8").astype(np.uint64)


def keys_checksum(keys: NDArray[np.uint64]) -> str:
    """SHA-256 of the keys in their order (big-endian bytes)."""
    return hashlib.sha256(keys.astype(">u8").tobytes()).hexdigest()


class SlideFoci(NamedTuple):
    index: NDArray[np.int32]  # carcinoma nuclei, positions in the id-sorted nuclei
    focus: NDArray[np.int32]
    focus_nuclei: NDArray[np.int32]


class FociTable:
    """The focus table, per slide; the nucleus keys are kept only as a per-slide checksum."""

    def __init__(self, uri: str) -> None:
        path = download_artifacts(uri) if "://" in uri or ":/" in uri else uri
        df = pd.read_parquet(path, columns=list(FOCI_COLUMNS))
        df = df.sort_values(["slide_key", "index"], ignore_index=True)
        self.index = df["index"].to_numpy(np.int32)
        self.focus = df["focus"].to_numpy(np.int32)
        self.focus_nuclei = df["focus_nuclei"].to_numpy(np.int32)
        skeys = df["slide_key"].to_numpy(np.uint64)
        starts = np.flatnonzero(np.r_[True, skeys[1:] != skeys[:-1]])
        ends = np.r_[starts[1:], len(skeys)]
        keys = df["key"].to_numpy(np.uint64)
        n = df["slide_nuclei"].to_numpy(np.int64)
        self.slides: dict[int, tuple[int, int, int, str]] = {
            int(skeys[s]): (int(s), int(e), int(n[s]), keys_checksum(keys[s:e]))
            for s, e in zip(starts, ends, strict=True)
        }
        self._checked: set[int] = set()

    def __len__(self) -> int:
        return len(self.slides)

    def get(
        self, slide_id: str, ids: NDArray[np.generic] | None = None
    ) -> SlideFoci | None:
        """The slide's carcinoma nuclei and foci (None if the table has no carcinoma for it).

        With `ids` (the slide's nuclei ids in `get_nuclei` order), asserts once per slide that
        the table was built for exactly these nuclei.
        """
        k = slide_key(slide_id)
        if k not in self.slides:
            return None
        s, e, n, checksum = self.slides[k]
        foci = SlideFoci(self.index[s:e], self.focus[s:e], self.focus_nuclei[s:e])
        if ids is not None and k not in self._checked:
            assert len(ids) == n, f"focus table: {n} nuclei, the slide has {len(ids)}"
            assert keys_checksum(nucleus_keys(ids[foci.index])) == checksum, (
                "focus table: the carcinoma nuclei differ from the slide's"
            )
            self._checked.add(k)
        return foci

    def focus_sizes(
        self, slide_id: str, n: int, ids: NDArray[np.generic] | None = None
    ) -> NDArray[np.float32]:
        """Focus size (carcinoma nuclei) of each of the slide's `n` nuclei; 0 outside carcinoma."""
        sizes = np.zeros(n, np.float32)
        foci = self.get(slide_id, ids)
        if foci is not None:
            sizes[foci.index] = foci.focus_nuclei
        return sizes

    def focus_seed(
        self,
        slide_id: str,
        pos_seeds: list[int],
        ids: NDArray[np.generic] | None = None,
    ) -> list[int]:
        """One positive seed: a focus drawn uniformly, then a positive nucleus within it.

        Only foci holding at least one of `pos_seeds` (the slide's supervised positive nuclei)
        are drawn. Without foci, a seed uniform over `pos_seeds` (the default seeding).
        """
        foci = self.get(slide_id, ids)
        if not pos_seeds:
            return []
        if foci is None:
            return [choice(pos_seeds)]
        keep = np.isin(foci.index, np.asarray(pos_seeds))
        if not keep.any():
            return [choice(pos_seeds)]
        index, focus = foci.index[keep], foci.focus[keep]
        labels = np.unique(focus)
        drawn = labels[randrange(len(labels))]
        members = index[focus == drawn]
        return [int(members[randrange(len(members))])]
