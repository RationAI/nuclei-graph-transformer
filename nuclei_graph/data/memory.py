"""Returning freed host memory to the OS.

Reading a slide's nuclei (`pd.read_parquet`) allocates gigabytes for the largest
slides (~8 GB for 3.9M nuclei), which are freed once the crop is cut. Neither Arrow's
default allocator (mimalloc) nor glibc returns that memory to the OS, so each
persistent DataLoader worker would keep ~4.7 GB after its largest slide (iCAIRD:
16 training workers, kept also during validation). With Arrow on the system
allocator (`ARROW_DEFAULT_MEMORY_POOL`, set in `nuclei_graph/__init__.py`) and the
heap trimmed after each batch, a worker keeps ~0.4 GB.
"""

import ctypes
import ctypes.util
import warnings
from collections.abc import Callable

import pyarrow as pa


_libc_name = ctypes.util.find_library("c")
_malloc_trim = (
    getattr(ctypes.CDLL(_libc_name), "malloc_trim", None) if _libc_name else None
)


def check_system_allocator() -> None:
    """Warns when Arrow keeps the memory it frees (it does not allocate with malloc).

    `release_memory` returns only malloc's freed memory. Arrow chooses its allocator
    when pyarrow is imported: `pa.set_memory_pool` later does not reach the Parquet
    reader.
    """
    backend = pa.default_memory_pool().backend_name
    if backend != "system":
        warnings.warn(
            f"Arrow allocates with {backend}, which keeps the memory it frees: import "
            "nuclei_graph before pyarrow, or set ARROW_DEFAULT_MEMORY_POOL=system",
            stacklevel=2,
        )


def release_memory() -> None:
    """Returns the freed memory of Arrow and of the C heap (glibc only) to the OS."""
    pa.default_memory_pool().release_unused()
    if _malloc_trim is not None:
        _malloc_trim(0)


class ReleaseMemory[T, B]:
    """A collate function that returns the freed memory to the OS after each batch.

    The batch's samples, and the slides they were cut from, are freed by then.
    """

    def __init__(self, collate_fn: Callable[[list[T]], B]) -> None:
        self.collate_fn = collate_fn

    def __call__(self, batch: list[T]) -> B:
        collated = self.collate_fn(batch)
        release_memory()
        return collated
