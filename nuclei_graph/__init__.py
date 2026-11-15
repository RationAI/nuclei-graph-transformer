import os


# Arrow allocates with malloc (instead of mimalloc), so that the memory it frees can be
# returned to the OS (`nuclei_graph.data.memory`). Arrow reads this when pyarrow is
# imported, which the package's modules do after this.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

__all__: list[str] = []
