from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-polygons-rasterization",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=2,
    memory="190Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m visualization.polygons2raster +experiment=visualization/... +data=sources/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
