from kube_jobs import storage, submit_job


EXPERIMENT_NAME = (
    ...
)  # one of "annotation_labels", "cam_labels", "outline_polygons", "predictions"

submit_job(
    job_name="nuclei-graph-polygons-rasterization",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=2,
    memory="190Gi",
    public=False,
    script=[
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run python -m visualization.polygons2raster +experiment=/visualization/{EXPERIMENT_NAME}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
