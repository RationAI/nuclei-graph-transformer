from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-data-exploration",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m exploration.<DATASET_NAME>.save_metadataset +data=sources/...",
    ],
    storage=[storage.secure.DATA],
)