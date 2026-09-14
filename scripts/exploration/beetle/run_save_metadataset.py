from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-data-exploration-beetle",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m exploration.beetle.save_metadataset +data=sources/beetle",
    ],
    storage=[storage.secure.PROJECTS],
)
