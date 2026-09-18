from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-data-exploration-icaird-cervix",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m exploration.icaird_cervix.save_metadataset +data=sources/icaird_cervix",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
