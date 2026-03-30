from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-data-exploration-prostate_cancer",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m exploration.prostate_cancer.save_metadataset +data=sources/prostate_cancer",
    ],
    storage=[storage.secure.DATA],
)
