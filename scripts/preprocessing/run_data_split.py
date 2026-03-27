from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-data-split",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.data_split +experiment=preprocessing/data_split/...",
    ],
    storage=[storage.secure.DATA],
)
