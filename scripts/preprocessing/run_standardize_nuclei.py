from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-nuclei-standardization",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=16,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.standardize_nuclei +data=sources/panda",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
