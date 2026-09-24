from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-icaird-cervix-qc-masks",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="16Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.qc_analysis +data=sources/icaird_cervix",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)