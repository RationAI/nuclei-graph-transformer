from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-roi-sampling-icaird-cervix",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=6,
    memory="64Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.roi_sampling.icaird_cervix",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
