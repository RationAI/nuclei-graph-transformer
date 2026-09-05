from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-icaird-cervical-dataset-isyntax2tif-conversion",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="200Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.isyntax2tif +data=sources/icaird_cervix",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)