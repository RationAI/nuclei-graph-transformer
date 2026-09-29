from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-beetle-qc-masks",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.qc_analysis +data=sources/beetle +experiment=preprocessing/quality_control/beetle",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
