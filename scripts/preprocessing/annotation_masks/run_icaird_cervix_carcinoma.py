from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-annotation-masks-icaird-cervix-carcinoma-masks",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=6,
    memory="64Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.annotation_masks.icaird_cervix_carcinoma",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)

