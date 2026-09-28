from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-icaird-endometrium-dataset-isyntax2tif-conversion-batch-b",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="128Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.isyntax2tif +experiment=preprocessing/isyntax2tif/icaird_endometrium_batch_b +data=sources/icaird_endometrium",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
