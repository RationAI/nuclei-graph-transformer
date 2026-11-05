from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-icaird-cervix-nuclei-segmentation-cellpose",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=40,
    memory="100Gi",
    gpu="A100",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.nuclei_segmentation +data=sources/icaird_cervix +experiment=preprocessing/nuclei_segmentation/icaird_cervix_cellpose",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
