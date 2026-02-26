from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-nuclei-segmentation",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=20,
    memory="80Gi",
    gpu="H100",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.nuclei_segmentation +data=sources/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
