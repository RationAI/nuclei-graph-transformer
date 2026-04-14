from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-annotation-nuclei-labeling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.annotation_labels +experiment=preprocessing/annotation_labels/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
