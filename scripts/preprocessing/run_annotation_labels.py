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
        "export MLFLOW_TRACKING_USERNAME=...",
        "export MLFLOW_TRACKING_PASSWORD=...",
        "export MLFLOW_TRACKING_URI='https://mlflow.rationai.cloud.e-infra.cz/'",
        "uv sync --frozen",
        "uv run python -m preprocessing.annotation_labels +experiment=preprocessing/annotation_labels/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
