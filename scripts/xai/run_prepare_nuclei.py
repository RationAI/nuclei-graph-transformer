from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-xai-prepare-nuclei",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=16,
    memory="128Gi",
    public=False,
    script=[
        "git clone --branch xai-sae https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "export MLFLOW_TRACKING_USERNAME='...'",
        "export MLFLOW_TRACKING_PASSWORD='...'",
        "export MLFLOW_TRACKING_URI='https://mlflow.rationai.cloud.e-infra.cz/'",
        "uv sync --frozen",
        "uv run python -m xai.prepare_nuclei +experiment=xai/radboud",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
