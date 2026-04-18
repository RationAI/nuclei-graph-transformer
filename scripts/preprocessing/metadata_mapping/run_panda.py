from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-metadata-mapping-panda",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "export MLFLOW_TRACKING_USERNAME='...'",
        "export MLFLOW_TRACKING_PASSWORD='...'",
        "export MLFLOW_TRACKING_URI='https://mlflow.rationai.cloud.e-infra.cz/'",
        "uv sync --frozen",
        "uv run python -m preprocessing.metadata_mapping.panda +experiment=preprocessing/metadata_mapping/...",
    ],
    storage=[storage.public.DATA, storage.public.PROJECTS],
)
