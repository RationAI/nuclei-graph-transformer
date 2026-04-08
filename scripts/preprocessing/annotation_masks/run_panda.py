from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-annot-masks-panda",
    username=...,
    cpu=16,
    memory="64Gi",
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "git checkout panda/annotation-masks",
        "git checkout a4778033fc80e620e52578e47666c05ff50447cd",
        "export MLFLOW_TRACKING_USERNAME=...",
        "export MLFLOW_TRACKING_PASSWORD=...",
        "export MLFLOW_TRACKING_URI='https://mlflow.rationai.cloud.e-infra.cz/'",
        "uv sync",
        "uv run python -m preprocessing.annotation_masks.panda",
    ],
    public=True,
    storage=[storage.public.DATA],
)