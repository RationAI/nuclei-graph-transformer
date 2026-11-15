"""Training job: iCAIRD cervix, nucleus level, shape + RoVE + k-NN attention (k = 24).

Config: configs/experiment/modeling/training/nuclei_level/icaird_cervix.yaml. The slide tables
are on the public MLflow (experiment 37) and the nuclei / labels on the secure PROJECTS storage,
so the job runs on the secure cluster with the public tracking URI. Fill in the credentials
locally when submitting; never commit them. One job = one seed (random); submit three times for
three seeds.

Memory: the data module reads the whole label folder (~193M nuclei, ~33 GiB in pandas) once for
training and once for validation before filtering, hence 128 GiB.
"""

from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-train-icaird-cervix-k24-efd-rove-knn",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=16,
    memory="128Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone --branch xai-sae https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "export MLFLOW_TRACKING_USERNAME='...'",
        "export MLFLOW_TRACKING_PASSWORD='...'",
        "export MLFLOW_TRACKING_URI='https://mlflow.rationai.cloud.e-infra.cz/'",
        "uv sync --frozen",
        "uv run python -m nuclei_graph experiment/modeling/training=nuclei_level/icaird_cervix",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
