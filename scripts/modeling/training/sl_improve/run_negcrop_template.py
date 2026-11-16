from kube_jobs import storage, submit_job


# Slide-label screening on MMCI (study sl_improve), seed 1: negative sub-crops (p = 0.3) on the parent set in negcrop_template.yaml (now attention + λ = 0.5); when the parent changes, re-pin the checkout to the commit with the new parent.
# Image and resources as the SL MMCI jobs of runs d9cd4264 / 425c23d5. The checkout is pinned
# to the commit with the sl_improve configs (200 epochs) and the EMA fix, so every screening run
# trains the same code.
submit_job(
    job_name="nuclei-graph-sli-negcrop",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=20,  # 16 train loader workers (data.num_workers) + the main process + headroom
    memory="128Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone --branch xai-sae https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "git checkout 2e258eeaffa229c16cba8bf548ced5cca778e50c",
        # experiment 97 and the MMCI slide tables live on the sensitive MLflow (no credentials)
        "export MLFLOW_TRACKING_URI='http://mlflow.rationai-mlflow:5000'",
        "uv sync --frozen",
        "uv run -m nuclei_graph experiment/modeling/training=graph_level/sl_improve/negcrop_template seed=1",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
