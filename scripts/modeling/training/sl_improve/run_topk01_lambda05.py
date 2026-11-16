from kube_jobs import storage, submit_job


# Slide-label screening on MMCI (study sl_improve), seed 1: top-k pooling (1 %) + λ = 0.5.
# Image and resources as the SL MMCI jobs of runs d9cd4264 / 425c23d5. The checkout is pinned
# to the commit that added the sl_improve configs, so every screening run trains the same code.
submit_job(
    job_name="nuclei-graph-sli-topk01-lam05",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=20,  # 16 train loader workers (data.num_workers) + the main process + headroom
    memory="128Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone --branch xai-sae https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "git checkout 0afe9f4abf31ddace9904b57efb10a20ef522115",
        # experiment 97 and the MMCI slide tables live on the sensitive MLflow (no credentials)
        "export MLFLOW_TRACKING_URI='http://mlflow.rationai-mlflow:5000'",
        "uv sync --frozen",
        "uv run -m nuclei_graph experiment/modeling/training=graph_level/sl_improve/topk01_lambda05 seed=1",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
