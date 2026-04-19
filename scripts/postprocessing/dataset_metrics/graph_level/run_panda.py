from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-dataset-level-graph-metrics-panda",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=2,
    memory="8Gi",
    public=True,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m postprocessing.dataset_metrics.graph_level +experiment=postprocessing/dataset_metrics/graph_level/...",
    ],
    storage=[storage.public.DATA],
)
