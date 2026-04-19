from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-per-slide-metrics-prostate-cancer-mmci-tl",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=2,
    memory="8Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m postprocessing.slide_metrics +experiment=postprocessing/slide_metrics/prostate_cancer_mmci_tl",
    ],
    storage=[storage.secure.DATA],
)
