from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-tile-thresholds-prostate-cancer-mmci-tl",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="64Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m postprocessing.tile.thresholds +experiment=postprocessing/tile/thresholds/prostate_cancer_mmci_tl",
    ],
    storage=[storage.secure.DATA],
)
