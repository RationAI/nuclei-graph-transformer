from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-compute-efds",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="120Gi",
    public=False,
    script=[
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m preprocessing.efd_features",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
