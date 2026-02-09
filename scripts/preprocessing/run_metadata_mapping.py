from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-metadata-mapping",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m preprocessing.metadata_mapping +experiment=preprocessing/metadata_mapping",
    ],
    storage=[storage.secure.DATA],
)
