from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-data-exploration",
    username="xrusnack",  # ...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "git checkout refactor/configs",
        "uv sync --frozen",
        "uv run python -m exploration.save_metadataset",
    ],
    storage=[storage.secure.DATA],
)
