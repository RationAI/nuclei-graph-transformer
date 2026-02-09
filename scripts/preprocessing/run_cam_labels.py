from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-cam-nuclei-labeling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="64Gi",
    public=False,
    script=[
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m preprocessing.cam_labels +experiment=preprocessing/cam_labels",
    ],
    storage=[storage.secure.DATA],
)
