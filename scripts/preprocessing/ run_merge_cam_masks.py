from kube_jobs import storage, submit_job


DATASET_NAME = "prostate_cancer"

submit_job(
    job_name="nuclei-graph-merge-cam-masks",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run python -m preprocessing.merge_cam_masks +experiment=preprocessing/cam_masks/{DATASET_NAME}",
    ],
    storage=[storage.secure.DATA],
)
