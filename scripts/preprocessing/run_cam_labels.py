from kube_jobs import storage, submit_job


EXPERIMENT_NAME = ...  # "annot_restricted_thr" or "default_thr"
DATASET_NAME = "prostate_cancer"

submit_job(
    job_name="nuclei-graph-cam-nuclei-labeling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run -m preprocessing.cam_labels +experiment=preprocessing/cam_labels/{EXPERIMENT_NAME}/{DATASET_NAME}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
