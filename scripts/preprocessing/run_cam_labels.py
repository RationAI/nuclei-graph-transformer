from kube_jobs import storage, submit_job


EXPERIMENT_NAME = ...  # "cam_labels_annot_restricted_thr" or "cam_labels_default_thr"

submit_job(
    job_name="nuclei-graph-cam-nuclei-labeling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="16Gi",
    public=False,
    script=[
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run python -m preprocessing.cam_labels +experiment=preprocessing/{EXPERIMENT_NAME}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
