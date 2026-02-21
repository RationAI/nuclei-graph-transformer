from kube_jobs import storage, submit_job


DATASET_NAME = "prostate_cancer"

submit_job(
    job_name="nuclei-graph-annotation-nuclei-labeling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run -m preprocessing.annotation_labels +data=datasets/raw/{DATASET_NAME}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
