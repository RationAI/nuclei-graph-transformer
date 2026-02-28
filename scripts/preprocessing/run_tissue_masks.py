from kube_jobs import storage, submit_job


EXPERIMENT_NAME = ...  # "raw" or "edge_eroded"
DATASET_NAME = "prostate_cancer_amacr"

submit_job(
    job_name="nuclei-graph-tissue-masks",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,  # TODO
    memory="64Gi",  # TODO
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run -m preprocessing.tissue_masks +experiment=preprocessing/tissue_masks/{EXPERIMENT_NAME}/{DATASET_NAME}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
