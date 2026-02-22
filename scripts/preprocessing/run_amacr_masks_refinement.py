from kube_jobs import storage, submit_job


DATASET_NAME = "prostate_cancer_amacr"

submit_job(
    job_name="nuclei-graph-amacr-masks-refinement",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=16,  # TODO
    memory="200Gi",  # TODO
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        f"uv run -m preprocessing.amacr_masks_refinement +experiment=preprocessing/amacr_masks_refinement/{DATASET_NAME}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
