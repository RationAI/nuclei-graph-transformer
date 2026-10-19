from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-nuclei-segmentation-gap-test",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",
    gpu="H100",
    public=False,
    script=[
        "git clone --branch tile-predictions https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.nuclei_segmentation_gap_test +data=sources/icaird_cervix +experiment=preprocessing/nuclei_segmentation_qc/icaird_cervix_gap_test",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
