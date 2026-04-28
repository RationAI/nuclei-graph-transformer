from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-merge-cam-masks",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.merge_cam_masks +experiment=preprocessing/cam_masks/...",
    ],
    storage=[storage.secure.DATA],
)
