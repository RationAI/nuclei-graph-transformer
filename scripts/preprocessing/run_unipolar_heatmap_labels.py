from kube_jobs import storage, submit_job


submit_job(
    job_name="nuclei-graph-heatmap-nuclei-labeling",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/nuclei-graph-transformer.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.unipolar_heatmap_labels +experiment=preprocessing/unipolar_heatmap_labels/...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
