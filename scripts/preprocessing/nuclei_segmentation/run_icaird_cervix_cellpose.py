import sys

from kube_jobs import storage, submit_job


# One job per shard; each segments every SHARDS-th slide (see `shard_count` in
# configs/preprocessing/nuclei_segmentation.yaml). Submits the shards given as
# arguments (e.g. `... run_icaird_cervix_cellpose.py 1`), or all of them.
SHARDS = 3

for shard in [int(arg) for arg in sys.argv[1:]] or range(SHARDS):
    submit_job(
        job_name=f"nuclei-graph-icaird-cervix-nuclei-segmentation-cellpose-{shard}",
        username=...,
        image="cerit.io/rationai/base:2.0.6",
        cpu=40,
        memory="100Gi",
        gpu="A100",
        public=True,
        script=[
            "git clone https://github.com/RationAI/nuclei-graph-transformer.git workdir",
            "cd workdir",
            "uv sync --frozen",
            (
                "uv run -m preprocessing.nuclei_segmentation +data=sources/icaird_cervix "
                "+experiment=preprocessing/nuclei_segmentation/icaird_cervix_cellpose "
                f"shard_count={SHARDS} shard_index={shard}"
            ),
        ],
        storage=[storage.public.DATA, storage.public.PROJECTS],
    )
