"""Per-nucleus probabilities of several checkpoints, with the model code of a given checkout.

Run as a script by ``postprocessing.slide_error_analysis_radboud`` in a subprocess whose
``PYTHONPATH`` starts with a git worktree of the evaluation runs' commit, so that the
``nuclei_graph`` imported here is that commit's (asserted with ``--code-root``). It mirrors
the evaluation runs' prediction (``configs/experiment/modeling/prediction``): the crop
``PredictionDataset`` on whole slides, ``GraphCollator(predict=True)``, sigmoid of the nucleus
logits, TF32 matmuls (``torch.set_float32_matmul_precision("high")`` as ``nuclei_graph/__main__``).

Inputs (``--job``, JSON): ``slides`` (parquet with the dataset's metadata columns), ``runs``
(list of {"checkpoint", "config"}: a checkpoint file and the training run's resolved config,
YAML), ``outputs`` (one parquet per run) and ``device``. Each output holds (slide, key, p) of
every nucleus, slide = row in ``slides``, key = the 64-bit prefix of the nucleus id, sorted; with
``"logits": true`` in the job also the logit, and for models with a self-evidence head (study
small_foci, B1) its two components ``logit_context`` and ``logit_self`` (logit = their sum).

Representations (optional, ``h_sample`` and ``h_outputs`` in the job): ``h_sample`` is a parquet
of (slide, key) rows sorted like the outputs; for each run, the output of ``net.final_norm``
(the input of ``class_head``) at those nuclei is written to its ``h_outputs`` path (.npy,
float16, rows in ``h_sample`` order). class_head(h) is checked against the logits on every
slide before the cast; the largest difference per run goes to ``<h_output>.json``.
Only standard-library and third-party modules are imported at module level.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path


DATA_KEYS = ("block_size", "k", "context")
DATASET_KEYS = ("embedding_mode", "alpha", "efd_order", "patch_size")


def nucleus_keys(ids):  # as postprocessing.bootstrap_sources.nucleus_keys
    import numpy as np

    assert (ids.str.len() == 64).all(), "unexpected nucleus id format"
    prefix = "".join(ids.str.slice(0, 16))
    return np.frombuffer(bytes.fromhex(prefix), dtype=">u8").astype(np.uint64)


def captured_dim(model) -> int:
    return model.net.class_head.in_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--code-root", required=True)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from hydra.utils import instantiate
    from lightning.fabric.utilities.apply_func import move_data_to_device
    from tqdm import tqdm

    import nuclei_graph
    from nuclei_graph.data.datamodules.collator import GraphCollator
    from nuclei_graph.data.datasets.crop.prediction import PredictionDataset

    root = Path(args.code_root).resolve()
    assert Path(nuclei_graph.__file__).resolve().is_relative_to(root), (
        f"nuclei_graph imported from {nuclei_graph.__file__}, not {root}"
    )
    job = json.loads(Path(args.job).read_text())
    device = job["device"]
    torch.set_float32_matmul_precision("high")

    cfgs = [yaml.safe_load(Path(r["config"]).read_text()) for r in job["runs"]]
    d = cfgs[0]["data"]
    for c in cfgs[1:]:
        assert {k: c["data"][k] for k in DATA_KEYS} == {k: d[k] for k in DATA_KEYS}
        assert {k: c["data"]["dataset"].get(k) for k in DATASET_KEYS} == {
            k: d["dataset"].get(k) for k in DATASET_KEYS
        }

    models = []
    for r, cfg in zip(job["runs"], cfgs, strict=True):
        model = instantiate(cfg["model"])
        state = torch.load(r["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        models.append(model.eval().to(device))

    slides = pd.read_parquet(job["slides"]).reset_index(drop=True)
    ds_kwargs = {
        "embedding_mode": d["dataset"]["embedding_mode"],
        "alpha": d["dataset"].get("alpha"),
        "efd_order": d["dataset"]["efd_order"],
        "patch_size": d["dataset"].get("patch_size"),
        "patches_path": d["dataset"].get("patches_path"),
    }
    accepted = inspect.signature(PredictionDataset).parameters
    ds = PredictionDataset(
        slides,
        **{k: v for k, v in ds_kwargs.items() if k in accepted and v is not None},
    )
    collate_kwargs = {"block_size": d["block_size"], "k": d["k"], "predict": True}
    if "context" in inspect.signature(GraphCollator).parameters:
        collate_kwargs["context"] = d["context"]
    else:  # commits before the context option build the k-NN mask only
        assert d.get("context", "knn") == "knn", "this commit's collator is k-NN only"
    collate = GraphCollator(**collate_kwargs)

    with_logits = bool(job.get("logits", False))
    frames = [[] for _ in models]

    h_sample = job.get("h_sample")
    captured: dict[int, object] = {}
    h_parts: list[list] = [[] for _ in models]
    h_diff = [0.0 for _ in models]
    if h_sample is not None:
        h_sample = pd.read_parquet(h_sample)
        sample_bounds = np.searchsorted(
            h_sample["slide"].to_numpy(), np.arange(len(slides) + 1)
        )
        for m, model in enumerate(models):
            model.net.final_norm.register_forward_hook(
                lambda _mod, _inp, out, m=m: captured.__setitem__(m, out.detach())
            )
    for s in tqdm(range(len(ds)), desc=Path(job["outputs"][0]).stem):
        batch = collate([ds[s]])
        inputs = move_data_to_device(
            {k: v for k, v in batch.items() if k != "metadata"}, device
        )
        ids = pd.Series(np.asarray(batch["metadata"]["nuclei_ids"][0]))
        assert batch["metadata"]["slide_id"][0] == slides.loc[s, "slide_id"]
        keys = nucleus_keys(ids)
        for m, model in enumerate(models):
            with torch.inference_mode():
                out = model(inputs)
            logits = out["nuclei"].squeeze(-1)
            p = torch.sigmoid(logits).float().cpu().numpy().ravel()
            assert len(p) == len(ids)
            if h_sample is not None:
                h = captured.pop(m).reshape(-1, captured_dim(model))[: len(ids)]
                with torch.inference_mode():
                    z = model.net.class_head(h).squeeze(-1)
                h_diff[m] = max(
                    h_diff[m], float((z.float() - logits.float().ravel()).abs().max())
                )
                lo, hi = sample_bounds[s], sample_bounds[s + 1]
                rows = pd.Index(keys).get_indexer(h_sample["key"].to_numpy()[lo:hi])
                assert (rows >= 0).all(), "sampled nuclei missing from the slide"
                h_parts[m].append(
                    h[torch.as_tensor(rows, device=h.device)].half().cpu().numpy()
                )
            cols = {"slide": np.full(len(p), s, np.int16), "key": keys, "p": p}
            if with_logits:
                cols["logit"] = logits.float().cpu().numpy().ravel()
                for name in ("context", "self"):
                    if f"nuclei_{name}" in out:
                        cols[f"logit_{name}"] = (
                            out[f"nuclei_{name}"]
                            .squeeze(-1)
                            .float()
                            .cpu()
                            .numpy()
                            .ravel()
                        )
            frames[m].append(pd.DataFrame(cols))
    if h_sample is not None:
        for m, path in enumerate(job["h_outputs"]):
            h_all = np.concatenate(h_parts[m])
            assert len(h_all) == len(h_sample)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.save(path, h_all)
            Path(path).with_suffix(".json").write_text(
                json.dumps({"rows": len(h_all), "class_head_max_abs_diff": h_diff[m]})
            )
    for f, path in zip(frames, job["outputs"], strict=True):
        df = pd.concat(f, ignore_index=True).sort_values(
            ["slide", "key"], ignore_index=True
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    print(f"nuclei_graph from {Path(nuclei_graph.__file__).parent}", file=sys.stderr)


if __name__ == "__main__":
    main()
