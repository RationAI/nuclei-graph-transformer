import json
import zlib
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from mlflow.tracking import MlflowClient
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from nuclei_graph.nuclei_graph_typing import Batch, Outputs


class ActivationsCallback(Callback):
    """Captures nucleus representations h_i (input of `class_head`) during prediction.

    A forward hook on `net.final_norm` (and optionally on every `net.layers[l]`)
    records the activations of all nuclei in the whole-slide prediction pipeline.
    `slides_path` (from `xai.prepare_nuclei`) gives every slide's `split`, `offset`,
    `n_nuclei` and `ids_crc32`; per split, the callback writes memory-mapped
    `h.npy` (n, d), `logits.npy` (n,) and optional `layers.npy` (n, L, d) whose rows
    follow `<split>/index.parquet` (slides contiguous, nuclei in id order).
    `head.npz` holds the weights `w` (d,) and bias `beta` of the binary `class_head`.

    `class_head(h)` is checked against the model logits before `h` is cast to
    `h_dtype`; the effect of the cast on the logits is reported separately.

    Resumable: after a slide's rows are flushed, `<split>/done/<slide_id>.json` records
    it (with its checks). With `resume=True`, existing arrays are reopened in place and
    finished slides are removed from the prediction dataset before inference.
    """

    def __init__(
        self,
        slides_path: str,
        output_dir: str,
        h_dtype: str = "float16",
        logits_dtype: str = "float32",
        save_layers: bool = False,
        resume: bool = True,
        atol: float = 1e-4,
        mlflow_artifact_path: str = "activations",
        mlflow_run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.slides_path = slides_path
        self.output_dir = Path(output_dir)
        self.h_dtype = np.dtype(h_dtype)
        self.logits_dtype = np.dtype(logits_dtype)
        self.save_layers = save_layers
        self.resume = resume
        self.atol = atol
        self.mlflow_artifact_path = mlflow_artifact_path
        self.mlflow_run_id = mlflow_run_id

        self.handles: list[RemovableHandle] = []
        self.captured: dict[str, Tensor] = {}
        self.slides = pd.DataFrame()
        self.arrays: dict[str, dict[str, np.memmap]] = {}

    def _hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            self.captured[name] = output.detach()

        return hook

    def _marker(self, split: str, slide_id: str) -> Path:
        return self.output_dir / split / "done" / f"{slide_id}.json"

    def _open(
        self, path: Path, dtype: np.dtype, shape: tuple[int, ...]
    ) -> tuple[np.memmap, bool]:
        """Reopens a matching existing array when resuming, else creates it."""
        if self.resume and path.exists():
            array = np.load(path, mmap_mode="r+")
            if array.dtype == dtype and array.shape == shape:
                return array, True
        return np.lib.format.open_memmap(path, "w+", dtype, shape), False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "predict":
            return
        net = pl_module.net
        assert net.class_head.out_features == 1, "Expected a binary class_head."
        dim, num_layers = net.class_head.in_features, len(net.layers)

        self.slides = pd.read_parquet(self.slides_path).set_index("slide_id")
        for split, n in self.slides.groupby("split")["n_nuclei"].sum().items():
            split, n = str(split), int(n)
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            shapes = {
                "h": (self.h_dtype, (n, dim)),
                "logits": (self.logits_dtype, (n,)),
            }
            if self.save_layers:
                shapes["layers"] = (self.h_dtype, (n, num_layers, dim))
            opened = {
                key: self._open(split_dir / f"{key}.npy", dtype, shape)
                for key, (dtype, shape) in shapes.items()
            }
            self.arrays[split] = {key: array for key, (array, _) in opened.items()}

            done_dir = split_dir / "done"
            if not all(reused for _, reused in opened.values()):
                for marker in done_dir.glob("*.json"):  # markers of replaced arrays
                    marker.unlink()
            done_dir.mkdir(exist_ok=True)

        # skip finished slides
        done = {
            slide_id
            for slide_id, split in self.slides["split"].items()
            if self._marker(split, slide_id).exists()
        }
        if done:
            dataset = trainer.datamodule.predict_dataset
            dataset.metadata = dataset.metadata[
                ~dataset.metadata["slide_id"].isin(done)
            ].reset_index(drop=True)
            print(
                f"[INFO] Resuming: {len(done)} slides done, {len(dataset)} to predict"
            )

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        net = pl_module.net

        self.handles = [net.final_norm.register_forward_hook(self._hook("final"))]
        if self.save_layers:
            self.handles += [
                layer.register_forward_hook(self._hook(f"layer_{i}"))
                for i, layer in enumerate(net.layers)
            ]

        np.savez(
            self.output_dir / "head.npz",
            w=net.class_head.weight.detach().cpu().numpy()[0],
            beta=net.class_head.bias.detach().cpu().numpy()[0],
        )

    def _captured(self, name: str, seq_len: int, order: Tensor) -> Tensor:
        # (1, padded_len, d) -> (n, d), in index order
        return self.captured[name].squeeze(0)[:seq_len][order]

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Outputs,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        metadata = batch["metadata"]
        assert metadata is not None, "Metadata is required to save activations."
        assert len(metadata["slide_id"]) == 1, "Batch size must be 1."

        slide_id = metadata["slide_id"][0]
        if slide_id not in self.slides.index:
            return
        slide = self.slides.loc[slide_id]

        # nuclei_ids follow the collator's spatial order, as do the model outputs
        nuclei_ids = np.asarray(metadata["nuclei_ids"][0])
        order = np.argsort(nuclei_ids, kind="stable")
        assert len(order) == slide.n_nuclei, f"Nuclei count mismatch: {slide_id}"
        checksum = zlib.crc32("\n".join(map(str, nuclei_ids[order])).encode())
        assert checksum == slide.ids_crc32, f"Nuclei ids mismatch: {slide_id}"

        seq_len = int(batch["seq_lens"].sum().item())
        order_t = torch.as_tensor(order, device=self.captured["final"].device)
        h = self._captured("final", seq_len, order_t)
        logits = outputs["nuclei"][order_t].squeeze(-1)

        head = pl_module.net.class_head
        diff = (head(h).squeeze(-1) - logits).abs().max().item()
        h_cast = h.to(getattr(torch, self.h_dtype.name))
        diff_cast = (head(h_cast.to(h.dtype)).squeeze(-1) - logits).abs().max().item()

        rows = slice(int(slide.offset), int(slide.offset + slide.n_nuclei))
        arrays = self.arrays[slide.split]
        arrays["h"][rows] = h_cast.cpu().numpy()
        arrays["logits"][rows] = logits.cpu().numpy().astype(self.logits_dtype)
        if self.save_layers:
            arrays["layers"][rows] = (
                torch.stack(
                    [
                        self._captured(f"layer_{i}", seq_len, order_t)
                        for i in range(len(pl_module.net.layers))
                    ],
                    dim=1,
                )
                .to(h_cast.dtype)
                .cpu()
                .numpy()
            )
        for array in arrays.values():
            array.flush()
        self._marker(slide.split, slide_id).write_text(
            json.dumps(
                {
                    "n_nuclei": int(slide.n_nuclei),
                    "class_head_max_abs_diff": diff,
                    f"class_head_max_abs_diff_{self.h_dtype.name}": diff_cast,
                }
            )
        )

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captured.clear()

        for arrays in self.arrays.values():
            for array in arrays.values():
                array.flush()
        self.arrays.clear()

        checks, missing = [], []
        for slide_id, split in self.slides["split"].items():
            marker = self._marker(split, slide_id)
            if marker.exists():
                checks.append(json.loads(marker.read_text()))
            else:
                missing.append(slide_id)
        cast_key = f"class_head_max_abs_diff_{self.h_dtype.name}"
        max_abs_diff = max((c["class_head_max_abs_diff"] for c in checks), default=0.0)
        summary = {
            "class_head_max_abs_diff": max_abs_diff,
            cast_key: max((c[cast_key] for c in checks), default=0.0),
            "slides_written": len(checks),
            "slides_missing": len(missing),
        }
        (self.output_dir / "activations_summary.json").write_text(
            json.dumps({**summary, "missing": missing}, indent=2)
        )
        self._log(summary)

        # checked after saving, so that a failed check does not discard the run
        assert not missing, f"{len(missing)} slides were not predicted: {missing[:5]}"
        assert max_abs_diff < self.atol, (
            f"class_head(h) does not reproduce the logits: {max_abs_diff:.2e}"
        )

    def _log(self, summary: dict[str, float]) -> None:
        run_id = self.mlflow_run_id
        if run_id is None and (active_run := mlflow.active_run()) is not None:
            run_id = active_run.info.run_id
        if run_id is None:
            return

        client = MlflowClient()
        for key, value in summary.items():
            client.log_metric(run_id, f"activations/{key}", value)
        # the activation arrays are large, they stay in `output_dir`
        for name in ("head.npz", "activations_summary.json"):
            client.log_artifact(
                run_id, str(self.output_dir / name), self.mlflow_artifact_path
            )
