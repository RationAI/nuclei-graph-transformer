# ruff: noqa: RUF002, RUF003  (Greek letters in text are intended)
r"""Which concepts separate carcinoma from benign epithelium (not from stroma)?

The k = 16 seed-1 SAE, for the encoder's codes κ (L1) and the NNLS refit κ̃ of
`xai.refit_magnitudes` separately; stored codes only, nothing is retrained.

Nucleus classes. From the Radboud label mask at level 0 (0 background, 1 stroma, 2 benign
epithelium, 3–5 Gleason pattern) by the thesis vertex rule
(`postprocessing.focus_geometry_radboud.vertex_classes`): carcinoma = the nucleus label
(≥ `overlap_threshold` of the polygon vertices in pattern ≥ 3; checked to equal the rule on
every nucleus), benign epithelium / stroma = ≥ `overlap_threshold` of the vertices with value
2 / 1, other = the rest (background, mixed, no mask). Epithelial nuclei = carcinoma ∪ benign
epithelium.

With φ_ij = κ_ij wᵀu_j, b₀ = wᵀb_D + β, ẑ_i = b₀ + Σ_j φ_ij (the SAE's reconstructed logit)
and z = class_head(h) (the model's logit), per split (validation, test):

1. Δ_j = mean φ_ij over carcinoma − mean φ_ij over benign epithelium; Σ_j Δ_j equals the
   difference of the class means of ẑ (b₀ cancels; checked). Share Δ_j / Σ_k |Δ_k|.
2. Δ_j over all nuclei: carcinoma − all non-carcinoma nuclei.
3. Activation frequency (κ_ij > 0) and mean κ_ij per class.
   1–2 get percentile 95 % CIs from a slide-level bootstrap (`bootstrap` resamples of the
   split's slides with replacement; class means = ratios of the resampled sums).
4. Test only: single-concept removal ẑ^(−j) = ẑ − φ_j + φ̄_j, φ̄_j the mean over all validation
   nuclei (no refit); AUROC / AUPRC within the epithelial nuclei and over all nuclei, and
   their drops from ẑ. (A constant shift does not change the ranking, so the choice of φ̄_j
   does not matter for these metrics.)
5. Test only: AUROC / AUPRC of ẑ and z within the epithelial nuclei and over all nuclei.

Outputs in `out_epithelium_dir` (no slide ids): `concepts.csv` (per variant, split and
concept), `class_means.csv` (ẑ and z per class), `baselines.csv` (5), `counts.csv`,
`checks.json`; the per-slide class cache
`classes/<split>/` (local only, named by slide id) and `classes_<split>.npy` (index row order).
"""

import json
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import (
    LinearHead,
    baseline,
    classification_metrics,
    concept_logit_weights,
)
from nuclei_graph.xai.sae import load_autoencoder
from postprocessing.focus_geometry_radboud import read_mask, vertex_classes
from xai.concept_masks import code_commit
from xai.concept_subsets import concept_names
from xai.evaluate_sae import head_logits
from xai.prepare_nuclei import ids_checksum


SPLITS = ("validation", "test")
VARIANTS = ("l1", "refit")
# disjoint groups: carcinoma = the nucleus label; the others by the mask class
OTHER, STROMA, BENIGN, CARCINOMA = 0, 1, 2, 3
GROUP_NAMES = {
    OTHER: "other",
    STROMA: "stroma",
    BENIGN: "benign_epithelium",
    CARCINOMA: "carcinoma",
}


# --- nucleus classes ---


def slide_classes(job: dict) -> dict:
    """Mask class of every nucleus of a slide (index order), cached as `<slide_id>.npy`."""
    out = Path(job["out"])
    if out.exists():
        return {"cached": True, "has_mask": bool(np.load(out.with_suffix(".mask.npy")))}
    nuclei = pd.read_parquet(job["nuclei_path"], columns=["id", "polygon"])
    nuclei = nuclei.sort_values("id", ignore_index=True)  # = get_nuclei (index) order
    ids = nuclei["id"].to_numpy()
    assert len(ids) == job["n_nuclei"] and ids_checksum(ids) == job["ids_crc32"], (
        "nuclei do not match the index"
    )
    mask_path = Path(job["mask_path"])
    has_mask = mask_path.exists()
    if has_mask:
        mask = read_mask(mask_path)
        with OpenSlide(job["slide_path"]) as slide:
            wx, wy = slide.dimensions
        _, cls = vertex_classes(
            nuclei["polygon"].tolist(),
            mask,
            (mask.shape[1] / wx, mask.shape[0] / wy),
            job["overlap_threshold"],
        )
    else:
        cls = np.zeros(len(ids), np.int8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".mask.npy"), np.array(has_mask))
    np.save(out, cls)
    return {"cached": False, "has_mask": has_mask}


def remap(path: str, path_remap: DictConfig | None) -> str:
    if path_remap is None:
        return path
    assert path.startswith(path_remap["from"]), path
    return path_remap["to"] + path[len(path_remap["from"]) :]


def split_classes(
    config: DictConfig, slides: pd.DataFrame, split: str, out: Path
) -> tuple[NDArray[np.int8], dict]:
    part = slides[slides["split"] == split].sort_values("offset")
    jobs = [
        {
            "out": str(out / "classes" / split / f"{row.slide_id}.npy"),
            "nuclei_path": remap(row.slide_nuclei_path, config.path_remap),
            "slide_path": row.slide_path,
            "mask_path": str(
                Path(config.label_masks_dir) / f"{row.slide_id}_mask.tiff"
            ),
            "n_nuclei": int(row.n_nuclei),
            "ids_crc32": int(row.ids_crc32),
            "overlap_threshold": float(config.overlap_threshold),
        }
        for row in part.itertuples(index=False)
    ]
    start = time.time()
    results = Parallel(n_jobs=config.n_jobs, verbose=5)(
        delayed(slide_classes)(job) for job in jobs
    )
    print(f"{split}: classes of {len(jobs)} slides in {time.time() - start:.0f}s")
    cls = np.concatenate([np.load(job["out"]) for job in jobs])
    has_mask = np.array([r["has_mask"] for r in results])
    checks = {
        "slides_without_mask": int((~has_mask).sum()),
        "carcinoma_slides_without_mask": int(
            (~has_mask & part["is_carcinoma"].to_numpy(bool)).sum()
        ),
    }
    return cls, checks


def stored_geometry_check(
    config: DictConfig, slides: pd.DataFrame, cls: NDArray[np.int8]
) -> dict:
    """Test classes against those of postprocessing.focus_geometry_radboud (exact)."""
    part = slides[slides["split"] == "test"].sort_values("offset")
    mismatch = compared = 0
    for row in part.itertuples(index=False):
        stored = pd.read_parquet(
            Path(config.test_geometry_dir) / "nuclei" / f"{row.slide_id}.parquet",
            columns=["cls"],
        )["cls"].to_numpy()
        mine = cls[row.offset : row.offset + row.n_nuclei]
        assert len(stored) == len(mine)
        mismatch += int((stored != mine).sum())
        compared += len(mine)
    return {"nuclei_compared": compared, "class_mismatches": mismatch}


def groups(labels: NDArray, cls: NDArray[np.int8]) -> tuple[NDArray[np.int8], dict]:
    group = np.where(
        labels == 1,
        CARCINOMA,
        np.where(cls == 2, BENIGN, np.where(cls == 1, STROMA, OTHER)),
    ).astype(np.int8)
    checks = {
        "label1_but_mask_class_below_3": int(((labels == 1) & (cls < 3)).sum()),
        "label0_but_mask_class_3_to_5": int(((labels == 0) & (cls >= 3)).sum()),
    }
    return group, checks


# --- class means and the slide bootstrap ---


def slide_group_sums(
    values: NDArray, slide: NDArray, group: NDArray, n_slides: int
) -> tuple[NDArray, NDArray]:
    """Per-slide, per-group sums (n_slides, 4, m) and counts (n_slides, 4)."""
    key = slide.astype(np.int64) * 4 + group
    size = n_slides * 4
    counts = np.bincount(key, minlength=size).reshape(n_slides, 4).astype(np.float64)
    sums = np.stack(
        [
            np.bincount(key, weights=values[:, c], minlength=size)
            for c in range(values.shape[1])
        ],
        -1,
    ).reshape(n_slides, 4, -1)
    return sums, counts


def contrasts(sums: NDArray, counts: NDArray) -> dict[str, NDArray]:
    """Class means and differences; `sums` (..., 4, m), `counts` (..., 4)."""
    non_carcinoma = [OTHER, STROMA, BENIGN]
    mean = {g: sums[..., g, :] / counts[..., g, None] for g in GROUP_NAMES}
    mean_non = (
        sums[..., non_carcinoma, :].sum(-2)
        / counts[..., non_carcinoma].sum(-1)[..., None]
    )
    return {
        "mean": mean,
        "mean_non_carcinoma": mean_non,
        "delta_epithelium": mean[CARCINOMA] - mean[BENIGN],
        "delta_all": mean[CARCINOMA] - mean_non,
    }


def bootstrap_counts(n_slides: int, n_boot: int, rng: np.random.Generator) -> NDArray:
    draws = rng.integers(0, n_slides, size=(n_boot, n_slides))
    return np.stack([np.bincount(d, minlength=n_slides) for d in draws]).astype(
        np.float64
    )


def percentile_ci(reps: NDArray) -> tuple[NDArray, NDArray]:
    return np.percentile(reps, 2.5, axis=0), np.percentile(reps, 97.5, axis=0)


# --- main ---


def compute(config: DictConfig) -> tuple[Path, dict]:
    out = Path(config.out_epithelium_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, meta = load_autoencoder(config.sae_model)
    assert (meta["k"], meta["seed"]) == (16, 1), meta
    decoder = model.decoder_weight.detach().double().numpy()
    b_d = model.decoder_bias.detach().double().numpy()
    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    weights = concept_logit_weights(decoder, head)
    b0 = baseline(b_d, head)
    k = len(weights)
    names, stable = concept_names(config.deck_key, list(config.stable_concepts))
    assert stable == sorted(config.stable_indices), (stable, config.stable_indices)
    letter_of = {j: names[j] for j in stable}
    order = sorted(stable, key=lambda j: letter_of[j]) + [
        j for j in range(k) if j not in stable
    ]

    slides = pd.read_parquet(config.slides)
    checks: dict = {"b0": b0, "concept_order": [names[j] for j in order]}
    data, group, slide_code = {}, {}, {}
    count_rows = []
    for split in SPLITS:
        data[split] = load_activations(config.activations_dir, split)
        labels = data[split].labels
        cls, class_checks = split_classes(config, slides, split, out)
        assert len(cls) == len(labels)
        np.save(out / f"classes_{split}.npy", cls)
        if split == "test" and config.test_geometry_dir is not None:
            class_checks["vs_focus_geometry"] = stored_geometry_check(
                config, slides, cls
            )
        group[split], label_checks = groups(labels, cls)
        checks[f"{split}/classes"] = class_checks | label_checks
        slide_ids = data[split].index["slide_id"].astype(str).to_numpy()
        starts = np.flatnonzero(np.r_[True, slide_ids[1:] != slide_ids[:-1]])
        assert len(np.unique(slide_ids)) == len(starts), "slides are not contiguous"
        slide_code[split] = np.repeat(
            np.arange(len(starts)), np.diff(np.r_[starts, len(slide_ids)])
        )
        n = np.bincount(group[split], minlength=4)
        pattern = np.bincount(cls[group[split] == CARCINOMA], minlength=6)
        slide_group = np.zeros((len(starts), 4), np.int64)
        np.add.at(slide_group, (slide_code[split], group[split]), 1)
        count_rows.append(
            {
                "split": split,
                "slides": len(starts),
                "nuclei": len(labels),
                **{f"n_{GROUP_NAMES[g]}": int(n[g]) for g in GROUP_NAMES},
                "n_carcinoma_g3": int(pattern[3]),
                "n_carcinoma_g4": int(pattern[4]),
                "n_carcinoma_g5": int(pattern[5]),
                "n_epithelial": int(n[CARCINOMA] + n[BENIGN]),
                "carcinoma_prevalence_epithelial": float(
                    n[CARCINOMA] / (n[CARCINOMA] + n[BENIGN])
                ),
                "carcinoma_prevalence_all": float(n[CARCINOMA] / len(labels)),
                "slides_with_carcinoma": int((slide_group[:, CARCINOMA] > 0).sum()),
                "slides_with_benign_epithelium": int(
                    (slide_group[:, BENIGN] > 0).sum()
                ),
            }
        )
        print(count_rows[-1], flush=True)
    counts = pd.DataFrame(count_rows)

    z = {s: head_logits(head, data[s]) for s in SPLITS}
    epithelial_test = np.isin(group["test"], (CARCINOMA, BENIGN))
    y_test = data["test"].labels
    model_metrics = {
        "epithelium": classification_metrics(
            y_test[epithelial_test], z["test"][epithelial_test]
        ),
        "all": classification_metrics(y_test, z["test"]),
    }
    subsets_checks = json.loads(Path(config.subsets_checks).read_text())
    checks["model_all_vs_concept_subsets"] = {
        "AUROC": [model_metrics["all"]["AUROC"], subsets_checks["model_test_AUROC"]],
        "AUPRC": [model_metrics["all"]["AUPRC"], subsets_checks["model_test_AUPRC"]],
    }

    concept_rows, mean_rows, baseline_rows = [], [], []
    for variant in VARIANTS:
        codes = {s: np.load(config.codes[variant][s]) for s in SPLITS}
        for s in SPLITS:
            assert codes[s].shape == (len(data[s]), k), (variant, s)
        mean_phi_val = weights * codes["validation"].mean(0, dtype=np.float64)
        z_hat_test = None
        for split_no, split in enumerate(SPLITS):
            kappa = np.asarray(codes[split], np.float64)
            phi = kappa * weights
            z_hat = b0 + phi.sum(1)
            if split == "test":
                z_hat_test, phi_test = z_hat, phi
            values = np.concatenate(
                [
                    phi,
                    z_hat[:, None],
                    z[split][:, None],
                    kappa,
                    (kappa > 0).astype(np.float64),
                ],
                1,
            )
            del kappa
            n_slides = int(slide_code[split].max()) + 1
            sums, cnt = slide_group_sums(
                values, slide_code[split], group[split], n_slides
            )
            del values
            point = contrasts(sums.sum(0), cnt.sum(0))
            rng = np.random.default_rng([config.seed, split_no])
            w = bootstrap_counts(n_slides, config.bootstrap, rng)
            reps = contrasts(np.einsum("bs,sgm->bgm", w, sums), w @ cnt)

            d_epi, d_all = point["delta_epithelium"], point["delta_all"]
            checks[f"{variant}/{split}/sum_delta_minus_delta_zhat_epithelium"] = float(
                d_epi[:k].sum() - d_epi[k]
            )
            checks[f"{variant}/{split}/sum_delta_minus_delta_zhat_all"] = float(
                d_all[:k].sum() - d_all[k]
            )
            share = d_epi[:k] / np.abs(d_epi[:k]).sum()
            share_reps = reps["delta_epithelium"][:, :k] / np.abs(
                reps["delta_epithelium"][:, :k]
            ).sum(1, keepdims=True)
            epi_lo, epi_hi = percentile_ci(reps["delta_epithelium"])
            all_lo, all_hi = percentile_ci(reps["delta_all"])
            share_lo, share_hi = percentile_ci(share_reps)
            for name, col in (("zhat", k), ("z", k + 1)):
                mean_rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "score": name,
                        "delta_epithelium": float(d_epi[col]),
                        "delta_epithelium_ci_low": float(epi_lo[col]),
                        "delta_epithelium_ci_high": float(epi_hi[col]),
                        "delta_all": float(d_all[col]),
                        "delta_all_ci_low": float(all_lo[col]),
                        "delta_all_ci_high": float(all_hi[col]),
                        "mean_carcinoma": float(point["mean"][CARCINOMA][col]),
                        "mean_benign_epithelium": float(point["mean"][BENIGN][col]),
                        "mean_stroma": float(point["mean"][STROMA][col]),
                        "mean_non_carcinoma": float(point["mean_non_carcinoma"][col]),
                    }
                )
            mean = point["mean"]
            for rank, j in enumerate(order, 1):
                kj, aj = k + 2 + j, 2 * k + 2 + j
                row = {
                    "variant": variant,
                    "split": split,
                    "order": rank,
                    "concept": names[j],
                    "sae_index": j,
                    "stable": j in stable,
                    "logit_weight": float(weights[j]),
                    "delta_epithelium": float(d_epi[j]),
                    "delta_epithelium_ci_low": float(epi_lo[j]),
                    "delta_epithelium_ci_high": float(epi_hi[j]),
                    "share_epithelium": float(share[j]),
                    "share_epithelium_ci_low": float(share_lo[j]),
                    "share_epithelium_ci_high": float(share_hi[j]),
                    "delta_all": float(d_all[j]),
                    "delta_all_ci_low": float(all_lo[j]),
                    "delta_all_ci_high": float(all_hi[j]),
                    "mean_phi_carcinoma": float(mean[CARCINOMA][j]),
                    "mean_phi_benign_epithelium": float(mean[BENIGN][j]),
                    "mean_phi_stroma": float(mean[STROMA][j]),
                    "mean_phi_non_carcinoma": float(point["mean_non_carcinoma"][j]),
                    "active_carcinoma": float(mean[CARCINOMA][aj]),
                    "active_benign_epithelium": float(mean[BENIGN][aj]),
                    "active_stroma": float(mean[STROMA][aj]),
                    "mean_kappa_carcinoma": float(mean[CARCINOMA][kj]),
                    "mean_kappa_benign_epithelium": float(mean[BENIGN][kj]),
                    "mean_kappa_stroma": float(mean[STROMA][kj]),
                }
                concept_rows.append(row)
            del phi, sums, reps

        # 4–5: removal and baselines on test
        assert z_hat_test is not None
        full = {
            "epithelium": classification_metrics(
                y_test[epithelial_test], z_hat_test[epithelial_test]
            ),
            "all": classification_metrics(y_test, z_hat_test),
        }
        checks[f"{variant}/zhat_all_test"] = full["all"]
        for subset, mask in (("epithelium", epithelial_test), ("all", slice(None))):
            for name, metrics in (("zhat", full[subset]), ("z", model_metrics[subset])):
                baseline_rows.append(
                    {
                        "variant": variant,
                        "split": "test",
                        "score": name,
                        "nuclei": subset,
                        "n": len(y_test[mask]),
                        "prevalence": float(y_test[mask].mean()),
                        **metrics,
                    }
                )
        for j in order:
            removed = z_hat_test - phi_test[:, j] + mean_phi_val[j]
            row = next(
                r
                for r in concept_rows
                if r["variant"] == variant
                and r["split"] == "test"
                and r["sae_index"] == j
            )
            row["mean_phi_validation"] = float(mean_phi_val[j])
            for subset, mask in (("epithelium", epithelial_test), ("all", slice(None))):
                m = classification_metrics(y_test[mask], removed[mask])
                row[f"AUROC_removed_{subset}"] = m["AUROC"]
                row[f"AUPRC_removed_{subset}"] = m["AUPRC"]
                row[f"dAUROC_{subset}"] = full[subset]["AUROC"] - m["AUROC"]
                row[f"dAUPRC_{subset}"] = full[subset]["AUPRC"] - m["AUPRC"]
            print(
                variant,
                names[j],
                {c: round(row[c], 4) for c in ("dAUPRC_epithelium", "dAUPRC_all")},
                flush=True,
            )
            del removed
        del phi_test, z_hat_test, codes

    concepts = pd.DataFrame(concept_rows)
    baselines = pd.DataFrame(baseline_rows)
    concepts.to_csv(out / "concepts.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(out / "class_means.csv", index=False)
    baselines.to_csv(out / "baselines.csv", index=False)
    counts.to_csv(out / "counts.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return out, checks


def metric_name(concept: str) -> str:
    return f"u{concept[1:]}" if concept.startswith("#") else concept


@with_cli_args(["+xai=epithelium_separation"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    commit, dirty = code_commit("xai", "nuclei_graph", "configs", "postprocessing")
    client.set_tag(
        logger.run_id, "xai.code_commit", f"{commit}{' (dirty)' if dirty else ''}"
    )
    for key, value in config.links.items():
        if value is not None:
            client.set_tag(logger.run_id, f"xai.{key}", value)
    out, _ = compute(config)
    for row in pd.read_csv(out / "counts.csv").to_dict("records"):
        logger.log_metrics(
            {f"{row['split']}/{k}": float(v) for k, v in row.items() if k != "split"}
        )
    for row in pd.read_csv(out / "baselines.csv").to_dict("records"):
        logger.log_metrics(
            {
                f"test/{row['variant']}/{row['nuclei']}/{row['score']}/{m}": row[m]
                for m in ("AUROC", "AUPRC")
            }
        )
    concepts = pd.read_csv(out / "concepts.csv")
    for row in concepts.to_dict("records"):
        prefix = f"{row['split']}/{row['variant']}/{metric_name(row['concept'])}"
        keys = ["delta_epithelium", "share_epithelium", "delta_all"]
        if row["split"] == "test":
            keys += ["dAUPRC_epithelium", "dAUPRC_all"]
        logger.log_metrics({f"{prefix}/{key}": float(row[key]) for key in keys})
    for name in (
        "concepts.csv",
        "class_means.csv",
        "baselines.csv",
        "counts.csv",
        "checks.json",
    ):
        logger.log_artifact(str(out / name))
    for source in (
        __file__,
        "configs/xai/epithelium_separation.yaml",
        "postprocessing/focus_geometry_radboud.py",
    ):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")


if __name__ == "__main__":
    main()
