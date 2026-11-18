r"""Decision flips of the concept reconstruction (tab:res-faithfulness), PANDA Radboud test nuclei.

Evaluation only, from stored values: the XAI arrays of model run c8f26853 (read-only mount),
the stored codes of the k = 16 l1 SAE seed 1 (κ, SAE training run 4cf1db8f), its NNLS-refitted
codes (κ̃, refit run 6532785d) and the PCA baseline with 16 components (`pca_k16.pt`, encoded
as in `faithfulness_recompute.py` of the table). Thesis notation: z = wᵀh + β (the head on the
stored h, as the table and `xai.refit_magnitudes`), ẑ = wᵀD(κ) + β.

A nucleus flips at threshold t if it lies on different sides of t under z and ẑ:
(z ≥ t) ≠ (ẑ ≥ t), for t = 0 as `xai.evaluate_topk_sae.flip_rates` (z > 0 vs ẑ > 0, i.e.
probability 0.5; ties do not occur, counted) and for the Youden threshold of the thesis
(the seed-mean validation probability of the three seeds, `slide_error_analysis/thresholds.json`,
p ≥ 0.4021). Reported overall, among nuclei near the threshold (|z - t| < 1 and < 0.5 logit),
and per label (carcinoma / benign), with the direction of the flips. Checks: the same with the
stored model logits in place of the head on h; the Youden threshold of this seed alone on its
validation nuclei; the stored flip rates of the refit run (metrics.csv).

Usage: python -m xai.faithfulness_flips <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import LinearHead
from nuclei_graph.xai.sae import encode, load_autoencoder
from postprocessing.patient_score_quantiles_mmci import youden_threshold


torch.set_float32_matmul_precision(
    "high"
)  # as configs/xai.yaml and the table's recompute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(
    "/mnt/public_projects/nuclei_based_wsi_analysis/xai/c8f268536be8416f9fe395eea95c458e"
)
ACT, SAE, CON = ROOT / "activations", ROOT / "sae", ROOT / "concepts/k16_seed1"
THRESHOLDS = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/"
    "slide_error_analysis/thresholds.json"
)
BANDS = (None, 1.0, 0.5)  # all nuclei, |z - t| < 1, |z - t| < 0.5
CH = 262144


def logit(p: float) -> float:
    return float(np.log(p) - np.log1p(-p))


def flip_rows(
    z: np.ndarray, z_hat: np.ndarray, y: np.ndarray, t: float, strict: bool
) -> list[dict]:
    above, above_hat = (z > t, z_hat > t) if strict else (z >= t, z_hat >= t)
    flip = above != above_hat
    rows = []
    for band in BANDS:
        near = np.ones_like(flip) if band is None else np.abs(z - t) < band
        for label, sel in (
            ("all", near),
            ("carcinoma", near & (y == 1)),
            ("benign", near & (y == 0)),
        ):
            n, k = int(sel.sum()), int((flip & sel).sum())
            down = int((flip & sel & above).sum())  # z on the carcinoma side, ẑ not
            rows.append(
                {
                    "band": "all" if band is None else f"|z-t|<{band:g}",
                    "label": label,
                    "n": n,
                    "flips": k,
                    "flip_rate": k / n if n else np.nan,
                    "flips_to_benign": down,
                    "flips_to_carcinoma": k - down,
                }
            )
    return rows


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    head = LinearHead.from_npz(str(ACT / "head.npz"))
    data = {s: load_activations(ACT, s) for s in ("validation", "test")}
    z_all = {
        s: np.concatenate([head(a.z(r)) for r in a.chunks(CH)]) for s, a in data.items()
    }
    test = data["test"]
    z, z_stored, y = z_all["test"], np.asarray(test.logits, np.float64), test.labels
    assert set(np.unique(y)) <= {0, 1}

    youden = json.loads(THRESHOLDS.read_text())["thresholds"]["youden"]
    p_y = float(youden["threshold"])
    own_t, own_j = youden_threshold(data["validation"].labels, z_all["validation"])
    thresholds = {
        "zero": {"logit": 0.0, "probability": 0.5, "strict": True},
        "youden": {"logit": logit(p_y), "probability": p_y, "strict": False},
    }
    info = {
        "n_test": len(z),
        "n_carcinoma": int(y.sum()),
        "youden_thesis": {
            "probability": p_y,
            "logit": logit(p_y),
            "J": youden["J"],
            "source": "seed-mean validation probability of the 3 seeds (thresholds.json)",
            "validation": youden["validation"],
        },
        "youden_this_seed_validation": {
            "logit": own_t,
            "probability": float(1 / (1 + np.exp(-own_t))),
            "J": own_j,
        },
        "max_abs_head_vs_stored_logits": float(np.abs(z - z_stored).max()),
        "device": DEVICE,
    }

    sources = {
        "sae_k16_seed1_l1": ("sae_k16_seed1_l13.pt", CON / "codes_test.npy"),
        "sae_k16_seed1_nnls_refit": (
            "sae_k16_seed1_l13.pt",
            CON / "refit_nnls/codes_refit_test.npy",
        ),
        "pca_k16": ("pca_k16.pt", None),
    }
    rows = []
    for variant, (file, path) in sources.items():
        model, meta = load_autoencoder(str(SAE / file))
        U = model.decoder_weight.detach().double().cpu().numpy()
        b_d = model.decoder_bias.detach().double().cpu().numpy()
        codes = (
            np.load(path, mmap_mode="r")
            if path is not None
            else encode(model.to(DEVICE), test.h, device=DEVICE).numpy()
        )
        assert codes.shape == (len(z), U.shape[0]), codes.shape
        z_hat = np.concatenate(
            [head(np.asarray(codes[r], np.float64) @ U + b_d) for r in test.chunks(CH)]
        )
        for name, t in thresholds.items():
            ties = int(((z == t["logit"]) | (z_hat == t["logit"])).sum())
            for reference, zz in (("head_h", z), ("stored_logits", z_stored)):
                for r in flip_rows(zz, z_hat, y, t["logit"], t["strict"]):
                    rows.append(
                        {
                            "variant": variant,
                            "threshold": name,
                            "t_logit": t["logit"],
                            "t_probability": t["probability"],
                            "z": reference,
                            **r,
                            "ties_at_t": ties,
                        }
                    )
        print(variant, json.dumps(meta, default=str), flush=True)
        del codes, z_hat

    table = pd.DataFrame(rows)
    table.to_csv(out / "faithfulness_flips.csv", index=False)

    # the stored "class flips" of the refit run (test): flips at t = 0, overall and |f| < 1
    stored = pd.read_csv(CON / "refit_nnls/metrics.csv").query("split == 'test'")
    stored = stored.set_index("variant")
    mine = table.query("threshold == 'zero' and z == 'head_h' and label == 'all'")
    mine = mine.set_index(["variant", "band"])
    info["refit_run_stored_vs_recomputed"] = {}
    for stored_name, name in (
        ("sae_l1", "sae_k16_seed1_l1"),
        ("sae_l1_nnls_refit", "sae_k16_seed1_nnls_refit"),
    ):
        s, overall, near = (
            stored.loc[stored_name],
            mine.loc[(name, "all")],
            mine.loc[(name, "|z-t|<1")],
        )
        info["refit_run_stored_vs_recomputed"][stored_name] = {
            "flip_rate": [float(s["flip_rate"]), float(overall["flip_rate"])],
            "flip_rate_small_logit": [
                float(s["flip_rate_small_logit"]),
                float(near["flip_rate"]),
            ],
            "share_small_logit": [float(s["share_small_logit"]), near["n"] / len(z)],
        }
    (out / "faithfulness_flips_info.json").write_text(json.dumps(info, indent=1))

    main_rows = table.query("z == 'head_h'")
    wide = main_rows.pivot_table(
        index=["threshold", "band", "label"],
        columns="variant",
        values="flip_rate",
        sort=False,
    )
    counts = main_rows.query("variant == 'pca_k16'").set_index(
        ["threshold", "band", "label"]
    )["n"]
    wide.insert(0, "n", counts)
    wide.to_csv(out / "faithfulness_flips_table.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(wide.to_string(float_format=lambda x: f"{x:.4f}"))
        check = (
            table.query("z == 'stored_logits'").set_index(
                ["variant", "threshold", "band", "label"]
            )["flip_rate"]
            - table.query("z == 'head_h'").set_index(
                ["variant", "threshold", "band", "label"]
            )["flip_rate"]
        )
        print("max |rate(stored logits) - rate(head on h)|:", float(check.abs().max()))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
