"""Thesis figure res_concept_subsets: deletion curves of the concepts with linear probes.

The k = 16 seed-1 L1 SAE (SAE training run 4cf1db8f) of model run c8f26853, PANDA Radboud.
Concepts are ranked by their share of the total mean |φ| on the validation nuclei (L1 codes;
the ranking of `xai.concept_subsets`, run 571ede3a: G B H A #9 F #7 #15 #13 E #14 D #10 #5
#11 C). Starting from all 16, concepts are removed one at a time, most important first, and
as a control least important first. After each removal, a linear probe is refitted on the
remaining concepts: unpenalised logistic regression of the nucleus label on their
activations (`xai.concept_subsets.fit_logistic`: Newton's method, no penalty), fitted on
the validation nuclei and evaluated on all 14,730,504 test nuclei (AUPRC, AUROC; labels as
the nucleus-level evaluation). Done for the encoder's codes κ (L1) and the NNLS-refitted
codes κ̃ (run 6532785d), and with two fit sets: all validation nuclei (as
`xai.concept_subsets`; the reported numbers) and only the SAE's training slides
(validation without the holdout slides of `sae/holdout_slides.csv`; a sensitivity check).
Also the probes on the 8 stable and on the 8 unstable concepts.

Outputs: `<OUT_DIR>/concept_deletion_probes.csv` (cache; `--recompute` refits),
`figures/res_concept_subsets.{pdf,png}` and `figures/res_concept_subsets_stats.json`.
No slide IDs are written. Not logged to MLflow.

    PYTHONPATH=. python visualization/concept_deletion_figure.py [--recompute]
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl


mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.transforms import blended_transform_factory

from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import classification_metrics
from xai.concept_morphometry import FIG_W_CM, THESIS_RC
from xai.concept_subsets import fit_logistic


XAI_DIR = Path(
    "/mnt/public_projects/nuclei_based_wsi_analysis/xai/c8f268536be8416f9fe395eea95c458e"
)
CONCEPT_DIR = XAI_DIR / "concepts" / "k16_seed1"
SUBSETS_DIR = CONCEPT_DIR / "subsets"
CODES = {
    "l1": {
        "validation": CONCEPT_DIR / "morphometry" / "codes_validation.npy",
        "test": CONCEPT_DIR / "codes_test.npy",
    },
    "refit": {
        "validation": CONCEPT_DIR / "refit_nnls" / "codes_refit_validation.npy",
        "test": CONCEPT_DIR / "refit_nnls" / "codes_refit_test.npy",
    },
}
OUT_DIR = Path("/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/concepts")
CSV = OUT_DIR / "concept_deletion_probes.csv"
FIG = Path("figures/res_concept_subsets.pdf")
RUNS = {
    "model_training": "c8f268536be8416f9fe395eea95c458e",
    "sae_training": "4cf1db8f2cff4fef844426e57a387b99",
    "refit_magnitudes": "6532785defde40c7a9669fa4ff6c5774",
    "concept_subsets_ranking": "571ede3adf42451a9d83e8ef27eab307",
}
FIT_SETS = ("all_validation", "sae_fit_slides")
ORDERS = {"most_first": "most important removed first", "least_first": "least important removed first"}
ORDER_COLORS = {"most_first": "#c0392b", "least_first": "#2a78d6"}
VARIANT_STYLE = {"l1": "-", "refit": "--"}
VARIANT_LABEL = {"l1": r"$\ell_1$ codes $\kappa$", "refit": r"refitted codes $\tilde\kappa$"}
REFERENCE = "#7a7a7a"


def ranking() -> tuple[list[str], dict[str, int], set[str]]:
    rank = pd.read_csv(SUBSETS_DIR / "ranking.csv")
    rank = rank[rank["variant"] == "l1"].sort_values("rank")
    checks = json.loads((SUBSETS_DIR / "checks.json").read_text())
    names = rank["concept"].tolist()
    assert checks["ranking_variant"] == "l1" and names == checks["ranking"], "ranking mismatch"
    index = dict(zip(rank["concept"], rank["sae_index"].astype(int), strict=True))
    return names, index, set(rank.loc[rank["stable"], "concept"])


def subsets(names: list[str], stable: set[str]) -> list[tuple[str, int, list[str]]]:
    """(order, number removed, concepts kept); k = 0 is shared by both orders."""
    out = []
    for k in range(16):
        out.append(("most_first", k, names[k:]))
        out.append(("least_first", k, names[: 16 - k]))
    out.append(("stable8", 8, [n for n in names if n in stable]))
    out.append(("unstable8", 8, [n for n in names if n not in stable]))
    return out


def compute() -> pd.DataFrame:
    names, index, stable = ranking()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation = load_activations(XAI_DIR / "activations", "validation")
    test = load_activations(XAI_DIR / "activations", "test")
    holdout_ids = set(pd.read_csv(XAI_DIR / "sae" / "holdout_slides.csv")["slide_id"].astype(str))
    fit_rows = {
        "all_validation": np.ones(len(validation), bool),
        "sae_fit_slides": ~validation.index["slide_id"].astype(str).isin(holdout_ids).to_numpy(),
    }
    y_val, y_test = validation.labels, test.labels
    rows = []
    for variant, paths in CODES.items():
        kv = np.load(paths["validation"]).astype(np.float64)
        kt = np.load(paths["test"]).astype(np.float64)
        assert len(kv) == len(y_val) and len(kt) == len(y_test)
        for fit_set, mask in fit_rows.items():
            cache: dict[tuple[str, ...], dict] = {}
            for order, k, kept in subsets(names, stable):
                key = tuple(sorted(kept))
                if key not in cache:
                    cols = [index[n] for n in kept]
                    coef, intercept, info = fit_logistic(kv[mask][:, cols], y_val[mask], device)
                    scores = kt[:, cols] @ coef + intercept
                    cache[key] = classification_metrics(y_test, scores) | {
                        "fit_iterations": info["iterations"],
                        "fit_grad_max": info["grad_max"],
                    }
                removed = [n for n in names if n not in kept]
                rows.append({
                    "variant": variant, "fit_set": fit_set, "order": order, "n_removed": k,
                    "removed_at_step": (names[k - 1] if order == "most_first" else names[16 - k])
                    if order in ORDERS and k > 0 else "",
                    "concepts_removed": " ".join(removed), "concepts_kept": " ".join(kept),
                    **cache[key],
                    "n_fit_nuclei": int(mask.sum()), "n_test_nuclei": len(y_test),
                })
                print(variant, fit_set, order, k, round(cache[key]["AUPRC"], 4), flush=True)
    df = pd.DataFrame(rows)
    df["prevalence_test"] = float(y_test.mean())
    df["model_test_AUPRC"] = json.loads((SUBSETS_DIR / "checks.json").read_text())["model_test_AUPRC"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    return df


def check_against_stored(df: pd.DataFrame) -> dict:
    """The all-validation probes against `xai.concept_subsets` (curves.csv)."""
    curves = pd.read_csv(SUBSETS_DIR / "curves.csv")
    diffs = []
    for _, r in df[(df["fit_set"] == "all_validation") & df["order"].isin(list(ORDERS))].iterrows():
        curve, m = (("deletion", r["n_removed"]) if r["order"] == "most_first"
                    else ("insertion", 16 - r["n_removed"]))
        s = curves[(curves["variant"] == r["variant"]) & (curves["curve"] == curve) & (curves["m"] == m)].iloc[0]
        assert set(str(s["concepts_kept"]).split()) == set(r["concepts_kept"].split())
        diffs.append(abs(s["head_AUPRC"] - r["AUPRC"]))
        diffs.append(abs(s["head_AUROC"] - r["AUROC"]))
    return {"max_abs_diff_vs_concept_subsets_curves": float(max(diffs)), "n_compared": len(diffs)}


def summary(df: pd.DataFrame) -> dict:
    auprc_f = float(df["model_test_AUPRC"].iloc[0])
    out = {"model_test_AUPRC": auprc_f, "prevalence_test": float(df["prevalence_test"].iloc[0])}
    for fit_set in FIT_SETS:
        for variant in CODES:
            part = df[(df["variant"] == variant) & (df["fit_set"] == fit_set)]
            named = {s: part[part["order"] == s].iloc[0] for s in ("stable8", "unstable8")}
            most = part[part["order"] == "most_first"].sort_values("n_removed")
            res = {f"{s}_{m}": float(named[s][m]) for s in named for m in ("AUPRC", "AUROC")}
            for label, thr in (("below_95pct_model", 0.95 * auprc_f),
                               ("below_unstable8", float(named["unstable8"]["AUPRC"]))):
                hit = most[most["AUPRC"] < thr]
                res[f"most_first_n_removed_{label}"] = int(hit["n_removed"].iloc[0]) if len(hit) else None
                res[f"most_first_removed_{label}"] = hit["concepts_removed"].iloc[0] if len(hit) else None
                res[f"threshold_{label}"] = thr
            least = part[part["order"] == "least_first"].sort_values("n_removed")
            hit = least[least["AUPRC"] < 0.95 * auprc_f]
            res["least_first_n_removed_below_95pct_model"] = int(hit["n_removed"].iloc[0]) if len(hit) else None
            out[f"{variant}/{fit_set}"] = res
    return out


def figure(df: pd.DataFrame, names: list[str]) -> plt.Figure:
    df = df[df["fit_set"] == "all_validation"]
    auprc_f, prevalence = float(df["model_test_AUPRC"].iloc[0]), float(df["prevalence_test"].iloc[0])
    width = FIG_W_CM / 2.54
    fig, ax = plt.subplots(figsize=(width, 2.7))
    ax.axhline(auprc_f, color=REFERENCE, lw=0.8, ls=":", zorder=1)
    ax.axhline(prevalence, color=REFERENCE, lw=0.8, ls="-.", zorder=1)
    for order in ORDERS:
        for variant in CODES:
            v = df[(df["variant"] == variant) & (df["order"] == order)].sort_values("n_removed")
            ax.plot(v["n_removed"], v["AUPRC"], color=ORDER_COLORS[order], ls=VARIANT_STYLE[variant],
                    lw=1.1, marker="o" if variant == "l1" else None, ms=2.3, zorder=3)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(0.3, 0.95)
    ax.set_xticks(range(16))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=2)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    rows = {"most_first": names, "least_first": names[::-1]}
    for r, order in enumerate(ORDERS):
        y = -0.06 - 0.075 * r
        for k in range(16):
            label = "all" if k == 0 else rows[order][k - 1]
            ax.text(k, y, label, transform=trans, ha="center", va="top", fontsize=6,
                    color="#333333" if k == 0 else ORDER_COLORS[order])
    ax.text(-0.9, -0.06, "most first", transform=trans, ha="right", va="top", fontsize=6,
            color=ORDER_COLORS["most_first"])
    ax.text(-0.9, -0.135, "least first", transform=trans, ha="right", va="top", fontsize=6,
            color=ORDER_COLORS["least_first"])
    ax.set_xlabel("concepts removed (the concept removed at each step; unstable ones by SAE index)",
                  labelpad=24)
    ax.set_ylabel("nucleus AUPRC (test)")
    for k in range(16):
        ax.text(k, 1.0, str(k), transform=trans, ha="center", va="bottom", fontsize=5.5, color="#777777")
    ax.grid(True, color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [
        mpl.lines.Line2D([], [], color=ORDER_COLORS["most_first"], lw=1.1),
        mpl.lines.Line2D([], [], color=ORDER_COLORS["least_first"], lw=1.1),
        mpl.lines.Line2D([], [], color="#333333", lw=1.1, ls="-", marker="o", ms=2.3),
        mpl.lines.Line2D([], [], color="#333333", lw=1.1, ls="--"),
        mpl.lines.Line2D([], [], color=REFERENCE, lw=0.8, ls=":"),
        mpl.lines.Line2D([], [], color=REFERENCE, lw=0.8, ls="-."),
    ]
    labels = ["most important first", "least important first", VARIANT_LABEL["l1"],
              VARIANT_LABEL["refit"], f"model ({auprc_f:.3f})", f"prevalence ({prevalence:.3f})"]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=6.5,
               handlelength=2.0, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.27, top=0.84)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    df = compute() if args.recompute or not CSV.exists() else pd.read_csv(CSV, keep_default_na=False)
    names, _, _ = ranking()
    stats = {"runs": RUNS, "mlflow": "not logged", "csv": str(CSV), "ranking": names,
             "probe": "unpenalised logistic regression (Newton; xai.concept_subsets.fit_logistic)",
             **check_against_stored(df), **summary(df)}
    with mpl.rc_context(THESIS_RC):
        fig = figure(df, names)
        FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG)
        fig.savefig(FIG.with_suffix(".png"), dpi=300)
    FIG.with_name(FIG.stem + "_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
