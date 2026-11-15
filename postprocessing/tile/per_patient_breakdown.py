# ruff: noqa: RUF001  (Greek letters in the report text are intended)
r"""Per-patient breakdown of the MMCI TL pixel vs nucleus comparison (NB-A-64 follow-up).

Reads only the outputs of the paired bootstrap run (`bootstrap_run`,
`postprocessing/tile/bootstrap_pixel_vs_nuclei.py`): `aligned_tiles.parquet` (tiles, labels,
scores, empty-ROI flags and ROI nucleus counts behind the bootstrap), `replicates.npz` (every
score column's AUPRC/AUROC per resample) and `comparisons.csv`. No retraining, no re-pooling.

Metrics: `tm_metrics` of the bootstrap script (torchmetrics BinaryAveragePrecision and
BinaryAUROC). Nucleus models are computed per seed and averaged over the seeds; the
`*_mean_of_seeds` score columns are not used.

1. Per patient: counts, prevalence, empty-ROI and few-nuclei tiles, the metrics of every model
   and seed, and the differences A - B (descriptive; no CIs, no tests).
2. Leave one patient out: pooled metrics and differences on the remaining patients.
3. Spearman correlations across patients of ΔAUPRC (Shape - pixel models) with the number of
   positive tiles, the share of empty-ROI tiles and the share of few-nuclei tiles.
4. Bootstrap detail from the stored replicates: the CI of each comparison to 4 decimals and the
   number of resamples with Δ >= 0 and Δ <= 0 (re-deriving the raw p of the bootstrap run).

Patients are labelled P1..Pn by number of positive tiles (descending; ties by fewer tiles
first, then id). The mapping to the real patient and case ids is written only to
`private_mapping_file`, never to `out_dir` or MLflow; every output is checked for real ids.
"""

import re
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args
from scipy.stats import spearmanr

from postprocessing.tile.bootstrap_pixel_vs_nuclei import (
    METRICS,
    REPO,
    _md,
    bootstrap_p,
    git_info,
    tm_metrics,
)


CODE = (
    "postprocessing/tile/per_patient_breakdown.py",
    "configs/postprocessing/tile/per_patient_breakdown.yaml",
)


def model_metrics(df: pd.DataFrame, config: DictConfig) -> dict[str, float]:
    """Metrics of every model (nucleus: per seed and mean of seeds) and every difference."""
    y = df["label"].to_numpy()
    out: dict[str, float] = {}
    for name in config.pixel_models:
        for m, v in tm_metrics(y, df[name].to_numpy()).items():
            out[f"{m}_{name}"] = v
    for name in config.nucleus_models:
        seeds = [
            tm_metrics(y, df[f"{name}_seed{i}"].to_numpy())
            for i in range(1, config.n_seeds + 1)
        ]
        for m in METRICS:
            for i, s in enumerate(seeds, start=1):
                out[f"{m}_{name}_seed{i}"] = s[m]
            out[f"{m}_{name}"] = float(np.mean([s[m] for s in seeds]))
    for a, b in config.comparisons:
        for m in METRICS:
            out[f"d{m}_{a}_minus_{b}"] = out[f"{m}_{a}"] - out[f"{m}_{b}"]
    return out


def counts(df: pd.DataFrame, few_max: int) -> dict[str, float]:
    n, pos = len(df), int(df["label"].sum())
    empty = df["empty_roi"].to_numpy()
    few = (df["roi_nuclei"] >= 1) & (df["roi_nuclei"] <= few_max)
    return {
        "n_slides": df["slide_id"].nunique(),
        "n_tiles": n,
        "n_pos": pos,
        "n_neg": n - pos,
        "prevalence": pos / n,
        "n_empty_roi": int(empty.sum()),
        "n_empty_roi_pos": int((empty & (df["label"] == 1)).sum()),
        "n_empty_roi_neg": int((empty & (df["label"] == 0)).sum()),
        "share_empty_roi": float(empty.mean()),
        f"share_few_nuclei_1to{few_max}": float(few.mean()),
        "median_roi_nuclei": float(df["roi_nuclei"].median()),
    }


def spearman_rows(pp: pd.DataFrame, xs: list[str], deltas: list[str]) -> pd.DataFrame:
    rows = []
    for x in xs:
        for d in deltas:
            rho, p = spearmanr(pp[x], pp[d])
            rows.append(
                {"x": x, "delta": d, "rho": float(rho), "p": float(p), "n": len(pp)}
            )
    return pd.DataFrame(rows)


def verdict(sp: pd.DataFrame) -> str:
    rhos, ps = sp["rho"].to_numpy(), sp["p"].to_numpy()
    if (rhos > 0).all() and (ps < 0.05).all():
        return "supported"
    if (rhos < 0).all() and (ps < 0.05).all():
        return "contradicted"
    return "inconclusive"


def compute(config: DictConfig, out: Path) -> dict:
    run = config.bootstrap_run
    df = pd.read_parquet(
        download_artifacts(run_id=run, artifact_path="aligned_tiles.parquet")
    )
    reps = np.load(download_artifacts(run_id=run, artifact_path="replicates.npz"))
    boot_comps = pd.read_csv(
        download_artifacts(run_id=run, artifact_path="comparisons.csv")
    )
    df["patient_id"] = df["patient_id"].astype(str)

    # ---- patient labels P1..Pn
    per = df.groupby("patient_id").agg(
        n_pos=("label", "sum"), n_tiles=("label", "size")
    )
    per["key"] = per.index.astype(int)
    per = per.sort_values(["n_pos", "n_tiles", "key"], ascending=[False, True, True])
    labels = {pid: f"P{i}" for i, pid in enumerate(per.index, start=1)}
    df["patient"] = df["patient_id"].map(labels)
    case_of = (
        df.groupby("patient_id")["slide_id"].first().str.extract(r"TP-(\d{4}_\d+)-")[0]
    )
    mapping = pd.DataFrame(
        {
            "patient": [labels[p] for p in per.index],
            "patient_id": per.index,
            "case_id": [case_of[p] for p in per.index],
            "n_pos": per["n_pos"].to_numpy(),
        }
    )
    private = Path(config.private_mapping_file)
    private.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(private, index=False)
    order = list(mapping["patient"])

    # ---- 1. per patient
    rows = []
    for p in order:
        g = df[df["patient"] == p]
        assert 0 < g["label"].sum() < len(g), f"{p}: needs both classes"
        rows.append(
            {
                "patient": p,
                **counts(g, config.few_nuclei_max),
                **model_metrics(g, config),
            }
        )
    pp = pd.DataFrame(rows)
    pp.to_csv(out / "per_patient.csv", index=False)

    # ---- 2. leave one patient out
    rows = [
        {
            "left_out": "none",
            **counts(df, config.few_nuclei_max),
            **model_metrics(df, config),
        }
    ]
    for p in order:
        g = df[df["patient"] != p]
        rows.append(
            {
                "left_out": p,
                **counts(g, config.few_nuclei_max),
                **model_metrics(g, config),
            }
        )
    lopo = pd.DataFrame(rows)
    lopo.to_csv(out / "leave_one_patient_out.csv", index=False)

    # ---- 3. correlations
    few_col = f"share_few_nuclei_1to{config.few_nuclei_max}"
    shape_deltas = ["dAUPRC_shape_minus_virchow2", "dAUPRC_shape_minus_vgg16"]
    sp_pos = spearman_rows(pp, ["n_pos"], shape_deltas)
    sp_other = spearman_rows(
        pp,
        ["prevalence", "share_empty_roi", few_col, "median_roi_nuclei"],
        shape_deltas,
    )
    # AUROC does not depend on prevalence: control for the AUPRC scale effect
    sp_auroc = spearman_rows(
        pp, ["n_pos"], [d.replace("AUPRC", "AUROC") for d in shape_deltas]
    )
    spear = pd.concat([sp_pos, sp_auroc, sp_other], ignore_index=True)
    spear.to_csv(out / "spearman.csv", index=False)

    # ---- 4. bootstrap detail from the stored replicates
    def rep(name: str, m: str) -> np.ndarray:
        if name in config.pixel_models:
            return reps[f"{name}__{m}"]
        return np.mean(
            [reps[f"{name}_seed{i}__{m}"] for i in range(1, config.n_seeds + 1)], axis=0
        )

    lo_q, hi_q = 100 * (1 - config.ci_level) / 2, 100 * (1 + config.ci_level) / 2
    detail = []
    for a, b in config.bootstrap_comparisons:
        for m in METRICS:
            d = rep(a, m) - rep(b, m)
            lo, hi = np.percentile(d, [lo_q, hi_q])
            row = {
                "comparison": f"{a} - {b}",
                "metric": m,
                "n_resamples": len(d),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_ge_0": int((d >= 0).sum()),
                "n_le_0": int((d <= 0).sum()),
                "n_eq_0": int((d == 0).sum()),
                "p_raw": bootstrap_p(d),
            }
            logged = boot_comps.set_index("comparison").loc[f"{a} - {b}"]
            assert np.isclose(row["p_raw"], logged[f"d{m}_p_raw"]), (a, b, m)
            assert np.isclose(lo, logged[f"d{m}_ci_lo"]) and np.isclose(
                hi, logged[f"d{m}_ci_hi"]
            )
            detail.append(row)
    detail = pd.DataFrame(detail)
    detail.to_csv(out / "bootstrap_detail.csv", index=False)

    write_report(out, config, pp, lopo, spear, sp_pos, sp_auroc, detail, few_col)
    check_no_real_ids(out, mapping)
    return {
        "verdict_auprc": verdict(sp_pos),
        "verdict_auroc": verdict(sp_auroc),
        "n_patients": len(order),
    }


def write_report(
    out: Path,
    config: DictConfig,
    pp: pd.DataFrame,
    lopo: pd.DataFrame,
    spear: pd.DataFrame,
    sp_pos: pd.DataFrame,
    sp_auroc: pd.DataFrame,
    detail: pd.DataFrame,
    few_col: str,
) -> None:
    lab = config.labels
    v, v_auroc = verdict(sp_pos), verdict(sp_auroc)
    rho_prev = spear[
        (spear["x"] == "prevalence") & spear["delta"].str.startswith("dAUPRC")
    ]["rho"]
    other = spear[spear["x"].isin(["share_empty_roi", few_col, "median_roi_nuclei"])]
    all_row = lopo[lopo["left_out"] == "none"].iloc[0]
    lines = [
        "# MMCI TL test: per-patient breakdown (steps 3 and 4)",
        "",
        (
            f"Source: bootstrap run `{config.bootstrap_run}` (`aligned_tiles.parquet`, "
            "`replicates.npz`). Patients P1-P10 are ordered by the number of positive tiles "
            "(descending). Per-patient metrics and differences are in `per_patient.csv`, the "
            "leave-one-patient-out results in `leave_one_patient_out.csv`. Nucleus models: per-seed "
            "metrics averaged over the 3 seeds."
        ),
        "",
        "## Step 3: is the shape model behind mainly on patients with few positive tiles?",
        "",
        (
            "If so, ΔAUPRC (Shape - pixel) would be most negative where positives are fewest, i.e. a "
            "positive Spearman correlation between the number of positive tiles and ΔAUPRC."
        ),
        "",
        _md(
            spear.assign(
                x=spear["x"], delta=spear["delta"].str.replace("dAUPRC_", "ΔAUPRC ")
            ),
            ".3f",
        ),
        "",
        (
            f"Rule (n = {len(pp)} patients): supported if ρ > 0 with p < 0.05 for both pixel models, "
            "contradicted if ρ < 0 with p < 0.05 for both, otherwise inconclusive. "
            f"**Verdict on ΔAUPRC: {v}. Same rule on ΔAUROC (prevalence-independent control): "
            f"{v_auroc}.**"
        ),
        "",
        (
            "Reading: the number of positive tiles and the prevalence are nearly interchangeable "
            "here (ρ with ΔAUPRC for prevalence: "
            + ", ".join(f"{r:.2f}" for r in rho_prev)
            + "), and "
            "per-patient AUPRC differences spread much more at low prevalence for any pair of "
            "models. AUROC, which does not depend on prevalence, "
            + (
                "shows the same association, so the explanation holds beyond the AUPRC scale."
                if v_auroc == "supported"
                else (
                    f"shows a weaker association in the same direction (ρ = "
                    f"{sp_auroc['rho'].min():.2f} to {sp_auroc['rho'].max():.2f}, p = "
                    f"{sp_auroc['p'].min():.2f} to {sp_auroc['p'].max():.2f}), so the data are "
                    "consistent with the explanation but cannot separate it from the "
                    "prevalence dependence of AUPRC."
                    if (sp_auroc["rho"] > 0).all()
                    else "does not show the association."
                )
            )
        ),
        "",
        (
            "Other pattern (share of empty-ROI tiles, of tiles with 1-"
            f"{config.few_nuclei_max} ROI nuclei, median ROI nuclei): |ρ| ≤ "
            f"{other['rho'].abs().max():.2f}, p ≥ {other['p'].min():.2f}; "
            + (
                "no association with where Shape loses."
                if (other["p"] >= 0.05).all()
                else "see the table."
            )
        ),
        "",
    ]
    base = lopo[lopo["left_out"] == "none"].iloc[0]
    rest = lopo[lopo["left_out"] != "none"].set_index("left_out")
    for d in ("dAUPRC_shape_minus_virchow2", "dAUPRC_shape_minus_vgg16"):
        shift = (rest[d] - base[d]).abs()
        p_max = shift.idxmax()
        lines.append(
            f"- Leave one out, {d.replace('dAUPRC_', 'ΔAUPRC ')}: range "
            f"{rest[d].min():+.4f} to {rest[d].max():+.4f} (all patients {base[d]:+.4f}); "
            f"leaving out {p_max} moves it most, to {rest.loc[p_max, d]:+.4f}."
        )
    lines.append("")
    for d in ("dAUPRC_shape_minus_virchow2", "dAUPRC_shape_minus_vgg16"):
        b = d.split("_minus_")[1]
        losers = pp.loc[pp[d] < 0, "patient"].tolist()
        lines.append(
            f"- Shape - {lab[b]}: Shape has the lower AUPRC on {len(losers)}/{len(pp)} patients "
            f"({', '.join(losers) or 'none'}); pooled Δ = {all_row[d]:+.4f}."
        )
    lines += [
        "",
        "Per patient (descriptive):",
        "",
        _md(
            pp[
                [
                    "patient",
                    "n_pos",
                    "prevalence",
                    "share_empty_roi",
                    few_col,
                    "median_roi_nuclei",
                    "dAUPRC_shape_minus_virchow2",
                    "dAUPRC_shape_minus_vgg16",
                ]
            ],
            ".4f",
        ),
        "",
        "Leave one patient out, pooled ΔAUPRC and ΔAUROC on the remaining patients:",
        "",
        _md(
            lopo[
                [
                    "left_out",
                    "n_pos",
                    *[f"dAUPRC_{a}_minus_{b}" for a, b in config.comparisons],
                    "dAUROC_shape_minus_virchow2",
                ]
            ],
            "+.4f",
        ),
        "",
        "## Step 4: bootstrap detail (2,000 resamples of the bootstrap run)",
        "",
    ]
    auroc_sv = detail[
        (detail["comparison"] == "shape - virchow2") & (detail["metric"] == "AUROC")
    ].iloc[0]
    lines += [
        (
            f"Shape - Virchow2 ΔAUROC, {config.ci_level:.0%} percentile CI: "
            f"[{auroc_sv['ci_lo']:+.4f}, {auroc_sv['ci_hi']:+.4f}] "
            f"(resamples with Δ ≥ 0: {auroc_sv['n_ge_0']}, Δ ≤ 0: {auroc_sv['n_le_0']}; "
            f"p raw = {auroc_sv['p_raw']:.4f})."
        ),
        "",
        (
            "Resample counts per comparison; p raw = 2 · min(n(Δ ≤ 0), n(Δ ≥ 0)) / 2000, capped at 1 "
            "(equal to the bootstrap run's comparisons.csv):"
        ),
        "",
        _md(detail, ".4f"),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def check_no_real_ids(out: Path, mapping: pd.DataFrame) -> None:
    """Fails if any output in out_dir contains a real patient or case id as a token."""
    tokens = [*mapping["patient_id"].astype(str), *mapping["case_id"].astype(str)]
    pattern = re.compile(
        r"(?<![\d.,])(" + "|".join(map(re.escape, tokens)) + r")(?![\d.,])|TP-\d{4}_"
    )
    for p in out.iterdir():
        if p.suffix in {".csv", ".md", ".yaml", ".json"}:
            hit = pattern.search(p.read_text())
            assert hit is None, f"{p.name} contains a real id: {hit.group(0)}"


# token prefixes split so the pattern itself is not flagged when this file is scanned
_PREFIXES = ("gh" + "p_", "github" + "_pat_", "gh" + "o_")
CREDENTIAL = re.compile("|".join(_PREFIXES) + r"|://[^/\s'\"]*@")


@with_cli_args(["+postprocessing=tile/per_patient_breakdown"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    out = Path(config.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = compute(config, out)
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    print(f"verdicts: {summary}; outputs in {out}")
    if not config.log_mlflow:
        return

    parent = client.get_run(config.bootstrap_run)
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        "mlflow.parentRunId": config.bootstrap_run,
        **git_info(*CODE),
    }
    if parent.data.tags.get("mlflow.user"):
        tags["mlflow.user"] = parent.data.tags["mlflow.user"]
    files = [
        *(
            out / n
            for n in ("per_patient.csv", "leave_one_patient_out.csv", "report.md")
        ),
        *(
            out / n
            for n in ("spearman.csv", "bootstrap_detail.csv", "config-resolved.yaml")
        ),
        *(REPO / c for c in CODE),
    ]
    for text in [*map(str, tags.values()), *(f.read_text() for f in files)]:
        assert not CREDENTIAL.search(text), "credential-like URL in a tag or artifact"

    run_id = client.create_run(
        parent.info.experiment_id, run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    client.log_param(run_id, "bootstrap_run", config.bootstrap_run)
    client.log_param(run_id, "few_nuclei_max", config.few_nuclei_max)
    client.log_param(run_id, "patient_order", "P1..P10 by positive tiles, descending")
    client.set_tag(run_id, "step3_verdict_auprc", summary["verdict_auprc"])
    client.set_tag(run_id, "step3_verdict_auroc", summary["verdict_auroc"])
    for f in files[:-2]:
        client.log_artifact(run_id, str(f))
    for f in files[-2:]:
        client.log_artifact(run_id, str(f), "code")
    client.set_terminated(run_id)
    print(f"MLflow run {run_id} (child of {config.bootstrap_run})")


if __name__ == "__main__":
    main()
