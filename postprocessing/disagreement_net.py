# ruff: noqa: RUF001, RUF002, ISC004  (mathematical symbols in text are intended)
r"""Net disagreement between configurations: recall and specificity differences per concept.

Same inputs, sets and threshold as `postprocessing.disagreement_analysis` (run cdaf1247): PANDA
Radboud test, all 14.7 M nuclei, seed-mean probabilities, threshold 0.40, the dominant concept of
run 3c6e64a5 and the epithelium of run 44ffd165.

For a pair X vs Y, a column (group) g and a true class c:
    Δ = (n(X>Y, g, c) − n(Y>X, g, c)) / n(g, c) = acc_X(g, c) − acc_Y(g, c),
the recall difference for c = carcinoma and the specificity difference for c = benign, in
percentage points. Columns: the dominant concepts A-H and "other"; "benign epithelial" and
"benign non-epithelial" (benign nuclei of each compartment); "all" (every nucleus of the class).
95 % percentile CIs from the shared stratified slide bootstrap (seed 20261108, B = 2,000; the
saved resamples): per resample, multiplicity-weighted sums of the slides' net counts over the
weighted sums of their nucleus counts. p = 2 x the smaller tail fraction, unadjusted.

Outputs in `out_dir` (the cdaf1247 folder; no slide ids): `net.csv`, `net.md`,
`figures/res_disagreement_{context,sources}_net.{pdf,png}`. With `log_run`, they are added to that
MLflow run (artifacts net.csv, net.md, figures/*_net.*, code/net/).
"""

import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args

from postprocessing.bootstrap_sources import bootstrap_p
from postprocessing.disagreement_analysis import load_inputs
from postprocessing.results_context import REPO
from visualization.disagreement_net_figure import FIGURES, save_net_figures


EXTRA = ("benign epithelial", "benign non-epithelial", "all")
CLASSES = (("carcinoma", 1), ("benign", 0))
SOURCES = (
    "postprocessing/disagreement_net.py",
    "visualization/disagreement_net_figure.py",
    "postprocessing/disagreement_analysis.py",
    "configs/postprocessing/disagreement_analysis.yaml",
)


def code_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *SOURCES)
    return f"{git('rev-parse', 'HEAD')}{' (dirty)' if dirty else ''}"


def cells(data: dict) -> list[tuple[str, str, np.ndarray]]:
    """(column, class, mask) in table order."""
    y, group, epi = data["y"], data["group"], data["epithelial"]
    out = []
    for g, name in enumerate(data["group_names"]):
        for cname, cval in CLASSES:
            out.append((name, cname, (group == g) & (y == cval)))
    out.append(("benign epithelial", "benign", (y == 0) & epi))
    out.append(("benign non-epithelial", "benign", (y == 0) & ~epi))
    for cname, cval in CLASSES:
        out.append(("all", cname, y == cval))
    return out


def net_table(config: DictConfig, data: dict) -> pd.DataFrame:
    t = float(config.threshold)
    y, slide = data["y"], data["slide"]
    n_slides = len(data["slide_ids"])
    w = np.load(config.resamples)["multiplicities"].astype(np.float64)
    pct = (1 - config.bootstrap.ci_level) / 2 * 100
    cell_list = cells(data)
    rows = []
    for pair in config.pairs:
        px, py = data["p_mean"][pair.x], data["p_mean"][pair.y]
        cx, cy = (px >= t) == (y == 1), (py >= t) == (y == 1)
        x_wins, y_wins = cx & ~cy, cy & ~cx
        for col, cname, mask in cell_list:
            per = np.stack([
                np.bincount(slide[mask & m], minlength=n_slides).astype(np.float64)
                for m in (x_wins, y_wins, np.ones_like(mask))
            ], axis=1)  # slides x (X>Y, Y>X, n)
            nx, ny, n = per.sum(0)
            rep = w @ per
            with np.errstate(invalid="ignore", divide="ignore"):
                reps = 100 * (rep[:, 0] - rep[:, 1]) / rep[:, 2]
            ok = np.isfinite(reps)
            lo, hi = np.percentile(reps[ok], [pct, 100 - pct])
            rows.append({
                "pair": pair.name,
                "question": pair.question,
                "x": pair.x_short,
                "y": pair.y_short,
                "column": col,
                "true_class": cname,
                "metric": "recall" if cname == "carcinoma" else "specificity",
                "n": int(n),
                "n_x_better": int(nx),
                "n_y_better": int(ny),
                "delta_pp": 100 * (nx - ny) / n,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "p": bootstrap_p(reps[ok]),
                "resamples_undefined": int((~ok).sum()),
            })
    return pd.DataFrame(rows)


def markdown(config: DictConfig, table: pd.DataFrame) -> str:
    lines = [
        "# Net disagreement: recall and specificity differences (PANDA Radboud test)",
        "",
        "Δ = (n(X>Y) − n(Y>X)) / n, percentage points; positive: X better. Carcinoma = recall "
        "difference, benign = specificity difference. Threshold "
        f"{config.threshold}, seed-mean probabilities, all test nuclei; 95 % CI from the "
        "stratified slide bootstrap (B = 2,000, seed 20261108). Cells: Δ [95 % CI].",
        "",
    ]
    columns = list(dict.fromkeys(table["column"]))
    for pair in config.pairs:
        part = table[table["pair"] == pair.name]
        lines += [f"## {pair.x_short} vs {pair.y_short} ({pair.question}); positive: "
                  f"{pair.x_short} better", "",
                  "| column | n carcinoma | carcinoma (recall) | n benign | benign (specificity) |",
                  "|---|---:|---|---:|---|"]
        for col in columns:
            cells_ = {}
            for cname in ("carcinoma", "benign"):
                r = part[(part["column"] == col) & (part["true_class"] == cname)]
                if len(r):
                    r = r.iloc[0]
                    cells_[cname] = (f"{r.n:,}",
                                     f"{r.delta_pp:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]")
                else:
                    cells_[cname] = ("–", "–")
            lines.append(f"| {col} | {cells_['carcinoma'][0]} | {cells_['carcinoma'][1]} | "
                         f"{cells_['benign'][0]} | {cells_['benign'][1]} |")
        lines.append("")
    return "\n".join(lines)


def compute(config: DictConfig) -> Path:
    out = Path(config.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data = load_inputs(config)
    table = net_table(config, data)
    table.to_csv(out / "net.csv", index=False)
    (out / "net.md").write_text(markdown(config, table))
    dirs = [out / "figures"]
    save_net_figures(out, dirs, suffix="_net")
    if config.get("repo_figures_dir"):
        save_net_figures(out, [Path(config.repo_figures_dir)], suffix="")
    return out


def log(config: DictConfig, out: Path) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    run = config.log_run
    for name in ("net.csv", "net.md"):
        client.log_artifact(run, str(out / name))
    for stem in FIGURES:
        for ext in ("pdf", "png"):
            client.log_artifact(run, str(out / "figures" / f"{stem}_net.{ext}"), "figures")
    for source in SOURCES:
        client.log_artifact(run, str(REPO / source), "code/net")
    client.set_tag(run, "net_commit", code_commit())
    print(f"logged to run {run}")


@with_cli_args(["+postprocessing=disagreement_analysis"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    out = compute(config)
    if config.get("log_run"):
        log(config, out)
    print(out)


if __name__ == "__main__":
    main()
