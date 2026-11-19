# ruff: noqa: RUF002  (mathematical symbols in text are intended)
r"""Analysis of the returned blinded review of texture vs shape disagreements (part B).

Reads the filled `review_form.csv` and the private `answer_key.csv` of
`postprocessing.disagreement_review_material` and writes, to `--out-dir` (default: next to the
answer key; private, not logged):

- `items.csv`: every item with its answers and key (repeats included, flagged);
- `basis_interior.csv`: share of items with basis_interior ticked in T>S vs C and S>T vs C
  (originals only): difference with a 95 % percentile CI from a bootstrap over the slides
  (B = 2,000, seed 20261108; pooled shares within a resample), the paired within-slide comparison
  (mean per-slide difference, slide-bootstrap CI, Wilcoxon signed-rank p over the slides with
  both sets) and Fisher's exact test on the pooled 2 x 2 table as a check;
- `agreement.csv`: reviewer call vs true class per set (yes/no/cannot tell; agreement among the
  decided calls) and certainty per set;
- `repeats.csv`: each repeat vs its original (carcinoma call, certainty, basis ticks);
- `summary.md`: the tables above.

An item counts as reviewed when its carcinoma column is filled. Tick columns are ticked when
they hold anything other than empty / 0 / no / false.

Usage:
    uv run -m postprocessing.disagreement_review_analysis --form FILLED.csv [--key answer_key.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon


PRIVATE = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/disagreement_review_private"
)
SEED, N_BOOT = 20261108, 2000
BASIS = ("basis_interior", "basis_geometry", "basis_arrangement")
COMPARISONS = (("T>S", "C"), ("S>T", "C"))


def normalize(form: pd.DataFrame) -> pd.DataFrame:
    """Short column names, parsed answers."""
    rename = {}
    for c in form.columns:
        base = c.split(" (")[0].strip().lower().replace(" ", "_")
        rename[c] = base
    f = form.rename(columns=rename)
    call = f["carcinoma"].fillna("").astype(str).str.strip().str.lower()
    f["call"] = call.replace({"y": "yes", "n": "no", "cannot": "cannot tell", "ct": "cannot tell"})
    bad = ~f["call"].isin(["", "yes", "no", "cannot tell"])
    assert not bad.any(), f"unreadable carcinoma answers: {f.loc[bad, 'carcinoma'].tolist()}"
    f["reviewed"] = f["call"] != ""
    f["certainty"] = pd.to_numeric(f["certainty"], errors="coerce")
    for b in BASIS:
        v = f[b].fillna("").astype(str).str.strip().str.lower()
        f[b] = ~v.isin(["", "0", "no", "false", "n"])
    return f


def slide_bootstrap(items: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> dict:
    """basis_interior share a − b: pooled and paired within slide, CIs over slides."""
    sub = items[items["set"].isin([a, b])]
    slides = sorted(sub["slide"].unique())
    per = (sub.groupby(["slide", "set"])["basis_interior"].agg(["sum", "count"])
           .unstack("set").reindex(slides).fillna(0))
    xa, na = per[("sum", a)].to_numpy(), per[("count", a)].to_numpy()
    xb, nb = per[("sum", b)].to_numpy(), per[("count", b)].to_numpy()
    point = xa.sum() / na.sum() - xb.sum() / nb.sum()
    both = (na > 0) & (nb > 0)
    d = xa[both] / na[both] - xb[both] / nb[both]
    idx = rng.integers(0, len(slides), size=(N_BOOT, len(slides)))
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = xa[idx].sum(1) / na[idx].sum(1) - xb[idx].sum(1) / nb[idx].sum(1)
    pidx = rng.integers(0, len(d), size=(N_BOOT, len(d))) if len(d) else None
    paired = d[pidx].mean(1) if pidx is not None else np.array([np.nan])
    table = [[int(xa.sum()), int(na.sum() - xa.sum())], [int(xb.sum()), int(nb.sum() - xb.sum())]]
    _, fisher_p = fisher_exact(table)
    nonzero = d[d != 0]
    wil_p = float(wilcoxon(d).pvalue) if len(nonzero) else 1.0
    return {
        "comparison": f"{a} vs {b}",
        "n_items_a": int(na.sum()),
        "n_items_b": int(nb.sum()),
        "share_a": xa.sum() / na.sum() if na.sum() else np.nan,
        "share_b": xb.sum() / nb.sum() if nb.sum() else np.nan,
        "difference": point,
        "ci_lo": float(np.nanpercentile(pooled, 2.5)),
        "ci_hi": float(np.nanpercentile(pooled, 97.5)),
        "paired_slides": int(both.sum()),
        "paired_mean_difference": float(d.mean()) if len(d) else np.nan,
        "paired_ci_lo": float(np.nanpercentile(paired, 2.5)),
        "paired_ci_hi": float(np.nanpercentile(paired, 97.5)),
        "wilcoxon_p": wil_p,
        "fisher_table": str(table),
        "fisher_p": float(fisher_p),
    }


def analyse(form: pd.DataFrame, key: pd.DataFrame) -> dict[str, pd.DataFrame]:
    f = normalize(form)
    key = key.rename(columns={"alias": "slide"})
    items = key.merge(f.drop(columns=["carcinoma"]), on=["slide", "item"], how="left",
                      validate="1:1")
    assert len(items) == len(key) == len(f), "form and key differ in items"
    items["repeat"] = items["repeat_of_item"].notna()
    truth = np.where(items["true_class"] == "carcinoma", "yes", "no")
    items["agrees"] = np.where(items["call"].isin(["yes", "no"]), items["call"] == truth, np.nan)

    originals = items[~items["repeat"] & items["reviewed"]]
    rng = np.random.default_rng(SEED)
    basis = pd.DataFrame([slide_bootstrap(originals, a, b, rng) for a, b in COMPARISONS])

    agree_rows = []
    for s, part in originals.groupby("set"):
        decided = part[part["call"].isin(["yes", "no"])]
        agree_rows.append({
            "set": s,
            "reviewed": len(part),
            "yes": int((part["call"] == "yes").sum()),
            "no": int((part["call"] == "no").sum()),
            "cannot_tell": int((part["call"] == "cannot tell").sum()),
            "agree_with_label": int(decided["agrees"].sum()),
            "agreement_share": decided["agrees"].mean() if len(decided) else np.nan,
            "certainty_mean": part["certainty"].mean(),
            "certainty_median": part["certainty"].median(),
            **{f"{b}_share": part[b].mean() for b in BASIS},
            "annotation_plausible_no": int(
                (part["annotation_plausible"].fillna("").astype(str).str.strip().str.lower()
                 == "no").sum()),
        })
    agreement = pd.DataFrame(agree_rows)

    rep = items[items["repeat"]].merge(
        items[~items["repeat"]], left_on=["slide", "repeat_of_item"],
        right_on=["slide", "item"], suffixes=("", "_original"), validate="1:1")
    repeats = pd.DataFrame({
        "slide": rep["slide"],
        "item": rep["item"],
        "original_item": rep["item_original"],
        "call": rep["call"],
        "call_original": rep["call_original"],
        "same_call": rep["call"] == rep["call_original"],
        "certainty_difference": rep["certainty"] - rep["certainty_original"],
        **{f"same_{b}": rep[b] == rep[f"{b}_original"] for b in BASIS},
    })
    return {"items": items, "basis_interior": basis, "agreement": agreement, "repeats": repeats}


def md_table(df: pd.DataFrame) -> str:
    def cell(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def markdown(res: dict[str, pd.DataFrame]) -> str:
    out = ["# Blinded review: texture vs shape disagreements", ""]
    for name in ("basis_interior", "agreement", "repeats"):
        out += [f"## {name}", "", md_table(res[name]), ""]
    rep = res["repeats"]
    if len(rep):
        out.append(f"Repeat consistency: same carcinoma call on {int(rep['same_call'].sum())} of "
                   f"{len(rep)} repeat items.")
    out += ["", "## All items", "", md_table(res["items"].drop(columns=["slide_id", "x", "y"])), ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--form", type=Path, required=True)
    parser.add_argument("--key", type=Path, default=PRIVATE / "answer_key.csv")
    parser.add_argument("--out-dir", type=Path, default=PRIVATE / "review_results")
    args = parser.parse_args()
    res = analyse(pd.read_csv(args.form, dtype=str, keep_default_na=False)
                  .astype({"item": int}), pd.read_csv(args.key))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in res.items():
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
    (args.out_dir / "summary.md").write_text(markdown(res))
    print(args.out_dir / "summary.md")


if __name__ == "__main__":
    main()
