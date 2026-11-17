# ruff: noqa: RUF001  (minus signs in report text are intended)
"""Old vs new Q2/Q3 sources results after replacing blank-token (position only) seed 1.

Compares two output folders of ``postprocessing.results_sources``: the thesis run (old seed 1,
evaluation c8f6f450) and the rerun with the retrained seed (evaluation 1f5e2e4a). Nothing is
recomputed except the replicate differences, which are rebuilt from both runs' ``replicates.npz``
exactly as ``results_sources`` builds them (seed mean per configuration, weighted sum).

Writes to ``--out-dir``: ``old_vs_new_configurations.csv``, ``old_vs_new_comparisons.csv``,
``zero_crossings.csv``, ``seed_sd.csv``, ``tab_res_sources_diff_rows.tex``  and ``report.md`` (every cell
with the blank token side by side, the numbers quoted in the text, and the zero-crossing claim).

Usage:
    uv run python scripts/reporting/sources_blank_seed1_diff.py --old OLD_DIR --new NEW_DIR --out-dir DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POSITION = "position_knn"
SOURCES = ("shape_self", "shape_knn", "texture_self", "texture_knn", POSITION)
METRICS = ("AUPRC", "AUROC")
# numbers quoted in the thesis text: (description, kind, key, metric, quoted string)
QUOTED = (
    ("position only AUPRC", "conf", POSITION, "AUPRC", "0.829"),
    (
        "shape − position only",
        "comp",
        "shape_vs_position_knn",
        "AUPRC",
        "+0.075 [0.069, 0.081]",
    ),
    (
        "texture − position only",
        "comp",
        "texture_vs_position_knn",
        "AUPRC",
        "+0.116 [0.109, 0.124]",
    ),
    (
        "position only − shape, self-only",
        "comp",
        "position_knn_vs_shape_self",
        "AUPRC",
        "+0.155 [0.142, 0.168]",
    ),
)


def text_signed(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x):.3f}"


DIFF_LABELS = {
    "shape_vs_texture_self": "shape $-$ texture, self-only",
    "shape_vs_texture_knn": "shape $-$ texture, with context",
    "shape_vs_position_knn": "shape $-$ position only",
    "texture_vs_position_knn": "texture $-$ position only",
    "context_gain_shape": "context gain, shape",
    "context_gain_texture": "context gain, texture",
    "interaction": "texture gain $-$ shape gain",
    "position_knn_vs_shape_self": "position only $-$ shape, self-only",
}
SOURCE_ROWS = (  # tab:res-sources: label, self-only, with context, context-gain comparison
    ("Position only", None, POSITION, None),
    ("Shape", "shape_self", "shape_knn", "context_gain_shape"),
    ("Texture", "texture_self", "texture_knn", "context_gain_texture"),
)
# configurations each thesis row depends on
ROW_CONFIGS = {
    **{label: {c for c in (s, k) if c} for label, s, k, _ in SOURCE_ROWS},
    "shape_vs_texture_self": {"shape_self", "texture_self"},
    "shape_vs_texture_knn": {"shape_knn", "texture_knn"},
    "shape_vs_position_knn": {"shape_knn", POSITION},
    "texture_vs_position_knn": {"texture_knn", POSITION},
    "context_gain_shape": {"shape_knn", "shape_self"},
    "context_gain_texture": {"texture_knn", "texture_self"},
    "interaction": {"shape_knn", "shape_self", "texture_knn", "texture_self"},
    "position_knn_vs_shape_self": {POSITION, "shape_self"},
}


def math_signed(x: float, star: bool = False) -> str:
    """$+0.230^{*}$ / $-0.105$ (thesis format)."""
    return f"${'+' if round(x, 3) >= 0 else '-'}{abs(x):.3f}{'^{*}' if star else ''}$"


def bound(x: float) -> str:
    """CI bound: 0.069 or $-0.113$."""
    return f"${'-'}{abs(x):.3f}$" if round(x, 3) < 0 else f"{x:.3f}"


def thesis_rows(conf: pd.DataFrame, comp: pd.DataFrame) -> dict[str, dict[str, str]]:
    def point(name: str, metric: str) -> str:
        return f"{conf[(conf.configuration == name) & (conf.metric == metric)]['point'].item():.3f}"

    def cmp(name: str, metric: str) -> pd.Series:
        return comp[(comp.comparison == name) & (comp.metric == metric)].iloc[0]

    sources = {}
    for label, self_name, ctx, gain in SOURCE_ROWS:
        cells = [
            point(self_name, "AUPRC") if self_name else "--",
            point(self_name, "AUROC") if self_name else "--",
            point(ctx, "AUPRC"),
            point(ctx, "AUROC"),
        ]
        if gain:
            g = cmp(gain, "AUPRC")
            cells.append(
                f"{math_signed(g.difference, g.significant == 'yes')} [{bound(g.ci_lo)}, {bound(g.ci_hi)}]"
            )
        else:
            cells.append("--")
        sources[label] = f"    {label} & " + " & ".join(cells) + r" \\"
    diff = {}
    for name, label in DIFF_LABELS.items():
        a, u = cmp(name, "AUPRC"), cmp(name, "AUROC")
        diff[name] = (
            f"    {label} & {math_signed(a.difference)} [{bound(a.ci_lo)}, {bound(a.ci_hi)}]"
            f" & {math_signed(u.difference)} [{bound(u.ci_lo)}, {bound(u.ci_hi)}] \\\\"
        )
    return {"res_sources": sources, "res_sources_diff": diff}


def replicate_differences(
    path: Path, comparisons: pd.DataFrame
) -> dict[tuple[str, str], np.ndarray]:
    z = np.load(path)
    names = z["configurations"]
    out = {}
    for _, c in comparisons.iterrows():
        w = json.loads(c["weights"])
        out[(c["comparison"], c["metric"])] = sum(
            float(wt) * z[c["metric"]][names == n].mean(0) for n, wt in w.items()
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # configurations ------------------------------------------------------------
    oc, nc = (pd.read_csv(d / "configurations.csv") for d in (args.old, args.new))
    cols = ["point", "ci_lo", "ci_hi", "seed_sd", "seed_1", "seed_2", "seed_3", "run_1"]
    conf = oc[oc.configuration.isin(SOURCES)][["configuration", "metric", *cols]].merge(
        nc[["configuration", "metric", *cols]],
        on=["configuration", "metric"],
        suffixes=("_old", "_new"),
    )
    for c in ("point", "ci_lo", "ci_hi", "seed_sd"):
        conf[f"{c}_change"] = conf[f"{c}_new"] - conf[f"{c}_old"]
    others = conf[conf.configuration != POSITION]
    unchanged_others = float(
        np.abs(
            others[
                [f"{c}_change" for c in ("point", "ci_lo", "ci_hi", "seed_sd")]
            ].to_numpy()
        ).max()
    )
    conf.to_csv(out / "old_vs_new_configurations.csv", index=False)
    nc[nc.configuration.isin(SOURCES)][
        [
            "configuration",
            "label",
            "metric",
            "n_seeds",
            "point",
            "seed_sd",
            "seed_1",
            "seed_2",
            "seed_3",
        ]
    ].to_csv(out / "seed_sd.csv", index=False)

    # comparisons ---------------------------------------------------------------
    keep = [
        "comparison",
        "label",
        "metric",
        "weights",
        "difference",
        "ci_lo",
        "ci_hi",
        "p",
        "p_holm",
        "significant",
    ]
    ocmp, ncmp = (
        pd.read_csv(d / "comparisons.csv").query("family == 'sources'")[keep]
        for d in (args.old, args.new)
    )
    comp = ocmp.drop(columns=["weights", "label"]).merge(
        ncmp, on=["comparison", "metric"], suffixes=("_old", "_new")
    )
    comp["involves_position_only"] = comp["weights"].str.contains(POSITION)
    for c in ("difference", "ci_lo", "ci_hi"):
        comp[f"{c}_change"] = comp[f"{c}_new"] - comp[f"{c}_old"]
    comp.to_csv(out / "old_vs_new_comparisons.csv", index=False)

    # zero crossings in the resamples ---------------------------------------------
    rows = []
    for tag, d, cmp in (("old", args.old, ocmp), ("new", args.new, ncmp)):
        for (name, metric), diff in replicate_differences(
            d / "replicates.npz", cmp
        ).items():
            point = cmp[(cmp.comparison == name) & (cmp.metric == metric)][
                "difference"
            ].item()
            wrong = (diff <= 0) if point > 0 else (diff >= 0)
            rows.append(
                {
                    "run": tag,
                    "comparison": name,
                    "metric": metric,
                    "difference": point,
                    "replicate_min": float(diff.min()),
                    "replicate_max": float(diff.max()),
                    "resamples": len(diff),
                    "resamples_at_or_across_zero": int(wrong.sum()),
                }
            )
    zero = pd.DataFrame(rows)
    zero.to_csv(out / "zero_crossings.csv", index=False)
    new_auprc_cross = int(
        zero[(zero.run == "new") & (zero.metric == "AUPRC")][
            "resamples_at_or_across_zero"
        ].sum()
    )

    # thesis-format rows (tab:res-sources, tab:res-sources-diff), old and new ----------
    tables = {
        tag: thesis_rows(c, m) for tag, c, m in (("old", oc, ocmp), ("new", nc, ncmp))
    }
    changed = []
    for table in ("res_sources", "res_sources_diff"):
        for key, new_row in tables["new"][table].items():
            old_row = tables["old"][table][key]
            if new_row != old_row:
                changed.append(
                    {
                        "table": table,
                        "row": key,
                        "involves_position_only": POSITION in ROW_CONFIGS[key],
                        "old": old_row,
                        "new": new_row,
                    }
                )
        (out / f"tab_{table}_rows.tex").write_text(
            "\n".join(tables["new"][table].values()) + "\n"
        )
        (out / f"tab_{table}_rows_old.tex").write_text(
            "\n".join(tables["old"][table].values()) + "\n"
        )
    changed = pd.DataFrame(
        changed, columns=["table", "row", "involves_position_only", "old", "new"]
    )
    changed.to_csv(out / "changed_rows.csv", index=False)
    unexpected = changed[~changed["involves_position_only"]]

    # report ----------------------------------------------------------------------
    L = [
        "# Q2/Q3 sources: retrained blank-token seed 1, old vs new",
        "",
        (
            f"Old: `{args.old}`. New: `{args.new}`. Other configurations unchanged: max |change| "
            f"{unchanged_others:.1e} over point, CI and seed SD."
        ),
        "",
        "## Position only (k-NN + rotated values)",
        "",
        "| Metric | Old | New | Change | Seed SD old → new | Seeds new |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in conf[conf.configuration == POSITION].iterrows():
        L.append(
            f"| {r.metric} | {r.point_old:.4f} [{r.ci_lo_old:.4f}, {r.ci_hi_old:.4f}] | "
            f"{r.point_new:.4f} [{r.ci_lo_new:.4f}, {r.ci_hi_new:.4f}] | {r.point_change:+.4f} | "
            f"{r.seed_sd_old:.4f} → {r.seed_sd_new:.4f} | {r.seed_1_new:.4f}, {r.seed_2_new:.4f}, {r.seed_3_new:.4f} |"
        )
    L += [
        "",
        "## Paired differences with position only",
        "",
        "| Comparison | Metric | Old | New | Change | Holm old → new |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in comp[comp.involves_position_only].iterrows():
        L.append(
            f"| {r.label} | {r.metric} | {text_signed(r.difference_old)} [{r.ci_lo_old:.3f}, {r.ci_hi_old:.3f}] | {text_signed(r.difference_new)} [{r.ci_lo_new:.3f}, {r.ci_hi_new:.3f}] | "
            f"{r.difference_change:+.4f} | {r.significant_old if isinstance(r.significant_old, str) else '–'} → "
            f"{r.significant_new if isinstance(r.significant_new, str) else '–'} |"
        )
    L += [
        "",
        "## Numbers quoted in the text",
        "",
        "| Quote | Thesis | New (3 decimals) | Changed |",
        "| --- | --- | --- | --- |",
    ]
    for desc, kind, key, metric, quoted in QUOTED:
        if kind == "conf":
            r = nc[(nc.configuration == key) & (nc.metric == metric)].iloc[0]
            new = f"{r.point:.3f}"
        else:
            r = ncmp[(ncmp.comparison == key) & (ncmp.metric == metric)].iloc[0]
            new = f"{text_signed(r.difference).replace('−', '-')} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
        L.append(
            f"| {desc} | {quoted} | {new} | {'**yes**' if new != quoted else 'no'} |"
        )
    zn = zero[(zero.run == "new") & (zero.metric == "AUPRC")]
    L += [
        "",
        (
            f"No AUPRC difference crosses zero in any resample: "
            f"{'still true' if new_auprc_cross == 0 else '**no longer true**'} "
            f"({new_auprc_cross} resample-comparisons at or across zero over {len(zn)} comparisons × "
            f"{int(zn.resamples.iloc[0])} resamples; closest to zero: "
            + ", ".join(
                f"{r.comparison} {min(abs(r.replicate_min), abs(r.replicate_max)):.3f}"
                for r in zn.sort_values("difference", key=abs).head(2).itertuples()
            )
            + ")."
        ),
        "",
        f"Thesis rows changed: {len(changed)} ({len(unexpected)} not involving position only"
        + (": **unexpected, check the run set / bootstrap**" if len(unexpected) else "")
        + ").",
        "",
        *[
            f"- `{r.row}`: `{r.old.strip()}` → `{r.new.strip()}`"
            for r in changed.itertuples()
        ],
        "",
        "Holm (AUPRC, eight comparisons): "
        + ", ".join(
            f"{r.comparison} {r.significant_new}"
            for r in comp[comp.metric == "AUPRC"].itertuples()
        )
        + ".",
        "",
    ]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
