# ruff: noqa: RUF002  (mathematical symbols in text are intended)
r"""Material for a blinded pathologist review of texture vs shape disagreements (part B).

PANDA Radboud test set; seed-mean probabilities of texture (T) and shape (S), k-NN + RoVE,
threshold 0.40, as `postprocessing.disagreement_analysis`.

Candidates per nucleus: T>S (T correct, S wrong) and S>T with |p_T − p_S| ≥ 0.5; controls C (both
correct, |p_T − p_S| < 0.1).

Slides (one generator, seed 20261108, used in this order): a slide is eligible by count if it has
≥ 3 T>S, ≥ 2 S>T and ≥ 3 C candidates of a true class present among its T>S candidates (the
class-only fallback of the matching). Each stratum (carcinoma slide / not) is put in random order
and its slides are tried in that order until 7 (3) have a complete item draw; a slide whose draw
fails (spacing) is skipped and reported. Taking the first feasible slides of a random order is a
uniform draw among the feasible slides. A short stratum is filled from the other one (reported).

Items per slide: 3 T>S, then 2 S>T, then for each T>S item one C of the same true class and
dominant concept (else the same class only; counted), each drawn at random among the
candidates ≥ 50 µm from every item already chosen. Aliases S01-S10 are a random permutation of
the slides. Repeats: on 8 random slides, one random item gets a second marker (`repeat_mode`,
see the config). Item numbers run 1..N over S01..S10, in random order within a slide.

Outputs in `private_dir` (not logged to MLflow; folder mode 700):
- `reviewer/S01.geojson` ... : GeoJSON point features (level-0 px), property `name` = item number;
- `reviewer/review_form.csv`: alias, item and empty answer columns;
- `answer_key.csv` (private): sets, labels, probabilities, concepts, matching, repeats, slide ids;
- `selection.json` (private): eligibility counts, skipped slides, matching success.
"""

import json
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import shapely
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args

from postprocessing.bootstrap_sources import nucleus_keys
from postprocessing.disagreement_analysis import load_inputs


SETS = ("T>S", "S>T", "C")
FORM_COLUMNS = [
    "slide", "item", "carcinoma (yes/no/cannot tell)", "certainty (1-5)",
    "basis_interior", "basis_geometry", "basis_arrangement",
    "annotation plausible (yes/no)", "comment",
]


def candidates(config: DictConfig, data: dict) -> pd.DataFrame:
    r = config.review
    t = float(config.threshold)
    y = data["y"]
    pt, ps = data["p_mean"][r.first], data["p_mean"][r.second]
    ct, cs = (pt >= t) == (y == 1), (ps >= t) == (y == 1)
    diff = np.abs(pt - ps)
    sets = {
        "T>S": ct & ~cs & (diff >= r.candidate_min_difference),
        "S>T": cs & ~ct & (diff >= r.candidate_min_difference),
        "C": ct & cs & (diff < r.control_max_difference),
    }
    rows = np.flatnonzero(sets["T>S"] | sets["S>T"] | sets["C"])
    label = np.select([sets[s][rows] for s in SETS], list(SETS), default="")
    return pd.DataFrame({
        "row": rows,
        "slide": data["slide"][rows],
        "key": data["key"][rows],
        "set": label,
        "true_class": np.where(y[rows] == 1, "carcinoma", "benign"),
        "p_T": pt[rows],
        "p_S": ps[rows],
        "concept": np.asarray(data["group_names"])[data["group"][rows]],
        "epithelial": data["epithelial"][rows],
    })


def eligible_by_count(cand: pd.DataFrame, items: DictConfig) -> pd.Series:
    out = {}
    for s, part in cand.groupby("slide"):
        ts = part[part["set"] == "T>S"]
        st = part[part["set"] == "S>T"]
        c = part[(part["set"] == "C") & part["true_class"].isin(ts["true_class"])]
        out[s] = len(ts) >= items["T>S"] and len(st) >= items["S>T"] and len(c) >= items["C"]
    return pd.Series(out)


def slide_nuclei(config: DictConfig, slide_row: pd.Series) -> pd.DataFrame:
    old, new = config.review.nuclei_root_remap
    path = str(slide_row["slide_nuclei_path"]).replace(old, new, 1)
    d = pd.read_parquet(path, columns=["id", "polygon", "centroid"])
    xy = np.stack(d["centroid"].to_numpy()).astype(np.float64)
    return pd.DataFrame({
        "key": nucleus_keys(d["id"]),
        "x": xy[:, 0],
        "y": xy[:, 1],
        "polygon": d["polygon"].to_numpy(),
    })


def draw_items(part: pd.DataFrame, items: DictConfig, min_px: float,
               rng: np.random.Generator) -> tuple[list[dict], dict] | None:
    """Items of one slide, or None if the spacing rule cannot be met."""
    chosen: list[dict] = []

    def far(row: pd.Series) -> bool:
        return all(np.hypot(row["x"] - c["x"], row["y"] - c["y"]) >= min_px for c in chosen)

    def pick(pool: pd.DataFrame) -> pd.Series | None:
        for i in rng.permutation(len(pool)):
            row = pool.iloc[i]
            if far(row):
                return row
        return None

    for s in ("T>S", "S>T"):
        pool = part[part["set"] == s]
        for _ in range(items[s]):
            row = pick(pool)
            if row is None:
                return None
            chosen.append(row.to_dict() | {"matched_to": None, "matched_on": None})
    controls = part[part["set"] == "C"]
    stats = {"class_and_concept": 0, "class_only": 0}
    for target in [c for c in chosen if c["set"] == "T>S"][: items["C"]]:
        same_class = controls[controls["true_class"] == target["true_class"]]
        row = pick(same_class[same_class["concept"] == target["concept"]])
        on = "class_and_concept"
        if row is None:
            row, on = pick(same_class), "class_only"
        if row is None:
            return None
        stats[on] += 1
        chosen.append(row.to_dict() | {"matched_to": target["key"], "matched_on": on})
    return chosen, stats


def repeat_marker(item: dict, nuclei: pd.DataFrame, config: DictConfig, mpp: float,
                  rng: np.random.Generator) -> tuple[float, float, int] | None:
    """(x, y, key of the marked nucleus) of a repeat marker, or None."""
    r = config.review
    xy = nuclei[["x", "y"]].to_numpy()
    if r.repeat_mode == "neighbour":
        d = np.hypot(xy[:, 0] - item["x"], xy[:, 1] - item["y"]) * mpp
        d[nuclei["key"].to_numpy() == item["key"]] = np.inf
        j = int(np.argmin(d))
        if d[j] > r.repeat_max_distance_um:
            return None
        return float(xy[j, 0]), float(xy[j, 1]), int(nuclei["key"].iloc[j])
    assert r.repeat_mode == "same_nucleus", r.repeat_mode
    assert r.repeat_offset_um <= r.repeat_max_distance_um
    poly = nuclei.loc[nuclei["key"] == item["key"], "polygon"].iloc[0]
    polygon = shapely.Polygon(np.asarray(poly, np.float64).reshape(-1, 2))
    for scale in (1.0, 0.75, 0.5):
        for angle in rng.uniform(0, 2 * np.pi, 16):
            off = scale * r.repeat_offset_um / mpp
            x, y = item["x"] + off * np.cos(angle), item["y"] + off * np.sin(angle)
            nearest = int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))
            if polygon.contains(shapely.Point(x, y)) and nuclei["key"].iloc[nearest] == item["key"]:
                return float(x), float(y), int(item["key"])
    return None


def geojson(items: pd.DataFrame) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(r.x, 1), round(r.y, 1)]},
                "properties": {"name": str(r.item)},
            }
            for r in items.sort_values("item").itertuples()
        ],
    }


def build(config: DictConfig) -> dict:
    r = config.review
    rng = np.random.default_rng(r.seed)
    data = load_inputs(config)
    slides = pd.read_parquet(config.epithelium.slides)
    slides = slides[slides["split"] == "test"].sort_values("slide_id", ignore_index=True)
    assert slides["slide_id"].tolist() == data["slide_ids"]

    cand = candidates(config, data)
    ok = eligible_by_count(cand, r.per_slide)
    ok = ok[ok].index.to_numpy()
    carcinoma = data["carcinoma_slide"]
    strata = {True: ok[carcinoma[ok]], False: ok[~carcinoma[ok]]}
    orders = {k: v[rng.permutation(len(v))] for k, v in strata.items()}
    want = {True: int(r.slides_with_carcinoma), False: int(r.slides_without_carcinoma)}

    selected: dict[int, tuple[list[dict], dict, pd.DataFrame]] = {}
    skipped = {True: 0, False: 0}
    position = {True: 0, False: 0}

    def take(stratum: bool, n: int) -> int:
        got = 0
        order = orders[stratum]
        while got < n and position[stratum] < len(order):
            s = int(order[position[stratum]])
            position[stratum] += 1
            nuclei = slide_nuclei(config, slides.iloc[s])
            mpp = float(slides["mpp_x"].iloc[s])
            part = cand[cand["slide"] == s].merge(nuclei[["key", "x", "y"]], on="key",
                                                  how="left", validate="1:1")
            assert part["x"].notna().all(), "candidate without a polygon"
            drawn = draw_items(part, r.per_slide, r.min_distance_um / mpp, rng)
            if drawn is None:
                skipped[stratum] += 1
                continue
            selected[s] = (*drawn, nuclei)
            got += 1
        return got

    shortfall = {}
    for stratum in (True, False):
        got = take(stratum, want[stratum])
        shortfall[stratum] = want[stratum] - got
    filled = {}
    for stratum in (True, False):  # fill a short stratum from the other one
        if shortfall[stratum]:
            filled[not stratum] = take(not stratum, shortfall[stratum])
    total_slides = sum(want.values())
    assert len(selected) == total_slides, f"only {len(selected)} slides"

    # aliases, repeats, numbering
    slide_order = list(selected)
    aliases = {s: f"S{i + 1:02d}" for i, s in
               enumerate(np.asarray(slide_order)[rng.permutation(len(slide_order))])}
    repeat_slides = set(rng.choice(sorted(selected), size=int(r.repeats), replace=False).tolist())
    rows, repeat_failures = [], 0
    for s in sorted(selected, key=lambda s: aliases[s]):
        chosen, _, nuclei = selected[s]
        mpp = float(slides["mpp_x"].iloc[s])
        slide_items = [dict(c, repeat_of=None, marker_key=c["key"], offset_um=0.0) for c in chosen]
        if s in repeat_slides:
            for i in rng.permutation(len(chosen)):
                marker = repeat_marker(chosen[i], nuclei, config, mpp, rng)
                if marker is not None:
                    x, y, key = marker
                    slide_items.append(dict(
                        chosen[i], x=x, y=y, marker_key=key, repeat_of=chosen[i]["key"],
                        matched_to=None, matched_on=None,
                        offset_um=float(np.hypot(x - chosen[i]["x"], y - chosen[i]["y"]) * mpp),
                    ))
                    break
                repeat_failures += 1
            else:
                raise RuntimeError(f"no repeat marker on {aliases[s]}")
        for it in slide_items:
            it["alias"], it["slide_id"] = aliases[s], data["slide_ids"][s]
        rows += [slide_items[i] for i in rng.permutation(len(slide_items))]
    def hexkey(k: object) -> str | None:
        return None if k is None else f"{int(k):016x}"

    for it in rows:
        for c in ("key", "matched_to", "repeat_of", "marker_key"):
            it[c] = hexkey(it[c])
    items = pd.DataFrame(rows).astype({"matched_to": object, "repeat_of": object})
    items["item"] = np.arange(1, len(items) + 1)
    number = {(a, k): i for a, k, i, rep in
              zip(items["alias"], items["key"], items["item"], items["repeat_of"], strict=True)
              if rep is None}
    items["matched_to_item"] = [number[(a, k)] if k is not None else None
                                for a, k in zip(items["alias"], items["matched_to"], strict=True)]
    items["repeat_of_item"] = [number[(a, k)] if k is not None else None
                               for a, k in zip(items["alias"], items["repeat_of"], strict=True)]
    return {
        "items": items,
        "cand": cand,
        "eligible": strata,
        "skipped": skipped,
        "shortfall": shortfall,
        "filled": filled,
        "repeat_failures": repeat_failures,
        "match_stats": {k: sum(v[1][k] for v in selected.values())
                        for k in ("class_and_concept", "class_only")},
        "mpp": {aliases[s]: float(slides["mpp_x"].iloc[s]) for s in selected},
        "carcinoma_aliases": sorted(aliases[s] for s in selected if carcinoma[s]),
    }


def write(config: DictConfig, result: dict) -> Path:
    out = Path(config.private_dir)
    (out / "reviewer").mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    items = result["items"]
    for alias, part in items.groupby("alias"):
        (out / "reviewer" / f"{alias}.geojson").write_text(json.dumps(geojson(part), indent=1))
    form = pd.DataFrame({"slide": items["alias"], "item": items["item"]})
    for c in FORM_COLUMNS[2:]:
        form[c] = ""
    form.to_csv(out / "reviewer" / "review_form.csv", index=False)
    (out / "reviewer" / "INSTRUCTIONS.txt").write_text(
        "Each slide file (S01-S10) holds numbered point markers, one nucleus each.\n"
        "For every item in review_form.csv:\n"
        "  carcinoma: yes / no / cannot tell (is the marked nucleus part of carcinoma?)\n"
        "  certainty: 1 (guess) - 5 (certain)\n"
        "  basis_interior / basis_geometry / basis_arrangement: put x in every column that\n"
        "    describes what your call relied on: the nucleus interior (chromatin, nucleolus),\n"
        "    its size and shape, or the arrangement of the surrounding cells and glands\n"
        "  annotation plausible: yes / no (is the reference annotation at this point plausible?)\n"
        "  comment: free text\n"
    )

    key = items.assign(
        epithelial=items["epithelial"].astype(bool),
        x=items["x"].round(1),
        y=items["y"].round(1),
    )[["alias", "item", "set", "true_class", "p_T", "p_S", "concept", "epithelial",
       "matched_to_item", "matched_on", "repeat_of_item", "offset_um", "slide_id", "x", "y"]]
    key = key.astype({"matched_to_item": "Int64", "repeat_of_item": "Int64"})
    key.to_csv(out / "answer_key.csv", index=False)

    originals = items[items["repeat_of"].isna()]  # repeats excluded
    per_alias = (originals.groupby(["alias", "set", "true_class"]).size()
                 .unstack(["set", "true_class"], fill_value=0))
    per_alias.columns = [f"{a} {b}" for a, b in per_alias.columns]
    selection = {
        "eligible_by_count": {"with_carcinoma": len(result["eligible"][True]),
                              "without_carcinoma": len(result["eligible"][False])},
        "skipped_spacing": {"with_carcinoma": result["skipped"][True],
                            "without_carcinoma": result["skipped"][False]},
        "stratum_shortfall": {"with_carcinoma": result["shortfall"][True],
                              "without_carcinoma": result["shortfall"][False]},
        "filled_from_other_stratum": {str(k): v for k, v in result["filled"].items()},
        "candidates_total": result["cand"]["set"].value_counts().to_dict(),
        "matching": result["match_stats"],
        "repeat_mode": config.review.repeat_mode,
        "repeat_marker_attempts_failed": result["repeat_failures"],
        "items": len(items),
        "items_per_alias": per_alias.to_dict(orient="index"),
        "carcinoma_slide_aliases": result["carcinoma_aliases"],
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=1, default=str))
    return out


@with_cli_args(["+postprocessing=disagreement_review_material"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    result = build(config)
    out = write(config, result)
    print(out)


if __name__ == "__main__":
    main()
