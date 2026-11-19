# ruff: noqa: RUF001, ISC004  (mathematical symbols in text are intended)
"""The pathologist's report for the blinded texture vs shape review (report_fast, xOpat v3).

Ten cards (S01-S10), each opening a PANDA Radboud test slide with two layers:
- "Review items": the numbered item markers of `postprocessing.disagreement_review_material`,
  rendered as a level-0 mask (ring around the nucleus, four ticks to find it at low zoom, the
  item number at the upper right; one style for all items), visible;
- "Reference annotation": the Radboud carcinoma mask of run cd504aaa (exp 38; red edge, opacity
  0.5, as the earlier Radboud report), hidden; the reviewer switches it on only for the
  "annotation plausible" column.
Nothing on the page names a set, a prediction or a concept. The slides are linked by their
PANDA paths (the slide ids are in the links; agreed for this report).

Steps (run in this order; the private answer key never leaves the private folder):
    uv run python reports/nucleus_review/build.py render
    uv run python reports/nucleus_review/build.py upload            # creates the run
    uv run python reports/nucleus_review/build.py build --run RUN    # local page + probe
    uv run python reports/nucleus_review/build.py build --run RUN --publish
Public MLflow credentials in the env for upload/publish.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
from openslide import OpenSlide
from PIL import Image, ImageDraw, ImageFont
from rationai.masks import write_big_tiff


HERE = Path(__file__).parent
PRIVATE = Path(
    "/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test/disagreement_review_private"
)
SLIDES = Path("/mnt/data/Public/Public/prostate/PANDA/train_images")
EXPERIMENT_ID = "37"
PARENT_RUN = "c8f268536be8416f9fe395eea95c458e"
RUN_NAME = "Nucleus review (Radboud test, 10 slides)"  # neutral: the reviewer may open this run
WEB = "https://mlflow.rationai.cloud.e-infra.cz/"
SLOTS = {0: "slide", 1: "markers", 2: "annotation"}
LABEL_MASKS = "public_mlflow/38/cd504aaafb8c408184584d808a99cf15/artifacts/carcinoma_mask"
# marker geometry (µm)
RING_RADIUS, RING_WIDTH = 9.0, 1.5
TICK_FROM, TICK_TO = 14.0, 28.0
FONT_HEIGHT, LABEL_OFFSET = 22.0, 16.0

TITLE = "Blinded nucleus review: 10 prostate slides, 80 marked nuclei"
SUBTITLE = "PANDA Radboud test slides S01–S10. Answers go in review_form.csv, one row per item."
CSS = ".rf-root { color: var(--rf-fg); background: var(--rf-bg); min-height: 100vh; }"


def key() -> pd.DataFrame:
    k = pd.read_csv(PRIVATE / "answer_key.csv")
    assert len(k) == 80 and k.groupby("alias").size().eq(8).all(), "expected 80 items, 8 per slide"
    return k


def render() -> None:
    """One marker mask per slide alias (0/255, level 0, same size as the slide)."""
    out = PRIVATE / "markers"
    out.mkdir(exist_ok=True)
    k = key()
    for alias, items in k.groupby("alias"):
        slide = SLIDES / f"{items['slide_id'].iloc[0]}.tiff"
        with OpenSlide(str(slide)) as s:
            w, h = s.dimensions
            mpp = float(s.properties["openslide.mpp-x"])
        px = 1.0 / mpp
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        font = ImageFont.load_default(size=round(FONT_HEIGHT * px))
        width = max(2, round(RING_WIDTH * px))
        for r in items.itertuples():
            x, y = r.x, r.y
            rr = RING_RADIUS * px
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), outline=255, width=width)
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                draw.line((x + dx * TICK_FROM * px, y + dy * TICK_FROM * px,
                           x + dx * TICK_TO * px, y + dy * TICK_TO * px), fill=255, width=width)
            off = LABEL_OFFSET * px
            draw.text((x + off, y - off), str(r.item), fill=255, font=font, anchor="lb")
        write_big_tiff(image=pyvips.Image.new_from_array(np.asarray(mask)),
                       path=out / f"{alias}.tiff", mpp_x=mpp, mpp_y=mpp)
        print(f"{alias}: {len(items)} markers, {w} x {h}", flush=True)


def upload() -> str:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    run = client.create_run(EXPERIMENT_ID, run_name=RUN_NAME, tags={
        "mlflow.parentRunId": PARENT_RUN,
    })
    run_id = run.info.run_id
    client.log_artifacts(run_id, str(PRIVATE / "markers"), "markers")
    client.log_artifacts(run_id, str(PRIVATE / "reviewer"), "reviewer")
    # no code here: the reviewer opens this run for the form, and the code describes the study
    print(f"run {run_id}")
    return run_id


def sessions(run_id: str) -> list:
    from report_fast import XopatEndpoint, load_design

    endpoint = XopatEndpoint(
        base_url="https://xopat.rationai.cloud.trusted.e-infra.cz/v3/",
        wsi_base_url="https://xopat.rationai.cloud.trusted.e-infra.cz/wsi-service/",
        image_protocol=None,
        mount_root="/mnt",
    )
    design = load_design(HERE / "design.json", slots=SLOTS, endpoint=endpoint)
    slides = key().drop_duplicates("alias").sort_values("alias")
    bound = [
        design.bind(
            slide=f"data/Public/Public/prostate/PANDA/train_images/{s}.tiff",
            markers=f"public_mlflow/{EXPERIMENT_ID}/{run_id}/artifacts/markers/{a}.tiff",
            annotation=f"{LABEL_MASKS}/{s}.tiff",
            name=a,
        )
        for a, s in zip(slides["alias"], slides["slide_id"], strict=True)
    ]
    return bound, endpoint


def build(run_id: str, publish: bool) -> None:
    from report_fast import Bullets, LinkList, Mlflow, Prose, Report, Section, SlideGrid
    from report_fast.provenance import for_page, logged_dir, write_for

    bound, endpoint = sessions(run_id)
    artifacts = f"{WEB}#/experiments/{EXPERIMENT_ID}/runs/{run_id}/artifacts"
    how_to = Section(
        title="How to review",
        blocks=[
            Prose(text=(
                "Each card opens one slide in the viewer. The layer 'Review items' marks the "
                "nuclei to assess: a ring around the nucleus (the nucleus itself is left "
                "clear), four ticks to find it at low magnification, and the item number at "
                "the upper right. Items are numbered 1–80 across the ten slides; each slide "
                "has 8. Record your answers in review_form.csv, one row per item."
            )),
            Bullets(items=[
                "carcinoma: is the marked nucleus part of carcinoma? yes / no / cannot tell.",
                "certainty: 1 (a guess) to 5 (certain).",
                "basis_interior, basis_geometry, basis_arrangement: put an x in every column "
                "your call relied on: the nucleus interior (chromatin, nucleoli), its size and "
                "shape, or the arrangement of the surrounding cells and glands.",
                "Answer carcinoma, certainty and the basis columns for all items of a slide "
                "before you switch on the layer 'Reference annotation' (off when the slide "
                "opens). Then fill 'annotation plausible' (yes / no): is the reference "
                "annotation at the marked nucleus plausible? Do not change earlier answers "
                "after seeing it.",
                "comment: anything else (artefact, out of focus, wrong nucleus marked).",
                "Nothing needs to be saved in the viewer.",
            ]),
            LinkList(links=[
                ("review_form.csv (the form to fill in)", f"{artifacts}/reviewer/review_form.csv"),
                ("INSTRUCTIONS.txt (the same instructions as text)",
                 f"{artifacts}/reviewer/INSTRUCTIONS.txt"),
            ]),
        ],
    )
    slides = Section(title="Slides", blocks=[SlideGrid(bound, endpoint=endpoint)])
    report = Report(title=TITLE, subtitle=SUBTITLE, blocks=[how_to, slides], theme="auto",
                    css=CSS)
    record = for_page(
        report, bound, endpoint=endpoint,
        design={"session": json.loads((HERE / "design.json").read_text()), "slots": SLOTS},
        inputs={
            "script": "nuclei-graph-transformer reports/nucleus_review/build.py",
            "items": "80 numbered nuclei on 10 slides",
            "markers": f"run {run_id} artifacts/markers",
            "annotation": "run cd504aaafb8c408184584d808a99cf15 carcinoma_mask (exp 38)",
        },
    )
    out_dir = PRIVATE / "report"
    out_dir.mkdir(exist_ok=True)
    out = report.write(out_dir / "report.html")
    write_for(out, record)
    print(f"wrote {out}: {len(bound)} cards")
    if publish:
        with logged_dir(record) as directory:
            published = Mlflow.from_env().publish(report, run_id=run_id, extra_dir=directory)
        print("published", published.artifact, "on run", published.run_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["render", "upload", "build"])
    parser.add_argument("--run")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.step == "render":
        render()
    elif args.step == "upload":
        upload()
    else:
        if not args.run:
            sys.exit("build needs --run")
        build(args.run, args.publish)


if __name__ == "__main__":
    main()
