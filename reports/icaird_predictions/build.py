"""report_fast page of the iCAIRD test predictions: polygons (shape) vs position only, seed 1 each.

Sections: the 60 re-annotated positive test slides by subtype (CIN2, CIN3, squamous carcinoma,
adenocarcinoma, CGIN, other), then the 185 normal / inflammation test slides. Layers (designs
hand-written in design_positive.json / design_negative.json):
    - prediction masks of evaluation runs 98f53183 (Polygons + RoVE, seed 1) and a3f5e184 (Position
      only + RoVE, seed 1): nucleus polygons filled with p × 255 at level 1, heatmap (opacity ∝ p);
    - positive slides: label-1 nuclei of the pathologist's regions (render_label_masks.py, uploaded to
      the report run under pathologist_regions/) and the iCAIRD dataset annotations (run 93f7b781),
      the latter hidden by default.

Usage: python build.py [--sessions-dir DIR] [--publish]
    --sessions-dir writes one bound session per slide (for `reportfast build --check-only`);
    --publish uploads the page and its provenance to RUN (public MLflow, experiment 37).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from report_fast import Bullets, Mlflow, Prose, Report, Section, SlideGrid, XopatEndpoint, load_design
from report_fast.provenance import for_page, logged_dir, write_for


D = Path(__file__).parent
RUN = "ce12605cdc54420386e016ea3bc3fd63"  # report run (exp. 37), child of results run b973c17b
SHAPE = "public_mlflow/37/98f53183f3064eb19f6d121e08445369/artifacts/prediction_masks"
POSITION = "public_mlflow/37/a3f5e18485d44849bacfe9d099e7d4f7/artifacts/prediction_masks"
LABELS = f"public_mlflow/37/{RUN}/artifacts/pathologist_regions"
ANNOT = "public_mlflow/37/93f7b7810011443eba78350bd1eb19a0/artifacts"
SLIDE = "data/Public/Public/cervix/iCAIRD/slides/{}.tiff"
TEST_SLIDES = Path("/mnt/projects/nuclei_based_wsi_analysis/evaluation/icaird_test/inference/test_slides.parquet")
SLOTS_POS = {0: "slide", 1: "shape", 2: "position", 3: "pathologist", 4: "high_grade", 5: "malignant"}
SLOTS_NEG = {0: "slide", 1: "shape", 2: "position"}
SUBTYPES = {  # page order, as in the annotation report
    "cin2": "High grade — CIN2",
    "cin3": "High grade — CIN3",
    "squamous_carcinoma": "Malignant — Squamous carcinoma",
    "adenocarcinoma": "Malignant — Adenocarcinoma",
    "cgin": "Malignant — CGIN",
    "other": "Malignant — Other",
}
ENDPOINT = XopatEndpoint(
    base_url="https://xopat.rationai.cloud.trusted.e-infra.cz/v3/",
    wsi_base_url="https://xopat.rationai.cloud.trusted.e-infra.cz/wsi-service/",
    image_protocol=None,
    mount_root="/mnt",
)
TITLE = "iCAIRD Cervix test: Polygons vs Position only"
SUBTITLE = "Carcinoma probability per nucleus, seed 1 of each model (k-NN attention, k = 24, rotated values)"
CSS = ".rf-root { color: var(--rf-fg); background: var(--rf-bg); min-height: 100vh; }"

LEGEND = Section(
    title="Layers",
    blocks=[
        Prose(
            text="Each card opens the slide in the viewer. Layer opacity follows the predicted "
            "probability (transparent at 0, full colour at 1); switch layers on and off in the "
            "viewer to compare the two models."
        ),
        Bullets(
            items=[
                "Orange: Polygons + RoVE (nucleus shape), seed 1, evaluation run 98f53183.",
                "Magenta: Position only + RoVE (blank token), seed 1, evaluation run a3f5e184.",
                "Cyan (positive slides): nuclei inside the pathologist's re-annotated regions, "
                "the carcinoma nuclei scored at test. The regions are not exhaustive: carcinoma "
                "outside them is unlabelled, not benign.",
                "Blue / green, hidden by default (positive slides): the iCAIRD dataset annotations, "
                "high grade and malignant.",
                "Negative slides (normal / inflammation): every nucleus is benign.",
            ]
        ),
    ],
)


def bind_all() -> list[tuple[str, list]]:
    """[(section title, [session, ...])] in page order."""
    pos_design = load_design(D / "design_positive.json", slots=SLOTS_POS, endpoint=ENDPOINT)
    neg_design = load_design(D / "design_negative.json", slots=SLOTS_NEG, endpoint=ENDPOINT)
    slides = pd.read_parquet(TEST_SLIDES).sort_values("slide_id")
    assert len(slides) == 245
    pos = slides[slides["is_carcinoma"]]
    neg = slides[~slides["is_carcinoma"]]
    assert len(pos) == 60 and len(neg) == 185 and set(pos["subcategory"]) == set(SUBTYPES)
    assert (neg["category"] == "normal_inflammation").all()

    sections = []
    for subtype, title in SUBTYPES.items():
        cases = pos.loc[pos["subcategory"] == subtype, "slide_id"]
        sessions = [
            pos_design.bind(
                slide=SLIDE.format(c),
                shape=f"{SHAPE}/{c}.tiff",
                position=f"{POSITION}/{c}.tiff",
                pathologist=f"{LABELS}/{c}.tiff",
                high_grade=f"{ANNOT}/high_grade/{c}.tiff",
                malignant=f"{ANNOT}/malignant/{c}.tiff",
                name=f"{title.split(' — ')[1]} · {c}",
            )
            for c in cases
        ]
        sections.append((f"{title} ({len(sessions)} slides)", sessions))
    sessions = [
        neg_design.bind(
            slide=SLIDE.format(c), shape=f"{SHAPE}/{c}.tiff", position=f"{POSITION}/{c}.tiff",
            name=f"Normal / inflammation · {c}",
        )
        for c in neg["slide_id"]
    ]
    sections.append((f"Negative — Normal / inflammation ({len(sessions)} slides)", sessions))
    return sections


def main() -> None:
    groups = bind_all()
    if "--sessions-dir" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--sessions-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        for _, sessions in groups:
            for s in sessions:
                cfg = s.to_config()
                case = cfg["data"][0]["dataID"].rsplit("/", 1)[-1].removesuffix(".tiff")
                (out / f"{case}.json").write_text(json.dumps(cfg, indent=1))
        print(f"bound {sum(len(g) for _, g in groups)} sessions -> {out}")
        return

    blocks, all_sessions = [LEGEND], []
    for title, sessions in groups:
        blocks.append(Section(title=title, blocks=[SlideGrid(sessions, endpoint=ENDPOINT)]))
        all_sessions += sessions
    report = Report(title=TITLE, subtitle=SUBTITLE, blocks=blocks, theme="auto", css=CSS)
    record = for_page(
        report,
        all_sessions,
        endpoint=ENDPOINT,
        design={
            "positive": {"session": json.loads((D / "design_positive.json").read_text()), "slots": SLOTS_POS},
            "negative": {"session": json.loads((D / "design_negative.json").read_text()), "slots": SLOTS_NEG},
        },
        inputs={
            "script": "reports/icaird_predictions/build.py",
            "slides": "test table of mapping run 8ac7be17 (60 re-annotated positives + 185 normal/inflammation)",
            "predictions": "prediction_masks of evaluation runs 98f53183 (polygons) and a3f5e184 (position only), seed 1",
            "labels": f"pathologist_regions/ of run {RUN} (render_label_masks.py)",
            "annotations": "run 93f7b781 high_grade/, malignant/",
        },
    )
    out = report.write(D / "report.html")
    write_for(out, record)
    print(f"wrote {out}: {len(all_sessions)} cases in {len(groups)} sections")
    if "--publish" in sys.argv:
        with logged_dir(record) as directory:
            published = Mlflow.from_env().publish(report, run_id=RUN, extra_dir=directory)
        print("published", published.artifact, "on run", published.run_id)


if __name__ == "__main__":
    main()
