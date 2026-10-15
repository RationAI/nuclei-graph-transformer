"""Material for a blinded pathologist review of the concepts of one SAE.

Writes to `output_dir`:
- `deck.pdf`: an instruction page, a worked example (an unstable concept), one page per
  stable concept and, optionally, the other unstable concepts. Concepts are relabeled
  with letters in a random order (fixed seed). A page shows the concept's regions
  (`regions.parquet` of `xai.concept_material`) in a random order as H&E next to the
  activation overlay (scale bar, shared color bar without values), and a strip of the
  nuclei with the highest activation. No concept numbers, weights, importances,
  activation values or labels appear in the deck.
- `key.csv` (letters -> concepts and their statistics) and `regions_key.csv` (panels and
  strip positions -> slides, regions and nuclei); not to be shown in the blind phase.
- `response_sheet.pdf` and `response_sheet.xlsx`: one row per letter.
- `overlays/concept_<j>/<slide_id>.tiff`: whole-slide activation masks of `overlays.concepts`
  in the format of `nuclei_graph.callbacks.masks` (nucleus polygons filled with
  ceil(255 · min(c_j / color_scale_max, 1)); 0 = no nucleus or inactive), for one carcinoma
  and one benign-dominant test slide per concept, listed in `overlays/overlays.csv`.
"""

import io
import string
import textwrap
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
from einops import rearrange
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from omegaconf import DictConfig
from openslide import OpenSlide
from PIL import Image, ImageDraw
from rationai.masks import slide_resolution, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import Activations, load_activations
from nuclei_graph.xai.runtime import setup_precision
from xai.concept_material import SlideReader, draw_region, nucleus_crop


SECTIONS = ("main", "additional")
RESPONSE_COLUMNS = [
    ("Concept", 9),
    ("Where active (free text)", 42),
    ("Nucleus / context / artifact", 16),
    ("Direction (carcinoma / benign / neither)", 18),
    ("Name", 22),
    ("Confidence (1-5)", 11),
    ("Match yes/no (after unblinding)", 14),
    ("Comment (after unblinding)", 30),
]
CHOICES = {
    2: ["nucleus", "context", "artifact"],
    3: ["carcinoma", "benign", "neither"],
    5: ["1", "2", "3", "4", "5"],
    6: ["yes", "no"],
}
INSTRUCTIONS = """Each page shows one concept that the model uses internally. Concepts have letters
in a random order; nothing on the pages tells what the model or the annotations say.

How to read a page
  • Four regions of 0.5 x 0.5 mm, each as H&E (left) next to the same region with every
    nucleus drawn as a polygon (right). Colored nuclei: the concept is active; brighter
    (towards yellow) = stronger. Light gray nuclei: not active. The color scale is the same
    for all regions of a page, but not across pages. The regions are in a random order;
    usually one of them shows little or no activity, for comparison.
  • Bottom: {strip} nuclei with the strongest activity (outlined in green, {strip_um:g} µm of context).
  • Scale bar: {scale:g} µm.

What to fill in (response sheet, one row per letter)
  • Where active: in your own words, where the colored nuclei are (cell types, tissue
    compartments, architecture, location in the tissue).
  • Nucleus / context / artifact: is the concept mostly about the nuclei themselves, about
    their surroundings, or about artifacts (edges, folds, debris, staining)?
  • Direction: does the concept point towards carcinoma, towards benign tissue, or neither?
  • Name: a short name you would give the concept.
  • Confidence 1-5: how sure you are about the description (1 = guess, 5 = certain).
  • Match and comment are filled in after unblinding.

The next page is a worked example: an additional concept shown in the same layout. The
table on the right is the response row it asks for; it only shows the form of the answers and
does not describe the example concept."""
EXAMPLE_ROW = [
    "Example",
    "<where the colored nuclei are, e.g. cell type, compartment, location>",
    "<nucleus / context / artifact>",
    "<carcinoma / benign / neither>",
    "<a short name>",
    "<1-5>",
    "",
    "",
]


def assign_letters(
    summary: pd.DataFrame, config: DictConfig
) -> tuple[pd.DataFrame, int]:
    """Letters for the concepts in a random order: stable ones first, then the others."""
    rng = np.random.default_rng(config.session_seed)
    unstable = summary.loc[summary["unstable"], "concept"].to_numpy()
    example = (
        int(config.example_concept)
        if config.example_concept is not None
        else int(rng.choice(unstable))
    )
    groups = {
        "main": rng.permutation(
            summary.loc[~summary["unstable"], "concept"].to_numpy()
        ),
        "additional": rng.permutation(unstable[unstable != example])
        if config.include_unstable
        else np.array([], dtype=int),
    }
    rows = []
    for section in SECTIONS:
        for concept in groups[section]:
            rows.append(
                {
                    "letter": string.ascii_uppercase[len(rows)],
                    "concept": int(concept),
                    "section": section,
                }
            )
    rows.append({"letter": "Example", "concept": example, "section": "example"})
    return pd.DataFrame(rows), example


def scale_bar(ax: plt.Axes, size_um: float, bar_um: float) -> None:
    x0, y = 0.05 * size_um, 0.94 * size_um
    ax.plot(
        [x0, x0 + bar_um], [y, y], color="white", linewidth=5, solid_capstyle="butt"
    )
    ax.plot(
        [x0, x0 + bar_um], [y, y], color="black", linewidth=3, solid_capstyle="butt"
    )
    ax.text(
        x0 + bar_um / 2,
        y - 0.03 * size_um,
        f"{bar_um:g} µm",
        ha="center",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.0},
    )


def render_page(
    title: str,
    regions: pd.DataFrame,
    strip: pd.DataFrame,
    c: np.ndarray,
    vmax: float,
    reader: SlideReader,
    test: Activations,
    slides: pd.DataFrame,
    config: DictConfig,
) -> Image.Image:
    """One deck page: 4 regions (H&E | overlay) in the given order and a nucleus strip."""
    size = config.regions.size_um
    cmap = plt.get_cmap(config.regions.cmap)
    norm = Normalize(0.0, vmax)
    width, height = config.page.width_in, config.page.height_in
    fig = plt.figure(figsize=(width, height), dpi=config.page.dpi)
    fig.text(0.03, 0.965, title, fontsize=16, fontweight="bold", va="center")

    side = 0.30 * height / width  # square panels, 30 % of the page height
    for panel, region in enumerate(regions.itertuples(index=False)):
        x = 0.03 + (panel % 2) * (2 * side + 0.07)
        y = 0.62 if panel < 2 else 0.29
        ax_he = fig.add_axes((x, y, side, 0.30))
        ax_map = fig.add_axes((x + side + 0.01, y, side, 0.30))
        draw_region(
            ax_he, ax_map, region, reader, test, slides, c, norm, cmap, config.regions
        )
        scale_bar(ax_he, size, config.page.scale_bar_um)
        ax_he.set_title(f"Region {panel + 1}", fontsize=10, loc="left")

    fig.text(
        0.03,
        0.245,
        f"Nuclei with the strongest activity ({config.strip.size_um:g} µm context, outlined)",
        fontsize=9,
    )
    offsets = slides["offset"]
    tile_w = 0.13 * height / width
    for n, nucleus in enumerate(strip.itertuples(index=False)):
        ax = fig.add_axes((0.03 + n * (tile_w + 0.008), 0.10, tile_w, 0.13))
        ax.imshow(
            nucleus_crop(
                nucleus, reader, offsets, config.strip.size_um, config.strip.tile_px
            )
        )
        ax.set_xticks([])
        ax.set_yticks([])

    cax = fig.add_axes((0.03, 0.045, 0.30, 0.022))
    colorbar = fig.colorbar(
        ScalarMappable(norm, cmap), cax=cax, orientation="horizontal"
    )
    colorbar.set_ticks([])
    cax.text(0.0, -0.6, "low", transform=cax.transAxes, ha="left", va="top", fontsize=8)
    cax.text(
        1.0, -0.6, "high", transform=cax.transAxes, ha="right", va="top", fontsize=8
    )
    cax.set_title("activity (this page)", fontsize=8, loc="left")
    fig.legend(
        handles=[Patch(color=config.regions.inactive_color, label="not active")],
        loc="lower left",
        bbox_to_anchor=(0.35, 0.03),
        fontsize=8,
        frameon=False,
    )
    return figure_image(fig)


def figure_image(fig: plt.Figure) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=fig.dpi)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def text_page(
    title: str, text: str, config: DictConfig, table: list[list[str]] | None = None
) -> Image.Image:
    fig = plt.figure(
        figsize=(config.page.width_in, config.page.height_in), dpi=config.page.dpi
    )
    fig.text(0.05, 0.93, title, fontsize=16, fontweight="bold")
    fig.text(
        0.05, 0.88, text, fontsize=10, va="top", family="DejaVu Sans", linespacing=1.4
    )
    if table is not None:  # field / value rows, right of the text
        fig.text(
            0.64, 0.80, "Worked example: response row", fontsize=11, fontweight="bold"
        )
        ax = fig.add_axes((0.64, 0.42, 0.33, 0.36))
        ax.axis("off")
        cells = ax.table(
            cellText=[[field, textwrap.fill(value, 34)] for field, value in table],
            colLabels=["Column", "Entry"],
            loc="upper left",
            cellLoc="left",
            colWidths=[0.42, 0.58],
        )
        cells.auto_set_font_size(False)
        cells.set_fontsize(8)
        for (row, _), cell in cells.get_celld().items():
            cell.set_height(0.13 if row > 0 else 0.08)
    return figure_image(fig)


def response_sheet_pdf(letters: list[str], path: Path, config: DictConfig) -> None:
    widths = [7, 36, 14, 14, 20, 13, 13, 27]
    header = [
        "Concept",
        "Where active (free text)",
        "Nucleus /\ncontext /\nartifact",
        "Direction",
        "Name",
        "Confidence\n(1-5)",
        "Match yes/no\n(after\nunblinding)",
        "Comment\n(after unblinding)",
    ]
    rows_per_page = 12
    with PdfPages(path) as pdf:
        for start in range(0, len(letters), rows_per_page):
            fig = plt.figure(figsize=(config.page.width_in, config.page.height_in))
            fig.text(
                0.04,
                0.95,
                "Concept review: response sheet",
                fontsize=14,
                fontweight="bold",
            )
            fig.text(
                0.04,
                0.915,
                "Reviewer: ______________________    Date: ____________",
                fontsize=10,
            )
            ax = fig.add_axes((0.04, 0.04, 0.92, 0.85))
            ax.axis("off")
            body = [
                [
                    letter,
                    "",
                    "nucleus\ncontext\nartifact",
                    "carcinoma\nbenign\nneither",
                    "",
                    "1  2  3  4  5",
                    "yes   no",
                    "",
                ]
                for letter in letters[start : start + rows_per_page]
            ]
            table = ax.table(
                cellText=body,
                colLabels=header,
                loc="upper center",
                cellLoc="left",
                colWidths=[w / sum(widths) for w in widths],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (row, _), cell in table.get_celld().items():
                cell.set_height(0.10 if row == 0 else 0.066)
                if row > 0:
                    cell.get_text().set_color("#555555")
            pdf.savefig(fig)
            plt.close(fig)


def response_sheet_xlsx(letters: list[str], path: Path) -> None:
    """A minimal OOXML workbook (no extra dependency) with dropdowns for the choices."""

    def column(index: int) -> str:
        return string.ascii_uppercase[index]

    def cell(ref: str, value: str, style: int = 0) -> str:
        return f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>{escape(value)}</t></is></c>'

    rows = [
        '<row r="1">'
        + "".join(
            cell(f"{column(i)}1", name, 1)
            for i, (name, _) in enumerate(RESPONSE_COLUMNS)
        )
        + "</row>"
    ]
    for r, letter in enumerate(letters, start=2):
        rows.append(f'<row r="{r}">{cell(f"A{r}", letter, 1)}</row>')
    last = len(letters) + 1
    cols = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{width}" customWidth="1"/>'
        for i, (_, width) in enumerate(RESPONSE_COLUMNS)
    )
    validations = "".join(
        f'<dataValidation type="list" allowBlank="1" showErrorMessage="1" sqref="{column(i)}2:{column(i)}{last}">'
        f'<formula1>"{",".join(options)}"</formula1></dataValidation>'
        for i, options in CHOICES.items()
    )
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f"<cols>{cols}</cols><sheetData>{''.join(rows)}</sheetData>"
        f'<dataValidations count="{len(CHOICES)}">{validations}</dataValidations>'
        "</worksheet>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>",
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>",
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Responses" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>",
        "xl/worksheets/sheet1.xml": sheet,
        "xl/styles.xml": styles,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def overlay_slides(
    concept: int,
    codes: np.ndarray,
    test: Activations,
    slides: pd.DataFrame,
    overlays: DictConfig,
) -> list[tuple[str, pd.Series]]:
    """The carcinoma and the benign-dominant test slide with the highest mean c_j."""
    starts = slides["offset"].to_numpy()
    labels = test.labels.astype(np.float64)
    stats = slides.assign(
        carcinoma_share=np.add.reduceat(labels, starts) / slides["n_nuclei"].to_numpy(),
        mean_c=np.add.reduceat(codes[:, concept].astype(np.float64), starts)
        / slides["n_nuclei"].to_numpy(),
    )
    stats = stats[stats["n_nuclei"] >= overlays.min_slide_nuclei]
    roles = {
        "carcinoma": stats[
            stats["is_carcinoma"]
            & (stats["carcinoma_share"] >= overlays.carcinoma_min_share)
        ],
        "benign_dominant": stats[stats["carcinoma_share"] <= overlays.benign_max_share],
    }
    return [
        (role, pool.loc[pool["mean_c"].idxmax()])
        for role, pool in roles.items()
        if len(pool)
    ]


def write_overlay(
    slide: pd.Series,
    c: np.ndarray,
    vmax: float,
    reader: SlideReader,
    overlays: DictConfig,
    path: Path,
) -> None:
    """Nucleus polygons filled with ceil(255 · min(c / vmax, 1)), as the masks callback."""
    with OpenSlide(slide.slide_path) as wsi:
        mask_size = wsi.level_dimensions[overlays.level]
        base_mpp_x, base_mpp_y = slide_resolution(wsi, 0)
        mask_mpp_x, mask_mpp_y = slide_resolution(wsi, overlays.level)
    scale = np.array([base_mpp_x / mask_mpp_x, base_mpp_y / mask_mpp_y])

    nuclei = reader.nuclei(slide.slide_id)  # index order (sorted by id)
    assert len(nuclei) == len(c), f"Nuclei mismatch: {slide.slide_id}"
    values = np.ceil(255 * np.clip(c / vmax, 0.0, 1.0)).astype(np.uint8)

    mask = Image.new("L", mask_size, color=0)
    canvas = ImageDraw.Draw(mask)
    for polygon, value in zip(nuclei["polygon"].to_numpy(), values, strict=True):
        if value == 0:
            continue
        points = rearrange(np.asarray(polygon), "(n c) -> n c", c=2) * scale
        canvas.polygon([tuple(p) for p in points], fill=int(value), outline=int(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_big_tiff(
        image=pyvips.Image.new_from_array(np.array(mask)),
        path=path,
        mpp_x=mask_mpp_x,
        mpp_y=mask_mpp_y,
        tile_width=overlays.tile_px,
        tile_height=overlays.tile_px,
    )


@with_cli_args(["+xai=session_material"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    for key, value in setup_precision(config.matmul_precision).items():
        logger.experiment.set_tag(logger.run_id, key, value)

    concept_dir = Path(config.concept_dir)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = load_activations(config.activations_dir, "test")
    slides = pd.read_parquet(config.slides)
    slides = (
        slides[slides["split"] == "test"]
        .sort_values("offset")
        .set_index("slide_id", drop=False)
    )
    summary = pd.read_csv(concept_dir / "concept_summary.csv")
    regions = pd.read_parquet(concept_dir / "regions.parquet")
    top = pd.read_parquet(concept_dir / "top_nuclei.parquet")
    codes = np.load(concept_dir / "codes_test.npy", mmap_mode="r")
    vmax = summary.set_index("concept")["color_scale_max"]

    letters, example = assign_letters(summary, config)
    reader = SlideReader(slides.reset_index(drop=True))

    key_rows, region_rows, pages = [], [], []
    pages.append(
        text_page(
            "Concept review: instructions",
            INSTRUCTIONS.format(
                strip=config.strip.num,
                strip_um=config.strip.size_um,
                scale=config.page.scale_bar_um,
            ),
            config,
            table=[
                [textwrap.fill(name, 22), value]
                for (name, _), value in zip(
                    RESPONSE_COLUMNS[:6], EXAMPLE_ROW[:6], strict=True
                )
            ],
        )
    )
    order = list(letters.itertuples(index=False))
    order = [order[-1], *order[:-1]]  # the worked example first
    for entry in order:
        concept = entry.concept
        rng = np.random.default_rng([config.session_seed, concept])
        concept_regions = regions[regions["concept"] == concept]
        concept_regions = concept_regions.iloc[rng.permutation(len(concept_regions))]
        strip = top[top["concept"] == concept].nsmallest(config.strip.num, "rank")
        c = np.asarray(codes[:, concept])
        title = (
            "Worked example (response row: see the instructions)"
            if entry.section == "example"
            else f"Concept {entry.letter}"
        )
        pages.append(
            render_page(
                title,
                concept_regions,
                strip,
                c,
                float(vmax[concept]),
                reader,
                test,
                slides,
                config,
            )
        )

        stats = summary.set_index("concept").loc[concept]
        key_rows.append(
            {
                "letter": entry.letter,
                "section": entry.section,
                "concept": concept,
                "logit_weight": stats["logit_weight"],
                "sign": int(stats["logit_weight_sign"]),
                "carcinoma_share_active": stats["carcinoma_share_active"],
                "activation_frequency": stats["activation_frequency"],
                "share_abs_phi": stats["share_abs_phi"],
                "stability": "unstable" if stats["unstable"] else "stable",
                **{
                    col: stats[col]
                    for col in summary.columns
                    if col.startswith("best_match_cosine")
                },
            }
        )
        for panel, region in enumerate(
            concept_regions.itertuples(index=False), start=1
        ):
            region_rows.append(
                {
                    "letter": entry.letter,
                    "item": f"region {panel}",
                    "region": region.region,
                    "slide_id": region.slide_id,
                    "x0_um": region.x0_um,
                    "y0_um": region.y0_um,
                }
            )
        for position, nucleus in enumerate(strip.itertuples(index=False), start=1):
            region_rows.append(
                {
                    "letter": entry.letter,
                    "item": f"nucleus {position}",
                    "region": "top_nucleus",
                    "slide_id": nucleus.slide_id,
                    "nucleus_id": nucleus.nucleus_id,
                    "x0_um": nucleus.centroid_x_um,
                    "y0_um": nucleus.centroid_y_um,
                }
            )
        print(f"page {entry.letter}: concept {concept} ({entry.section})")

    pages[0].save(
        out_dir / "deck.pdf",
        save_all=True,
        append_images=pages[1:],
        resolution=config.page.dpi,
        quality=config.page.jpeg_quality,
    )
    pd.DataFrame(key_rows).to_csv(out_dir / "key.csv", index=False)
    pd.DataFrame(region_rows).to_csv(out_dir / "regions_key.csv", index=False)

    response_letters = [
        e.letter for e in letters.itertuples(index=False) if e.section != "example"
    ]
    response_sheet_pdf(response_letters, out_dir / "response_sheet.pdf", config)
    response_sheet_xlsx(response_letters, out_dir / "response_sheet.xlsx")

    overlay_rows = []
    for concept in config.overlays.concepts:
        for role, slide in overlay_slides(
            concept, codes, test, slides, config.overlays
        ):
            sl = slice(int(slide.offset), int(slide.offset + slide.n_nuclei))
            path = (
                out_dir
                / "overlays"
                / f"concept_{concept:02d}"
                / f"{slide.slide_id}.tiff"
            )
            write_overlay(
                slide,
                np.asarray(codes[sl, concept]),
                float(vmax[concept]),
                reader,
                config.overlays,
                path,
            )
            overlay_rows.append(
                {
                    "concept": concept,
                    "role": role,
                    "slide_id": slide.slide_id,
                    "n_nuclei": int(slide.n_nuclei),
                    "carcinoma_share": slide.carcinoma_share,
                    "mean_c": slide.mean_c,
                    "color_scale_max": float(vmax[concept]),
                    "level": config.overlays.level,
                    "path": str(path.relative_to(out_dir)),
                }
            )
            print(f"overlay concept {concept} {role}: {slide.slide_id}")
    pd.DataFrame(overlay_rows).to_csv(
        out_dir / "overlays" / "overlays.csv", index=False
    )

    logger.log_hyperparams(
        {"example_concept": example, "session_seed": config.session_seed}
    )
    logger.log_artifacts(str(out_dir), "session")


if __name__ == "__main__":
    main()
