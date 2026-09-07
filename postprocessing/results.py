"""Loads markdown results tables, fetches AUROC/AUPRC/etc. metrics from MLflow
(via each row's Evaluation-column run link), and writes one tidy CSV per
table to output_dir — no plotting. Write plot functions against the CSVs,
or against load_tables()/TableData directly if importing this module.

Table structure (PE, Attention, k, etc.) is read from the markdown — that's
what labels the config rows. Metric VALUES come directly from the MLflow
run referenced in each row's "Evaluation" column link, not from hand-typed
numbers elsewhere in the table.

Confirmed real MLflow key convention (from actual logged runs):
    test_thresholded/nuclei_AUROC
    test_thresholded/nuclei_AUPRC
    test_thresholded/nuclei_accuracy
    test_thresholded/nuclei_precision
    test_thresholded/nuclei_recall
    test_thresholded/nuclei_specificity
    test_thresholded/nuclei_AUROC_lower_95   / _upper_95
    test_thresholded/nuclei_AUPRC_lower_95   / _upper_95
"""

import json
import re
import sys
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

# --------------------------------------------------------------------------
# MLflow metric-key configuration:
#    Convention: "test_thresholded/{level}_{metric}", e.g.
#    "test_thresholded/nuclei_AUROC", "test_thresholded/nuclei_AUROC_lower_95".
#    Change METRIC_PREFIX if you need tile-/graph-level metrics instead of
#    nuclei-level (e.g. "test_thresholded/graph_").
# --------------------------------------------------------------------------

METRIC_PREFIX = "test_thresholded/nuclei_"

_METRIC_KEY_CANDIDATES: dict[str, list[str]] = {
    "AUROC": [f"{METRIC_PREFIX}AUROC"],
    "AUPRC": [f"{METRIC_PREFIX}AUPRC"],
    "Accuracy": [f"{METRIC_PREFIX}accuracy"],
    "Precision": [f"{METRIC_PREFIX}precision"],
    "Recall": [f"{METRIC_PREFIX}recall"],
    "Specificity": [f"{METRIC_PREFIX}specificity"],
}

_CI_KEY_CANDIDATES: dict[str, list[str]] = {
    "AUROC_lo": [f"{METRIC_PREFIX}AUROC_lower_95"],
    "AUROC_hi": [f"{METRIC_PREFIX}AUROC_upper_95"],
    "AUPRC_lo": [f"{METRIC_PREFIX}AUPRC_lower_95"],
    "AUPRC_hi": [f"{METRIC_PREFIX}AUPRC_upper_95"],
}

# --------------------------------------------------------------------------
# 1. Markdown table extraction
# --------------------------------------------------------------------------

_SEP_RE = re.compile(r"^\|?[\s\-:|]+\|?$")


def extract_tables(md_text: str) -> list[tuple[str, str, list[str]]]:
    """Find every (heading_context, header_line, [data_lines]) markdown table."""
    lines = md_text.splitlines()
    tables: list[tuple[str, str, list[str]]] = []
    
    active_headings: dict[int, str] = {}
    
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            raw_text = line.lstrip("#").strip()
            text = clean_cell(raw_text)
            active_headings[level] = text

        if line.startswith("|") and i + 1 < n:
            sep = lines[i + 1].strip()
            if _SEP_RE.match(sep) and "-" in sep:
                header = lines[i]
                j = i + 2
                data_rows = []
                while j < n and lines[j].strip().startswith("|"):
                    data_rows.append(lines[j])
                    j += 1
                
                sorted_levels = sorted(active_headings.keys())
                if not sorted_levels:
                    combined_heading = "untitled"
                else:
                    deepest_text = active_headings[sorted_levels[-1]].lower()
                    if "k" in deepest_text and "sweep" in deepest_text and len(sorted_levels) >= 2:
                        relevant_levels = sorted_levels[-2:]
                    else:
                        relevant_levels = [sorted_levels[-1]]
                        
                    combined_heading = " - ".join(
                        active_headings[lvl] for lvl in relevant_levels
                    )
                    
                tables.append((combined_heading, header, data_rows))
                i = j
                continue
        i += 1
        
    return tables


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return line.split("|")


# --------------------------------------------------------------------------
# 2. Cell cleaning (label columns only; metric values always come from MLflow)
# --------------------------------------------------------------------------

_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def clean_cell(cell: str) -> str:
    cell = cell.strip()
    cell = _BOLD_RE.sub(r"\1", cell)
    cell = _BR_RE.sub(" ", cell)
    cell = _LINK_RE.sub(r"\1", cell)  # keep link text, drop the URL
    return cell.strip()


# --------------------------------------------------------------------------
# 3. MLflow run-ID extraction and metric fetching
# --------------------------------------------------------------------------

_RUN_ID_RE = re.compile(r"/runs/([a-f0-9]{32})")
_RAW_LINK_URL_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def extract_run_id(raw_cell: str) -> str | None:
    """Pulls an MLflow run_id out of a raw (uncleaned) markdown cell.

    Must run on the RAW cell, before clean_cell() strips the URL away.
    """
    url_match = _RAW_LINK_URL_RE.search(raw_cell)
    if not url_match:
        return None
    run_match = _RUN_ID_RE.search(url_match.group(1))
    return run_match.group(1) if run_match else None


class MlflowMetricFetcher:
    def __init__(self, tracking_uri: str | None, cache_path: Path, use_cache: bool = True):
        import mlflow

        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.cache: dict[str, dict[str, float]] = {}
        if use_cache and cache_path.exists():
            self.cache = json.loads(cache_path.read_text())

    def _save_cache(self) -> None:
        if self.use_cache:
            self.cache_path.write_text(json.dumps(self.cache, indent=2))

    def get_raw_metrics(self, run_id: str) -> dict[str, float]:
        if self.use_cache and run_id in self.cache:
            return self.cache[run_id]
        run = self.mlflow.get_run(run_id)
        metrics = dict(run.data.metrics)
        if self.use_cache:
            self.cache[run_id] = metrics
            self._save_cache()
        return metrics

    def fetch(self, run_id: str) -> dict[str, float]:
        """Returns canonical metric names -> values for one run, e.g.
        {'AUROC': 0.8293, 'AUROC_lo': 0.8201, 'AUROC_hi': 0.8380, ...}.
        Missing metrics/CIs are simply absent from the returned dict.
        """
        raw = self.get_raw_metrics(run_id)
        out: dict[str, float] = {}
        for canon, candidates in _METRIC_KEY_CANDIDATES.items():
            for key in candidates:
                if key in raw:
                    out[canon] = raw[key]
                    break
        for canon, candidates in _CI_KEY_CANDIDATES.items():
            for key in candidates:
                if key in raw:
                    out[canon] = raw[key]
                    break
        return out

# --------------------------------------------------------------------------
# 4. Table -> tidy DataFrame, sourcing metric values from MLflow
# --------------------------------------------------------------------------

_DROP_KEYWORDS = ("training",)  # "evaluation" is kept — it's the run-ID source
_SWEEP_COL_NAMES = {"k", "efd order", "order", "epoch", "epochs"}


@dataclass
class TableData:
    """One parsed+fetched markdown table."""

    index: int
    heading: str
    title: str
    df: pd.DataFrame
    label_cols: list[str]
    metrics_present: list[str]
    sweep_col: str | None = None


def table_to_raw_df(header_line: str, data_lines: list[str]) -> pd.DataFrame | None:
    cols = [clean_cell(c) for c in _split_row(header_line)]
    if len(cols) < 2:
        return None
    rows = []
    for line in data_lines:
        cells = _split_row(line)
        if len(cells) < len(cols):
            cells += [""] * (len(cols) - len(cells))
        rows.append(cells[: len(cols)])
    if not rows:
        return None
    return pd.DataFrame(rows, columns=cols)


def find_evaluation_column(cols: list[str]) -> str | None:
    for c in cols:
        if "evaluation" in c.lower():
            return c
    return None


def classify_label_columns(cols: list[str], eval_col: str | None) -> list[str]:
    label_cols = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in _DROP_KEYWORDS):
            continue
        if c == eval_col:
            continue
        if any(k in cl for k in ("auprc", "auroc", "accuracy", "precision", "recall", "specificity")):
            continue
        label_cols.append(c)
    return label_cols


def detect_sweep_column(label_cols: list[str]) -> str | None:
    for lc in label_cols:
        if lc.strip().lower() in _SWEEP_COL_NAMES:
            return lc
    return None


def build_table_data(
    idx: int, heading: str, header: str, data_lines: list[str], fetcher: MlflowMetricFetcher
) -> TableData | None:
    raw_df = table_to_raw_df(header, data_lines)
    if raw_df is None:
        return None

    cols = list(raw_df.columns)
    eval_col = find_evaluation_column(cols)
    label_cols = classify_label_columns(cols, eval_col)

    out = pd.DataFrame(index=raw_df.index)
    for lc in label_cols:
        out[lc] = raw_df[lc].map(clean_cell)

    title = f"Table {idx} [{heading}] ({', '.join(label_cols) or 'rows'})"

    if eval_col is None:
        print(f"[table {idx}] no Evaluation column found — skipping", file=sys.stderr)
        return TableData(idx, heading, title, out.iloc[0:0], label_cols, [])


    metrics_present: set[str] = set()
    fetched_rows: list[dict[str, float]] = []
    for raw_cell in raw_df[eval_col]:
        run_id = extract_run_id(raw_cell)
        if run_id is None:
            fetched_rows.append({})
            continue
        try:
            m = fetcher.fetch(run_id)
        except Exception as e:  # noqa: BLE001 — surface, don't crash the whole batch
            print(f"[table {idx}] WARNING: failed to fetch run {run_id}: {e}", file=sys.stderr)
            m = {}
        fetched_rows.append(m)
        metrics_present.update(k for k in m if not k.endswith(("_lo", "_hi")))

    for canon in metrics_present:
        out[canon] = [r.get(canon, np.nan) for r in fetched_rows]
        lo_key, hi_key = f"{canon}_lo", f"{canon}_hi"
        if any(lo_key in r for r in fetched_rows):
            out[lo_key] = [r.get(lo_key, np.nan) for r in fetched_rows]
        if any(hi_key in r for r in fetched_rows):
            out[hi_key] = [r.get(hi_key, np.nan) for r in fetched_rows]

    metrics_present_list = sorted(metrics_present)
    if metrics_present_list:
        out = out[out[metrics_present_list].notna().any(axis=1)].reset_index(drop=True)

    sweep_col = detect_sweep_column(label_cols)
    return TableData(idx, heading, title, out, label_cols, metrics_present_list, sweep_col)

def load_tables(md_text: str, fetcher: MlflowMetricFetcher) -> list[TableData]:
    raw_tables = extract_tables(md_text)
    if not raw_tables:
        print("No markdown tables found.", file=sys.stderr)
        return []

    results = []
    for idx, (heading, header, data_lines) in enumerate(raw_tables, start=1):
        print(f"[table {idx}] fetching metrics from MLflow...")
        td = build_table_data(idx, heading, header, data_lines, fetcher)
        if td is None:
            continue
        print(f"[table {idx}] {len(td.df)} rows, columns: {td.label_cols} | metrics: {td.metrics_present}")
        results.append(td)
    return results


def slugify(text: str) -> str:
    """Converts a heading like '"k" Sweep' into a safe filename like 'k_sweep'."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_') or "table"


@with_cli_args(["+postprocessing=results"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    fetcher = MlflowMetricFetcher(
        tracking_uri=config.mlflow_tracking_uri,
        cache_path=Path(config.cache_file),
        use_cache=config.get("use_cache", True),
    )
    text = Path(config.input_path).read_text(encoding="utf-8")
    tables = load_tables(text, fetcher)

    with TemporaryDirectory() as output_dir:
        output_dir_path = Path(output_dir)
        for t in tables:
            if t.df.empty:
                continue

            safe_heading = slugify(t.heading)
            filename = f"{t.index:02d}_{safe_heading}.csv"
            
            csv_path = output_dir_path / filename
            t.df.to_csv(csv_path, index=False)
            print(f"[table {t.index}] wrote {csv_path}")

        logger.log_artifacts(
            local_dir=str(output_dir_path), 
            artifact_path=config.get("mlflow_artifact_path", "tables")
        )

if __name__ == "__main__":
    main()