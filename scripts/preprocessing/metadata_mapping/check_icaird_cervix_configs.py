"""Checks the iCAIRD training and testing configs against the mapping tables and labels.

Usage (from the repo root, public MLflow credentials in the environment):
    PYTHONPATH=. python scripts/preprocessing/metadata_mapping/check_icaird_cervix_configs.py [run id]
The run id defaults to the mapping run in configs/data/sources/icaird_cervix.yaml.

1. Training config (configs/ml.yaml -> training/nuclei_level/icaird_cervix): resolved data section;
   train / valid through BaseDataModule.load_train_val_dfs (the training code path); no low grade in
   train or valid; validation supervised as training (the same sparse strategy): on a positive slide
   exactly the label-1 nuclei are supervised (as 1), on a negative slide every nucleus (as 0).
2. Testing config (testing/nuclei_level/icaird_cervix): resolved data section; the main test table as
   the test stage loads it (no low grade); composition (re-annotated positives by subcategory,
   normal / inflammation negatives) and labelled nuclei per group; the test supervision built by
   BaseDataModule.prepare_supervision: on a positive slide exactly the nuclei inside the
   pathologist's regions are supervised (all as 1) and every other nucleus is ignored; on a negative
   slide every nucleus is supervised as 0. Positive slides whose regions are not exported yet are
   reported as pending. The dense `annotation` mode must refuse the re-annotation labels.
3. The separate low-grade test table (`dataset.mlflow_uris.metadata.test_low_grade`, not read by any
   config by default): low-grade test slides only, none in train / valid / test, all negative, every
   nucleus supervised as 0 by the test strategy.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from mlflow.artifacts import download_artifacts
from omegaconf import OmegaConf

from nuclei_graph.data.datamodules.base import BaseDataModule
from nuclei_graph.data.supervision import SupervisionStrategy


REPO = str(Path(__file__).resolve().parents[3])
POSITIVE = {"high_grade", "malignant"}
LOCAL = os.environ.get(
    "ICAIRD_LOCAL_TABLES"
)  # testing only: tables read from this folder


def config(experiment: str | None, run_id: str | None) -> OmegaConf:
    overrides = []
    if experiment is not None:
        overrides.append(f"+experiment/modeling/testing={experiment}")
    if run_id is not None:
        base = f"mlflow-artifacts:/37/{run_id}/artifacts/icaird_cervix"
        overrides += [
            f"dataset.mlflow_uris.metadata.{s}={base}/slides_mapping_{s}.parquet"
            for s in ("train", "valid", "test", "test_low_grade")
        ]
    with initialize_config_dir(config_dir=f"{REPO}/configs", version_base=None):
        if experiment is None:  # ml.yaml selects the iCAIRD training experiment
            return compose(config_name="ml", overrides=["mode=fit", *overrides])
        return compose(
            config_name="ml",
            overrides=[
                "~experiment/modeling/training",
                "checkpoint=unused",
                *overrides,
            ],
        )


def show(cfg: OmegaConf, title: str) -> None:
    d = cfg.data
    print(f"== resolved {title} config (data)")
    print(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "metadata": OmegaConf.to_container(d.metadata, resolve=True),
                    "split_size": d.get("split_size"),
                    "supervision": OmegaConf.to_container(d.supervision, resolve=True),
                }
            )
        )
    )


def datamodule(cfg: OmegaConf) -> SimpleNamespace:
    dm = SimpleNamespace(
        metadata_uris_cfg=cfg.data.metadata,
        split_size=cfg.data.get("split_size"),
        load_df=lambda uri, cols=None: pd.read_parquet(
            Path(LOCAL, Path(uri).name) if LOCAL else download_artifacts(uri),
            columns=cols,
        ),
        get_train_val_dfs=None,
    )
    dm.get_carcinoma_map = lambda df: BaseDataModule.get_carcinoma_map(dm, df)
    return dm


def composition(name: str, df: pd.DataFrame) -> None:
    t = df.groupby(["category", "is_carcinoma"]).agg(
        slides=("slide_id", "size"), nuclei=("nuclei_count", "sum")
    )
    print(f"{name}: {len(df)} slides, {int(df['nuclei_count'].sum()):,} nuclei")
    print(t.to_string(), "\n")


run_id = sys.argv[1] if len(sys.argv) > 1 else None

# 1. training ---------------------------------------------------------------------------------
train_cfg = config(None, run_id)
show(train_cfg, "training")
dm = datamodule(train_cfg)
train, valid = BaseDataModule.load_train_val_dfs(dm)
composition("train", train)
composition("valid", valid)
for name, df in (("train", train), ("valid", valid)):
    assert not df["category"].eq("low_grade").any(), f"low grade in {name}"
    assert (df["is_carcinoma"] == df["category"].isin(POSITIVE)).all()
sup_cfg = train_cfg.data.supervision
assert sup_cfg.eval_strategy == sup_cfg.train_strategy, "valid supervised unlike train"
assert sup_cfg.eval_strategy.mode == "annotation_sparse"
assert sup_cfg.eval_strategy.get("ignore_dir") is None
strategy = instantiate(sup_cfg.eval_strategy)
sup = BaseDataModule.prepare_supervision(dm, valid, strategy.paths, strategy)
v_annot = Path(strategy.paths["annot_dir"])
v_pos = v_neg = 0
for row in valid.itertuples():
    ns = sup.supervision_map[row.slide_id].nuclei_supervision
    n = int(row.nuclei_count)
    mask, targets = ns.get_sup_mask(n), ns.get_targets(n)
    if row.is_carcinoma:
        labels = pd.read_parquet(v_annot / f"{row.slide_id}.parquet")["annot_label"]
        assert len(labels) == n and (mask.numpy() == (labels.to_numpy() == 1)).all()
        assert (targets[mask] == 1).all()
        v_pos += int(mask.sum())
    else:
        assert mask.all() and (targets == 0).all()
        v_neg += n
print(
    f"OK train {len(train)} / valid {len(valid)} slides, no low grade; valid supervised as training "
    f"(annotation_sparse): {v_pos:,} positive + {v_neg:,} negative nuclei, the rest ignored\n"
)

# 2. testing ----------------------------------------------------------------------------------
test_cfg = config("nuclei_level/icaird_cervix", run_id)
show(test_cfg, "testing")
assert test_cfg.data.supervision.eval_strategy.mode == "annotation_sparse"
assert test_cfg.data.supervision.eval_strategy.get("ignore_dir") is None
tdm = datamodule(test_cfg)
test = tdm.load_df(test_cfg.data.metadata.test)
assert not test["category"].eq("low_grade").any(), "low grade in the main test table"
assert not set(test["slide_id"]) & (set(train["slide_id"]) | set(valid["slide_id"]))
assert (test["is_carcinoma"] == test["category"].isin(POSITIVE)).all()
annot_dir = Path(test_cfg.data.supervision.eval_strategy.annot_dir)
try:
    SupervisionStrategy("annotation", annot_dir=str(annot_dir))
except ValueError:
    print(f"OK dense `annotation` refuses {annot_dir}")
else:
    raise AssertionError(
        "dense annotation mode accepted the sparse re-annotation labels"
    )

pos = test[test["is_carcinoma"]]
neg = test[~test["is_carcinoma"]]
has_labels = pos["slide_id"].map(lambda s: (annot_dir / f"{s}.parquet").exists())
ready = pd.concat([pos[has_labels], neg])
strategy = instantiate(test_cfg.data.supervision.eval_strategy)
sup = BaseDataModule.prepare_supervision(tdm, ready, strategy.paths, strategy)

rows = []
for row in ready.itertuples():
    ns = sup.supervision_map[row.slide_id].nuclei_supervision
    n = int(row.nuclei_count)
    mask, targets = ns.get_sup_mask(n), ns.get_targets(n)
    if row.is_carcinoma:
        labels = pd.read_parquet(annot_dir / f"{row.slide_id}.parquet")["annot_label"]
        inside = labels.to_numpy() == 1
        assert len(labels) == n, "label file != nuclei"
        assert (mask.numpy() == inside).all(), "supervised != inside the regions"
        assert (targets[mask] == 1).all(), "a supervised nucleus is not positive"
    else:
        assert mask.all() and (targets == 0).all(), "negative slide not all 0"
    rows.append(
        {
            "category": row.category,
            "subcategory": row.subcategory if row.is_carcinoma else "",
            "nuclei": n,
            "labelled": int(mask.sum()),
            "labelled_positive": int((targets[mask] == 1).sum()),
            "ignored": n - int(mask.sum()),
        }
    )
stats = pd.DataFrame(rows)

print("== test composition")
by_sub = pos.groupby(["category", "subcategory"]).size().rename("slides")
print("re-annotated positives by subcategory:\n" + by_sub.to_string())
print(
    f"\npositive slides: {len(pos)} (labels for {int(has_labels.sum())}, "
    f"regions not exported yet for {int((~has_labels).sum())})"
)
groups = stats.assign(
    group=stats["category"].where(
        stats["category"].isin(["normal_inflammation", "low_grade"]),
        "positive (" + stats["subcategory"] + ")",
    )
).groupby("group")
table = groups.agg(
    slides=("nuclei", "size"),
    nuclei=("nuclei", "sum"),
    labelled=("labelled", "sum"),
    labelled_positive=("labelled_positive", "sum"),
    ignored=("ignored", "sum"),
)
table.loc["total"] = table.sum()
print("\nlabelled nuclei per group (slides with labels):\n" + table.to_string())
print(
    f"\nOK test: {int(has_labels.sum())} positive slides supervised only inside the pathologist's "
    f"regions (all 1, the rest ignored); {len(neg)} negative slides fully supervised as 0"
)
if (~has_labels).any():
    print(
        f"PENDING: {int((~has_labels).sum())} re-annotated slides have no labels yet "
        "(re-run preprocessing.reannotation_labels_icaird_cervix after their export)"
    )

# 3. separate low-grade test table -------------------------------------------------------------
lg = tdm.load_df(test_cfg.data.metadata.test_low_grade)
assert (
    len(lg) > 0
    and lg["category"].eq("low_grade").all()
    and not lg["is_carcinoma"].any()
)
assert (lg["nuclei_count"] > 0).all()
assert not set(lg["slide_id"]) & (
    set(train["slide_id"]) | set(valid["slide_id"]) | set(test["slide_id"])
)
sup = BaseDataModule.prepare_supervision(tdm, lg, strategy.paths, strategy)
for row in lg.itertuples():
    ns = sup.supervision_map[row.slide_id].nuclei_supervision
    n = int(row.nuclei_count)
    assert ns.get_sup_mask(n).all() and (ns.get_targets(n) == 0).all()
print(
    f"OK separate low-grade test table: {len(lg)} slides, {int(lg['nuclei_count'].sum()):,} nuclei, "
    "every nucleus supervised as 0; not in train, valid or the main test table"
)
