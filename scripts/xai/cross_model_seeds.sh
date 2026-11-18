#!/usr/bin/env bash
# Concepts across training seeds (xai/cross_model_concepts.py): for each other seed of the
# analysed model, prepare its XAI folder, extract h and logits for the analysed model's
# validation and test nuclei, train the l1 SAEs (k = 16, seeds 0-2, the thesis λ grid and
# selection), evaluate them, then match their concepts to A-H.
#
# Runs on a notebook with the public cluster's projects mounted read-only at
# /mnt/public_projects (nuclei, labels, the analysed model's XAI folder) and a writable
# /mnt/projects. Needs the public MLflow; asks for the password if it is not in the
# environment and then continues in the background (log: $LOG). Finished steps are skipped
# on a rerun (the extraction resumes per slide).
set -euo pipefail

REFERENCE=c8f268536be8416f9fe395eea95c458e
RUNS=(5401eebe1212414d99b19e903a22db31 084da1a0f82748049381c84d87b74462)
PUBLIC=/mnt/public_projects/nuclei_based_wsi_analysis
OUT=/mnt/projects/nuclei_based_wsi_analysis/xai
LOG=${LOG:-$OUT/cross_model_seeds.log}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

export MLFLOW_TRACKING_URI=https://mlflow.rationai.cloud.e-infra.cz/
if [[ -z "${CROSS_MODEL_CHILD:-}" ]]; then
    if [[ -z "${MLFLOW_TRACKING_PASSWORD:-}" ]]; then
        read -rsp "MLflow password for ${MLFLOW_TRACKING_USERNAME}: " MLFLOW_TRACKING_PASSWORD
        echo
        export MLFLOW_TRACKING_PASSWORD
    fi
    mkdir -p "$OUT"
    CROSS_MODEL_CHILD=1 nohup bash "$0" >>"$LOG" 2>&1 &
    echo "running in the background (pid $!), log: $LOG"
    exit 0
fi

cd "$REPO"
echo "=== $(date -Is) start, commit $(git rev-parse --short HEAD)"
run() { echo "--- $(date -Is) $*"; uv run "$@"; }

for RUN in "${RUNS[@]}"; do
    DIR=$OUT/$RUN
    [[ -f $DIR/source.json ]] || run python -m xai.cross_model_prepare "$RUN" \
        "$PUBLIC/xai/$REFERENCE" "$DIR" "/mnt/projects/nuclei_based_wsi_analysis=$PUBLIC"
    CHECKPOINT=$(python -c "import json; print(json.load(open('$DIR/source.json'))['checkpoint'])")

    [[ -f $DIR/activations/activations_summary.json ]] || run python -m nuclei_graph \
        '~experiment/modeling/training' '+experiment/modeling/prediction=nuclei_level/radboud_activations' \
        checkpoint="'$CHECKPOINT'" mlflow_parent_run_id="$RUN" \
        data.metadata.predict="$DIR/slides_predict.parquet" \
        dataset.providers.radboud.supervision.annotation="$PUBLIC/nuclei_supervision/PANDA/Radboud/annotation" \
        dataset.paths.nuclei.patches="$PUBLIC/nuclei_patches/PANDA_45deg_18um_40px"

    [[ -f $DIR/activations/check.json ]] || run python -m xai.check_activations \
        +experiment=xai/radboud parent_run_id="$RUN"
    [[ -f $DIR/sae/models.csv ]] || run python -m xai.train_sae \
        +experiment=xai/radboud parent_run_id="$RUN" 'num_concepts=[16]'
    [[ -f $DIR/completeness.csv ]] || run python -m xai.evaluate_sae \
        +experiment=xai/radboud parent_run_id="$RUN"
done

run python -m xai.cross_model_concepts +experiment=xai/radboud parent_run_id="$REFERENCE" \
    reference.dir="$PUBLIC/xai/$REFERENCE" \
    "models=[{name: '5401eebe', run_id: ${RUNS[0]}, dir: $OUT/${RUNS[0]}},{name: '084da1a0', run_id: ${RUNS[1]}, dir: $OUT/${RUNS[1]}}]"
echo "=== $(date -Is) done"
