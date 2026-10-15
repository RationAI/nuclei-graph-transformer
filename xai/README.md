# Concept discovery with sparse autoencoders

Concept-based explanation of a nucleus-level model, following Kůr et al., *A Unifying
Framework of Concept-based XAI with Completeness Guarantees*. The model is split as
f = g ∘ h, where z = h(x) is the output of `final_norm` (∈ R^256) and
g(z) = wᵀz + β is `class_head`. A concept autoencoder with an affine decoder
D(c) = Uᵀc + b_D gives exact concept contributions φ_j = (wᵀu_j) c_j, with
b + Σ_j φ_j = g(D(E(z))) and b = wᵀb_D + β.

All outputs go to `out_dir` (default `${project_path}/xai/<parent_run_id>`, with its own
`README.md`); every step is logged to MLflow as a child run of `parent_run_id`. All
steps use the repo's TF32 matmul precision (`matmul_precision: high`).

| Step | Command | Output |
|---|---|---|
| 1. Nuclei index (CPU) | `python -m xai.prepare_nuclei +experiment=xai/radboud` | `slides.parquet`, `activations/<split>/index.parquet`, `concept_sample.parquet` (probe analysis only) |
| 2. Activations (GPU) | `python -m nuclei_graph '~experiment/modeling/training' '+experiment/modeling/prediction=nuclei_level/radboud_activations'` | `activations/head.npz`, `activations/<split>/{h,logits}.npy`, per-slide `done/` markers (resumable) |
| 2b. Checks | `python -m xai.check_activations +experiment=xai/radboud` | `activations/check.json` |
| 3. Autoencoders (GPU) | `python -m xai.train_sae +experiment=xai/radboud` | `sae/*.pt`, `sae/lambda_grid.csv`, `sae/models.csv`, `sae/holdout_slides.csv` |
| 4. Completeness (GPU) | `python -m xai.evaluate_sae +experiment=xai/radboud` | `completeness.csv`, `completeness_summary.csv`, `stability.csv`, `pca_spectrum.csv` |
| 5. Concept material | `python -m xai.concept_material +experiment=xai/radboud num_concepts=16` | `concepts/k<k>_seed<s>/…` (own `README.md`) |
| 6. Session material | `python -m xai.session_material +experiment=xai/radboud` | `session/`: blinded deck, response sheets, key, whole-slide concept overlays |
| Optional | `python -m xai.precision_check` | test logits in full fp32 next to the TF32 ones |

- **Steps 1–2.** Every nucleus of the validation slides (the training run's own split)
  and the test slides is indexed. `h` is stored as float16 and `logits` as float32
  memory maps, in index row order. The callback checks |class_head(h) − logits| < 1e-4
  before the float16 cast.
- **Step 3.** SAEs (k ∈ {8, 16, 32} × seeds {0, 1, 2} × the λ grid) and PCA baselines
  are trained on all validation nuclei, with minibatches balanced carcinoma:benign = 1:1
  and no slide above about 1% of the samples. Per k and seed, λ is the one with
  L0 ∈ [2, 4] and dead fraction ≤ 5% on the held-out validation slides (10%, stratified
  by Gleason score) with the lowest held-out RE; it is never relaxed. Existing models
  of the grid are reused (`reuse_existing`).
- **Step 4.** RE, FE, MCE (OLS and ridge), L_g·RE, ATE, relative L1 reconstruction
  error of h, explained variance, logit R², correlation, shrinkage and recalibrated
  fidelity on all test nuclei, unweighted, with hard checks (MCE ≤ FE ≤ L_g·RE,
  |ATE − FE| < 1e-4). Test data are used for reporting only.
- **Step 5.** For the chosen k (seed with the lowest held-out RE): concept summary,
  per-nucleus codes and contributions, concept maps (high regions need ≥ 150 nuclei and
  ≥ 60% tissue by the tissue mask) and galleries.
- **Step 6.** A blinded pathologist review: concepts relabeled with letters (fixed
  seed), regions in a random order, no concept numbers, weights, importances,
  activation values or labels in the deck; `key.csv` maps letters back. Overlays are
  written like `nuclei_graph.callbacks.masks` (polygons filled with
  ceil(255 · min(c_j / color_scale_max, 1)), level 0).

Set `XAI_DEVICE=cpu` to run steps 3–5 without a GPU. The jobs in `scripts/xai/` clone
the `xai-sae` branch, so it must be pushed first.
