# DExOmics revision analysis (`analysis/`)

Scripts behind the 2026-09 revision of *An Interpretable Multi-Omics Integration Framework for Explaining
Differential Expression in Cancer* (TCGA-LIHC primary analysis, ICGC LIRI-JP external replication, frozen
TCGA-CESC confirmation). They regenerate every number and figure of the revised cancer-specific part of the
manuscript; the pan-cancer model and the survival analyses are unchanged and come from the original
repository code (`scripts/`).

The scripts were written during the revision (2026-09-18 .. 2026-09-21) and run on a Mac Studio (M2 Ultra,
macOS 27, Python 3.12.13, PyTorch on the MPS backend). This README, `dexo_paths.py`, `run_all.sh`,
`smoke_test.sh`, `requirements*.txt` and the scripts `22b`, `22c`, `29b` were added on 2026-10-29 to make the
set runnable outside that machine; the analysis code itself was not changed (only the path handling, a CUDA
fallback, and one additional bootstrap pair in `06`).

## 1. Quick start

```bash
# 1. environment (any OS; macOS needs `brew install libomp` for xgboost)
python3.12 -m venv ~/Data/DExOmics/venv && ~/Data/DExOmics/venv/bin/pip install -r analysis/requirements.txt
#    R >= 4.3 with Bioconductor limma + edgeR is needed once, by 12_label_decomposition.py (calls 11_voom_labels.R)
#    conda users: conda env create -f analysis/environment.yml

# 2. data (~15 GB, several hours depending on the GDC download speed; see section 3)
bash analysis/run_all.sh data

# 3. everything else, in dependency order (about 12-14 h on an M2 Ultra; see the table in section 4)
bash analysis/run_all.sh all

# smoke test: the same stages with --quick (few epochs, small grids; minutes to ~1 h)
bash analysis/run_all.sh --quick all
```

Every script is independent (no package to install); run one with
`python analysis/<script>.py [flags]` from any working directory. All paths are resolved by
`analysis/dexo_paths.py`:

| what | how to set | default |
|---|---|---|
| **data root** (raw downloads + every result) | `--data-root DIR` on any script, or `DEXOMICS_DATA` | `~/Data/DExOmics` |
| **project root** (optional figure copies to `notes/figures`, TeX folders) | `--proj-root DIR`, or `DEXOMICS_PROJ` | parent of `analysis/` |

Keep the data root outside the git repository (it holds ~20 GB). `run_all.sh --data-root DIR` sets it for
every step and logs each step to `<data root>/results/logs/`.

## 2. Layout of the data root

```
<data root>/
  venv/                         Python environment (optional location; run_all.sh looks here first)
  data/                         processed data distributed with the original paper (Google Drive, 4 tar.gz)
    pancan_data/                GENCODE v26 GTF, pancanatlas_expid.txt (pan-cancer median expression; pre-training target)
    promoter_features/          encode_hepg2_promoter.pkl / encode_hela_promoter.pkl (+ *_feature_name.txt.gz): 550 (62) TFs x 30 bins
    rna_features/               encode_hepg2_rna.pkl / encode_hela_rna.pkl: 86 (31) RBPs x variable bins
    TCGAprocessed/{LIHC,CESC}/  DEG_info.csv (original labels, used only for the label audit), mapped_geneID.csv
  gdc/{LIHC,CESC}/              GDC open-access files (manifest_*.csv + star/ maf/ cnv/ methy/; star_counts -> star)
  gdc/annot/methy_probe_annot.tsv.gz   HM450 hg38 probe annotation (zhou-lab InfiniumAnnotationData, GENCODE v36)
  icgc/LIRI-JP/                 ICGC 25K open-tier release 28 (exp_seq, specimen, sample, donor, headers/)
  results/
    features/                   gene-level feature tables (00, 02-05, 13)
    de_lihc/ de_cesc/ de_liri/  DESeq2 (pydeseq2) log2FC and labels (01, 08) + voom approximation and label decomposition (11-12)
    exp1_nested_cv[_cesc]/      chromosome-blocked nested CV of the non-deep baselines (06, 06b, 16, 17)
    exp2_liri/                  external validation on LIRI-JP (09)
    exp3_direction/             direction-specific diagnosis and calibration probe (10, 15)
    tensors/                    positional tensors prom [N,550,30], rna [N,86,32] (19; 1.4 GB)
    exp4_dl/                    positional baselines, CNNs, seed averages, pre-training, CESC confirmation, ensembles, classification (20-29b)
    exp5_interp/                SHAP / expected-gradients attributions (30, 32; 1.5 GB)
    figures/                    all figure files (18, 24, 26, 31, 33, 34)
    logs/                       run_all.sh logs
```

## 3. Data acquisition (`run_all.sh data`)

| source | what | how | size |
|---|---|---|---|
| Google Drive folder `1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk` (link-shared, from the original README) | `pancan_data`, `promoter_features`, `rna_features`, `TCGAprocessed` tar.gz | `gdown --folder ...` then extract into `data/` | 200 MB |
| GDC API (open access), TCGA-LIHC and TCGA-CESC | STAR counts, masked somatic MAF, ASCAT3 gene-level CNV, 450k SeSAMe beta values | `gdc/gdc_download_all.sh LIHC` / `CESC` (manifests via `gdc_manifest*.py`, 6 parallel downloads, resumable) | 14 GB |
| zhou-lab InfiniumAnnotationData | `HM450.hg38.manifest.gencode.v36.tsv.gz` -> `gdc/annot/methy_probe_annot.tsv.gz` (columns probeID, genesUniq, CGIposition) | curl | 18 MB |
| ICGC 25K open bucket (`object.genomeinformatics.org/icgc25k-open`, release 28) | LIRI-JP exp_seq / specimen / sample / donor | `07_liri_download.py` | 170 MB |

The files used for the manuscript were downloaded on 2026-09-20 (GDC) and 2026-09-20 (ICGC). GDC data
releases can change file sets slightly; the manifests actually used (424 STAR / 372 MAF / 368 CNV / 430
methylation files for LIHC; 309 / 289 / 294 / 312 for CESC) are kept with the frozen results.

## 4. Pipeline

Stages of `run_all.sh`, in dependency order. "Flags" are the arguments used for the manuscript results;
"quick" marks scripts that accept `--quick` (smoke test). Times are approximate wall-clock on the M2 Ultra
(MPS); the CNN steps are 3-10x slower on CPU.

### `features` (minutes)

| script | input | output (`results/features/`) | time |
|---|---|---|---|
| `00_gene_annotation.py` | GENCODE v26 GTF | `gene_annotation_gencode26.parquet` | 1 min |
| `02_binding_features.py` | HepG2 promoter/RNA pkl | `binding_gene_level.parquet` (636 columns + bin counts), `gene_index.csv` | 1 min |
| `03_cnv_features.py` | `gdc/LIHC/cnv` | `cnv_gene_level.parquet` (Amp_rate, Del_rate, mean_cn, frac_na), `cnv_patients.csv` | 2 min |
| `04_mutation_features.py` | `gdc/LIHC/maf` | `mutation_gene_level.parquet` (9 original classes + extended `mutall__`), `mutation_patients.csv` | 1 min |
| `05_methylation_features.py` | `gdc/LIHC/methy`, probe annotation | `methylation_gene_level.parquet` (`methy_sig__*` original-style, `methy_all__*` label-free), `methylation_probe_stats.parquet` | 10 min |
| `13_cesc_features.py` | HeLa pkl, `gdc/CESC/*` | `binding_gene_level_hela.parquet`, `{cnv,mutation,methylation}_gene_level_cesc.parquet` (`--only binding,cnv,...` to redo parts) | 10 min |

### `de` (minutes; R needed for 12)

| script | flags | output | time |
|---|---|---|---|
| `01_de_tcga_only.py` | (`CANCER=LIHC` default) | `de_lihc/`: `counts_matrix.parquet`, `de_results.csv` (log2FC, padj, label UP/DOWN/NonDEG), `summary.json`, `crosstab_vs_current_labels.csv` | 1 min |
| `01_de_tcga_only.py` | `CANCER=CESC` (env) | `de_cesc/` (same files; exploratory, only 3 normals) | 1 min |
| `08_liri_logfc.py` | | `de_liri/`: `counts_matrix.parquet`, `de_results.csv`, `summary.json` | 1 min |
| `12_label_decomposition.py` | `--force` (recompute voom labels) | writes voom inputs, calls `Rscript 11_voom_labels.R` (limma-voom approximation of the original labels), then `de_lihc/label_decomposition.json`, `decomposition_method_crosstab.csv`, `voom_labels.csv` | 1 min |

### `exp1` (2-4 h)

| script | flags / quick | output (`results/exp1_nested_cv[_cesc]/`) | time |
|---|---|---|---|
| `06_nested_cv_baselines.py` | quick; `--allow-partial` | `folds.csv` (chromosome groups, fixed), `oof_predictions.parquet`, `metrics.csv`, `metrics_bootstrap_spearman.csv`, `bootstrap_differences.json`, `run_meta.json` | 1-2.5 h |
| `06b_mlp.py` | quick | `oof_mlp.parquet`, `metrics_mlp.csv` (aggregate-feature MLP, same folds) | 20 min |
| `06_nested_cv_baselines.py` | `CANCER=CESC`, quick | `exp1_nested_cv_cesc/` (same files) | 1 h |
| `16_loco_sensitivity.py` | | `loco_sensitivity.csv` (leave-one-chromosome-out) | 5 min |
| `17_importance_stability.py` | | `importance_by_fold.parquet`, `importance_stability.json` | 2 min |

### `exp2` (minutes)

| script | flags | output | time |
|---|---|---|---|
| `09_liri_external_validation.py` | | `exp2_liri/metrics.csv`, `bootstrap_ci.csv`, `bootstrap_differences.json` | 1 min |
| `10_direction_diagnosis.py` | default caliper 0.5; also `--caliper 0.3`, `--caliper 1.0` (`--model "all\|xgboost"` to restrict) | `exp3_direction/diagnosis[_caliperX].csv`, `matched_pairs.csv`, `balance.csv` | 1 min each |
| `15_calibration_probe.py` | | `exp3_direction/calibration_probe.csv` | 1 min |

### `exp4` (about 8 h; the deep-learning block)

| script | flags / quick | output (`results/exp4_dl/` unless noted) | time |
|---|---|---|---|
| `19_positional_tensors.py` | | `tensors/prom.npy`, `rna.npy`, `genes.csv` (gene order, fold, log2FC), `meta.json` | 2 min |
| `20_positional_baselines.py` | quick; and `--with-agg` | `oof_positional[_agg].parquet`, `metrics_positional[_agg].csv` (flattened positional features x XGBoost/ridge) | 2 h / 1.5 h |
| `21_dl_positional.py` | quick; `--arch all` default, `--inner-rot 4`; then `--stage2` (positional embedding), `--hybrid` | `oof_dl[_stage2\|_hybrid].parquet`, `metrics_dl*.csv`, `dl_training_log*.json` | 2 h / 25 min / 25 min |
| `22_dl_seed_robustness.py` | | `oof_dl_seedavg.parquet` (stage-2 CNN, seeds 1-5; fold-wise configs from the stage-2 log) | 30 min |
| `22b_xgb_seed_robustness.py` | quick | `oof_xgb_seedavg.parquet`, `xgb_seedavg_out.txt` (XGBoost, seeds 1-5, fold-wise params from 06) | 10 min |
| `25_original_arch_pe.py` | quick | `oof_orignet.parquet`, `metrics_orignet.csv`, `orignet_training_log.json` (original ConcatedNet regression version, with/without PE) | 1 h |
| `27_pretrain_cnn.py` | `--seeds 3`, quick | `oof_dl_pretrained.parquet`, `pretrain_log.json` (pan-cancer pre-training then fine-tuning; **primary CNN**) | 40 min |
| `23_cesc_positional.py` | (no quick mode) | `cesc_tensors/`, `oof_dl_cesc.parquet`, `ensemble_cesc.json` (frozen-rule confirmation, stage-2 CNN) | 15 min |
| `28_cesc_pretrained_confirm.py` | quick | `oof_dl_cesc_pretrained.parquet`, `ensemble_cesc_pretrained.json` (frozen-rule confirmation, pre-trained CNN) | 20 min |
| `22c_ensemble_eval.py` | | `ensemble_lihc.json` (all single-model / ensemble points and paired chromosome-bootstrap CIs, internal and on LIRI-JP), `fig5_cells.json` | 5 min |
| `29_direct_classification.py` | quick; runs `--part xgb/cnn/eval` as subprocesses | `cls_prob_{xgb,cnn}.npy`, `classification_direct.csv`, `oof_classification.parquet` | 45 min |
| `29b_classification_from_scores.py` | | `classification_from_scores.csv`, `classification_paired_diff.json` (regression-score thresholds vs direct classifiers) | 2 min |

### `exp5` (about 1 h)

| script | flags / quick | output (`results/exp5_interp/`) | time |
|---|---|---|---|
| `30_interpretability.py` | quick; parts `xgb`, `xgb-nocnv`, `cnn`, `analysis` as subprocesses | `shap_xgb_oof.npy` (+ cols/per-seed csv), `eg_{prom,rna,omics}_oof.npy`, `eg_meta.json`, `pos_profile_*.npy`, `crossmodel_agreement.csv`, `interp_summary.json`, `xgb_nocnv.json` | 45 min |
| `32_cesc_interp.py` | quick; parts `xgb`, `cnn`, `analysis` | `cesc_*` (same structure) + `cesc_interp_summary.json`, `cesc_shared_factors.json` | 10 min |

### `figures` (minutes)

| script | figures (`results/figures/`, PDF + PNG) | manuscript |
|---|---|---|
| `33_workflow_fig.py` | `fig0_workflow` | Fig 1 |
| `18_figures.py` | `fig1_label_decomposition`, `fig2_nested_cv_forest`, `fig3_liri_external`, `fig4_direction_diagnosis` | Fig 4, 5, 7, 8 |
| `24_fig5.py` | `fig5_dl_complementarity` | Fig 6 |
| `31_paper_figures.py` | `fig6_classification`, `fig7_interpretability`, `figS1_model_landscape`, `figS2_pretraining`, `figS3_robustness`, `figS4_positional` | Fig 9, 11, supplementary |
| `26_fig_arch.py` | `fig_s_architecture` | supplementary |
| `34_cesc_interp_fig.py` | `figS_cesc_interpret` | Supplementary Fig. S4 |

Pan-cancer figures (manuscript Fig 2, 3, 10) are the original JPGs. The file-name numbers differ from the
manuscript figure numbers; the full mapping is in `00_README_revision_en.md` section 5 of the revision package.
Figure scripts also drop PNG copies in `<project root>/notes/figures/` and, when the TeX folders exist,
PDFs into `2024_Yi_Bioinformatics_TeX/{fig,supple_fig}/`.

## 5. Reproducibility notes

* **Seeds and folds are fixed in the scripts.** Outer folds are chromosome groups written once by `06`
  (`folds.csv`) and reused everywhere; `rng = default_rng(0)` and `torch.manual_seed(0)` throughout;
  multi-seed runs use seeds 1-5 (XGBoost) and 1-3 or 1-5 (CNN).
* **Frozen hyper-parameters are transcribed between scripts.** `22`, `27`, `28`, `29`, `30`, `32` carry the
  fold-wise CNN configurations chosen by the inner CV of `21 --stage2` (LIHC) and `23` (CESC) as `FOLD_CFG`
  constants, `22b`/`30`/`32` carry the fold-wise XGBoost parameters chosen by `06` as `XPARAMS`/`XPAR`, and `32`
  carries the pretext epochs of `28` (`PRE_EP`). This was done on purpose (no re-selection after the rules were
  frozen). If you rerun `06`, `21 --stage2` or `23` and the inner-CV choices change, update those constants from
  the new logs (`run_meta.json` / `metrics.csv` `params` column, `dl_training_log_stage2.json`, `run_cesc_dl.log`).
* **Hardware.** CNNs were trained on Apple MPS; the scripts fall back to CUDA, then CPU. Results on another
  backend reproduce the reported values to about two decimals of Spearman rho, not bit-for-bit. XGBoost
  (`hist`, fixed seeds) is deterministic for a given library version.
* **`22b`, `22c`, `29b` re-implement analyses that were run interactively** on 2026-09-21 (XGBoost seed
  average, ensemble bootstrap tables, score-threshold classification). Point estimates are deterministic and
  identical; bootstrap CIs are regenerated from seed 0 and can differ from the manuscript values in the third
  decimal. `31` reads the regenerated paired difference (`classification_paired_diff.json`) and `24` the
  regenerated ensemble differences (`ensemble_lihc.json`, `ensemble_cesc_pretrained.json`) when present.
* **Threshold calibration in `29b`** uses, for fold k, the OOF predictions of the other four folds. Those
  predictions come from models trained on data including fold k, so the calibration is not strictly nested;
  this matches the procedure used for the manuscript and is kept for comparability (see the script header).
* **Multi-process scripts.** `29`, `30`, `32` run their XGBoost and PyTorch parts in separate processes
  (`--part`) because loading xgboost and torch in one process deadlocks libomp on macOS. `--data-root` is
  propagated to the sub-processes through `DEXOMICS_DATA`.
* **Original-label files** (`data/TCGAprocessed/*/DEG_info.csv`) are used only to audit the previous labels
  (`01`, `12`); no model is trained on them.
* **Verification (2026-10-29, Mac Studio, `smoke_test.sh`).** Re-running `22b` on the real intermediates
  reproduced the XGBoost seed results exactly (per-seed 0.3469/0.3542/0.3523/0.3464/0.3488, average 0.357);
  `22c` reproduced every point estimate exactly and the manuscript's paired CIs to within 0.001-0.002
  (e.g. ens_pre - XGBoost: internal +0.029 [0.021, 0.037], LIRI-JP +0.029 [0.018, 0.041]; pretrained - random
  init +0.014 [0.005, 0.023]); `29b` reproduced all threshold-free metrics exactly and macro-F1 / recalls to
  within 0.002 / 0.02 (threshold grid differs from the interactive run); the direct-vs-score paired difference came
  out as +0.033 [0.021, 0.044] (macro-F1) and +0.078 [0.061, 0.098] (PR-AUC DOWN) against +0.031 / +0.077 in the
  manuscript. All figure scripts ran. The staged `--quick` run then completed features, de (incl. R), exp1 and
  exp2 without errors.
* `smoke_test.sh` creates a fresh test root (inputs symlinked, small results copied) and runs `22b/22c/29b`
  + the figure scripts on the real intermediates, then `run_all.sh --quick` for every stage, without touching
  the real data root.

## 6. Files

```
analysis/
  dexo_paths.py        path resolution (imported by every script)
  run_all.sh           staged driver (data | features | de | exp1 | exp2 | exp4 | exp5 | figures | all; --quick)
  smoke_test.sh        isolated end-to-end check
  requirements.txt     direct dependencies (pinned to the versions used)
  requirements-lock.txt  full freeze of the environment that produced the results
  environment.yml      conda equivalent (adds R + limma/edgeR)
  gdc/                 GDC manifest + download helpers (gdc_download_all.sh LIHC|CESC)
  00..34, 22b, 22c, 29b   analysis scripts (header docstring of each = inputs, outputs, design, flags)
```
