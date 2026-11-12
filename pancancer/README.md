# DExOmics pan-cancer study (`pancancer/`)

Code of the pan-cancer part of *An Interpretable Multi-Omics Integration Framework for Differential Expression
Prediction in Cancer* (PanCanAtlas and PCAWG). This directory is the original project as recovered from the
author's server (`DEomics_Pancancer/`, imported 2026-11-12 as `pancancer_source/` and renamed here); the scripts
are unchanged. The copy that was published with the 2024 README (`scripts/pan_cancer/`, `scripts/network.py`,
`scripts/utils/`) is archived under `../legacy/scripts_2024/`; see section 7 for the differences.

The study has its own environment (Python 3.8 / PyTorch 1.13 / shap 0.44, `environment.yml`) and does not share
code or an environment with the revised cancer-specific study in `../cancer_specific/`.

## 1. Layout

```
pancancer/
  01_bed_to_RNA_coord.R, 01_run.sh      regulator BED files -> transcript-relative bins (POSTAR3 / GTRD / TargetScan / ENCODE)
  02_to_sparse.py                        binned table -> sparse pickle
  03_expr_process/                       PanCanAtlas / PCAWG expression tables -> per-cancer-type median expression, gene splits
  04_modeling/                           train.py, eval.py, network.py (interactome_CNN), utils/, calc_performance.R, *_visual.R
  05_interpret/                          compute_shap.py, summarize_SHAP.R, shap_plot.R
  config/{pancanatlas,pcawg}_params.json model hyper-parameters as distributed (copied into the workspace, see section 3)
  config/gene_split/<p>_{train,val,test}.csv  gene splits as distributed (chromosome 1 = test)
  sample_data/pcawg_sample.zip           small example of the processed inputs (not used by any script)
  environment.yml, R_requirements.txt    conda environment (2024: `dexomics`) and R packages
  setup_workspace.sh                     creates the working directories the scripts expect (section 3)
```

`<p>` is the project name, `pancanatlas` or `pcawg`. Both are supported by every script through `-p <p>` or a
positional argument.

## 2. Environment

```bash
conda env create -f pancancer/environment.yml     # creates `dexomics` (python 3.8, torch 1.13, shap 0.44, numpy 1.24, pandas 2.0)
conda activate dexomics
```
R >= 4 with the packages in `R_requirements.txt` (tidyverse, ggpubr, pheatmap, optparse; Bioconductor
SummarizedExperiment, rtracklayer, plyranges, clusterProfiler, org.Hs.eg.db; `03_expr_process` additionally loads
TCGAbiolinks and caTools). The 2024 runs used Ubuntu 24.04 with conda 24.4 (CPU or GPU; the batch scripts
`01_run.sh`, `04_run.sh`, `05_run.sh` are the SGE/Slurm wrappers used on the server and only document arguments).

## 3. Working directories (read this before running anything)

Every script addresses its inputs and outputs with fixed relative paths from its own directory:

| run from | path in the code | meaning |
|---|---|---|
| `pancancer/04_modeling/`, `pancancer/05_interpret/` | `../../data/<p>_expid.txt`, `../../data/modeling/` | processed expression target, gene splits, `<p>_params.json` |
| same | `../../results/rna_features/`, `../../results/promoter_features/` | sparse binding features (01 + 02 outputs) |
| same | `../../<p>_model/reg0.001/` | checkpoints (`best_model.pth`, `epoch<N>.pth`) and `feature_norm_stats.txt` |
| same | `../../test_metrics_<p>/`, `../../plots/`, `../../results/shap/` | evaluation tables, figures, SHAP values |
| `pancancer/` (01_run.sh) | `../data/<gtf>`, `../results/{rna,promoter}_features/`, `../../data/<regulator BEDs>` | 01 inputs and outputs |
| `pancancer/03_expr_process/` | `../../../data/GDCdata/`, `../../../DEomics/data/TCGAdata/LIHC/LIHC.exp.rda` | raw source tables, see section 4a |

With the code under `pancancer/`, `../../<name>` is the repository root. Do **not** create these directories inside
the clone; run

```bash
bash pancancer/setup_workspace.sh [WORKSPACE]      # default WORKSPACE: $PANCANCER_WORKSPACE or ~/Data/DExOmics/pancancer
```

once. It creates `data/`, `results/`, `plots/`, `pancanatlas_model/`, `pcawg_model/`, `test_metrics_pancanatlas/`,
`test_metrics_pcawg/` at the repository root as symlinks into the workspace (all ignored by git, see `.gitignore`)
and copies `config/<p>_params.json` and `config/gene_split/<p>_*.csv` to `<workspace>/data/modeling/`. The
parameter file is copied rather than linked because `train.py` rewrites `data/modeling/<p>_params.json` in place
(it stores the input dimensions it infers from the data); the tracked copy in `config/` stays as distributed.

The `cancer_specific/` study does not use any of these names (it takes `--data-root`, default `~/Data/DExOmics`),
so the two workspaces do not collide even when they share the same parent directory.

## 4. Inputs

### 4a. Raw inputs (outside the repository; needed only to regenerate the processed bundle)

| used by | file(s) | origin |
|---|---|---|
| `01_run.sh` | `data/references_v8_gencode.v26.GRCh38.genes.gtf` | GENCODE v26 (also in the 2024 Drive archive `pancan_data.tar.gz`) |
| `01_run.sh` | `../data/postar3/bed_rna/`, `../data/gtrd/Meta-clusters_by_TF/bed/`, `../data/targetscan/hg38_beds/` (sibling of the repository) | POSTAR3, GTRD, TargetScan BED files; the extraction of the BED directories from the database downloads is not scripted here |
| `03_expr_process/get_pancanatlas_exp.R` | `../data/GDCdata/PanCanAtlas/EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv`, `clinical_PANCAN_patient_with_followup.tsv` | GDC PanCanAtlas publication files |
| `03_expr_process/get_pcawg_exp.R` | `../data/GDCdata/PCAWG/tophat_star_fpkm.v2_aliquot_gl.tsv`, `rnaseq.extended.metadata.aliquot_id.V4.tsv` | PCAWG expression release |
| `03_expr_process/split_*.R` | `data/modeling/{rbp_genes,miRNA_genes,tf_genes}.csv` | regulator gene lists; **not produced by any script in this repository** (origin to be documented) |
| `03_expr_process/split_*.R` | `../DEomics/data/TCGAdata/LIHC/LIHC.exp.rda` | gene annotation taken from the TCGA-LIHC download of the 2024 cancer-specific study (`legacy/data_download_2024/load_expr.R`) |

Only the POSTAR3 line of `01_run.sh` is active; the GTRD and TargetScan lines are commented out although their
outputs (`output_gtrd_promoter.pkl`, `output_targetscan_rna.pkl`) are required inputs of `04`/`05`.

### 4b. Minimum processed input bundle (the supported starting point)

Regenerating 4a is not practical for most users. `04_modeling` and `05_interpret` need exactly the following files
per project, as they were distributed with the 2024 paper (Google Drive folder linked from `../legacy/README_2024.md`,
archive `pancan_data.tar.gz`; the 2024 layout `data/pancan_data/transcriptome/<p>_expid.txt`,
`data/pancan_data/{rna,promoter}_features/` maps onto the workspace as shown):

| workspace path | content | produced by |
|---|---|---|
| `data/<p>_expid.txt` | per-cancer-type median expression of each gene (prediction target, `n_out` columns) | `03_expr_process` |
| `data/modeling/<p>_{train,val,test}.csv` | gene splits (tracked: `config/gene_split/`) | `03_expr_process/split_<p>_data.R` |
| `data/modeling/<p>_params.json` | hyper-parameters (tracked: `config/`) | distributed; rewritten by `train.py` |
| `results/rna_features/output_postar_rna.pkl`, `output_postar_rna_feature_name.txt.gz` | RBP binding bins | `01` + `02` (POSTAR3) |
| `results/rna_features/output_targetscan_rna.pkl`, `output_targetscan_rna_feature_name.txt.gz` | miRNA target-site bins | `01` + `02` (TargetScan) |
| `results/promoter_features/output_gtrd_promoter.pkl`, `output_gtrd_promoter_feature_name.txt.gz` | TF binding bins | `01` + `02` (GTRD) |

To evaluate or interpret the distributed models without training, add `<p>_model/reg0.001/best_model.pth`,
`epoch1..100.pth` (`eval.py` loads every epoch) and `feature_norm_stats.txt` from the "pretrained models" Drive
folder of the 2024 README.

`sample_data/pcawg_sample.zip` contains the same file set for a small gene subset and can be used to check that the
environment loads the inputs; no script depends on it.

## 5. Running

All commands below assume the workspace of section 3 and the `dexomics` environment. `<p>` = `pancanatlas` or `pcawg`.

```bash
# 1-2. binding features (only if the pkl files are regenerated; see 01_run.sh for the exact arguments)
cd pancancer
Rscript 01_bed_to_RNA_coord.R -b ../../data/postar3/bed_rna -n 100 -g ../data/references_v8_gencode.v26.GRCh38.genes.gtf \
        -t rna -o ../results/rna_features/output_postar_rna -s POSTAR3
python 02_to_sparse.py ../results/rna_features/output_postar_rna.txt
# (repeat with -s GTRD -t promoter -> ../results/promoter_features/output_gtrd_promoter, and -s TARGETSCAN -t rna -> output_targetscan_rna)

# 3. expression target and gene splits (raw tables of section 4a required)
cd pancancer/03_expr_process
Rscript get_<p>_exp.R && Rscript split_<p>_data.R

# 4. training, evaluation, performance tables and figures
cd pancancer/04_modeling
python train.py ../../<p>_model/reg0.001/ -p <p> -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.001
python eval.py  ../../<p>_model/reg0.001/ -p <p> -n 100 -reg 0.001
Rscript calc_performance.R <p>
Rscript <p>_visual.R <p>

# 5. attribution
cd pancancer/05_interpret
python compute_shap.py ../../results/shap/ExpectedGrad_<p>/ -p <p>        # see section 7 on the explainer
Rscript summarize_SHAP.R 0.001 <p> ../../results/shap/ExpectedGrad_<p>/   # args: l2 regularisation, project, SHAP directory
Rscript shap_plot.R <p> ../../results/shap/ExpectedGrad_<p>/ ../../plots/
```

Output directories passed as arguments must end with `/` (the scripts concatenate file names to them).

## 6. Outputs

| step | files |
|---|---|
| `train.py` | `<p>_model/reg0.001/best_model.pth`, `epoch<N>.pth`, `feature_norm_stats.txt`; `plots/<p>_pretrained/train_val_loss_reg0.001.png`; rewrites `data/modeling/<p>_params.json` |
| `eval.py` | `test_metrics_<p>/evaluation.csv` (per-epoch test PCOR), `actual.txt.gz`, `prediction.txt.gz`, `geneid.txt.gz`; `plots/<p>_pretrained/test_AUCs.png` |
| `calc_performance.R` | `test_metrics_<p>/cor_tbl.txt` (per-cancer-type Spearman correlation) |
| `<p>_visual.R` | `plots/<p>_spearman.pdf`, `<p>_combined_actual_pred.pdf`, `<p>_MedianExp_actual_pred.pdf`, `<p>_pairwise_heatmap*.{pdf,png}` (+ `<p>_system_spearman.pdf` for PanCanAtlas) |
| `compute_shap.py` | `results/shap/<method>_<p>/RNA_<k>.txt.gz`, `DNA_<k>.txt.gz` (one per output cancer type `k`) |
| `summarize_SHAP.R` | `results/shap/<method>_<p>/{RNA,promoter}_importance_mean.txt` |
| `shap_plot.R` | `plots/*` summary figures |

None of these are tracked by git.

## 7. Differences from the published copy, and caveats

* `compute_shap.py` contains both a DeepLIFT block (`shap.DeepExplainer`) and an Expected Gradients block
  (`shap.GradientExplainer`); one of them is commented out with a `'''` string. In this recovered version the
  **GradientExplainer block is active** and the DeepLIFT block is commented out; in the published copy
  (`legacy/scripts_2024/pan_cancer/compute_shap.py`) it is the other way round. The output file names are the
  same for both methods, so the method is only identifiable from the output directory name used
  (`05_run.sh` uses `DeepLIFT_<p>` and `ExpectedGrad_<p>`). Which method produced the manuscript values has to be
  confirmed by the authors; the code was not changed here.
* Function names differ from the published copy (`prep_ml_data_split`/`batch_iter`/`evaluate` here versus
  `pancan_prep_ml_data_split`/`pancan_batch_iter`/`pancan_evaluate` there), the model definition
  `interactome_CNN` is the same computation, and this version additionally writes the loss and PCOR plots.
* `train.py` rewrites `data/modeling/<p>_params.json` in place (unchanged behaviour; the file now lives in the
  workspace, so the clone stays clean).
* `summarize_SHAP.R` takes three arguments here (`<l2_reg> <p> <dir>`), the published copy two (`<p> <dir>`).
* `01_bed_to_RNA_coord.R` is the same script as the 2024 `scripts/01_bed_to_RNA_coord.R` with explicit
  GTRD / TARGETSCAN / POSTAR3 branches for the regulator name; the `-s ENCODE` branch used by the 2024
  cancer-specific study is still present.
* `05_run.sh` and the file names written by `shap_plot.R` (`*_DeepLIFT_*.pdf`) label the results "DeepLIFT"
  regardless of which explainer block is active in `compute_shap.py`. The label is not evidence of the method.
* `environment.yml` was exported on Linux and pins Linux builds; on macOS create the environment from the package
  list by hand (python 3.8, torch 1.13, shap 0.44, numpy 1.24, pandas 2.0, scikit-learn, scipy, matplotlib, seaborn).
* `eval.py` and `compute_shap.py` call `torch.load` without `map_location`; checkpoints saved on a GPU need a GPU
  (or a one-line `map_location='cpu'` edit, not made here) to load on a CPU-only machine.
* `04_modeling/utils/generate_data.py` writes a synthetic 3-class demo table to `../../results/demo_*.csv` (relative
  to the working directory `04_modeling/`); nothing reads it.

## 8. Checks after cloning (no data needed)

From the repository root, in the `dexomics` environment:

```bash
bash pancancer/setup_workspace.sh /tmp/pancancer_ws_check        # prints LINK/COPY lines, exits 0
( cd pancancer/04_modeling && python -B -c 'import network, utils.data_tool, utils.model_utils' && python -B train.py --help >/dev/null && python -B eval.py --help >/dev/null )
( cd pancancer/05_interpret && python -B -c 'import sys; sys.path.append("../04_modeling/"); import shap, network, utils.data_tool' && python -B compute_shap.py --help >/dev/null )
Rscript -e 'for (f in list.files("pancancer", pattern = "[.]R$", recursive = TRUE, full.names = TRUE)) invisible(parse(file = f))'
```

With the processed bundle of section 4b in place, `eval.py` on the distributed checkpoints followed by
`calc_performance.R` is the first end-to-end check; its `test_metrics_<p>/cor_tbl.txt` should reproduce the
per-cancer-type correlations of the 2024 metrics folder.
