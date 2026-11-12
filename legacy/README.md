# `legacy/` — archived code of the 2024 repository

Nothing in this directory is maintained or runnable from this layout. It is kept so that (a) the provenance of the
processed files distributed with the 2024 paper remains in the repository, and (b) reviewers can see what the 2026
revision replaced and what it dropped. The live code is in `../pancancer/` (pan-cancer study, original 2024
pipeline) and `../cancer_specific/` (revised cancer-specific study, 2026).

To execute any of the files below as they were run in 2024, check out the last pre-restructuring commit
(`cd37887`, branch `pancancer-source-import`, or `e1fd240` on `main`; a tag `pre-restructure-2026` will mark it),
where every relative path is as `README_2024.md` describes. The paths were not adapted to this directory.

## What is here

| path | was | why kept |
|---|---|---|
| `README_2024.md` | top-level `README.md` as of commit `cd37887`: the 2024 README plus the short "Revision analysis (2026)" section added in October 2026, whose `analysis/` links are left as they were (that directory is `cancer_specific/` today) | the only record of the original command lines and Google Drive links |
| `scripts_2024/01_bed_to_RNA_coord.R`, `02_to_sparse.py` | `scripts/` | produced the ENCODE binding features (`encode_hepg2_*`, `encode_hela_*`) that `cancer_specific/02` and `13` still read; the pan-cancer copy of the same scripts is live in `../pancancer/` |
| `scripts_2024/network.py`, `utils/` | `scripts/` | `ConcatedNet` (2024 cancer-specific model) and the published variant of the pan-cancer helpers (`pancan_*`) in one module |
| `scripts_2024/pan_cancer/` | `scripts/pan_cancer/` | the pan-cancer entry points as published with the 2024 README: a historical variant of `../pancancer/04_modeling` and `05_interpret` with the same model definition but a different data layout, renamed helpers and, in `compute_shap.py`, the DeepLIFT explainer active instead of Expected Gradients (differences in `../pancancer/README.md` §7). Kept complete, together with `network.py` and `utils/`, until the attribution provenance is settled; not claimed to be computationally identical to the live copy |
| `scripts_2024/cancer_specific/` | `scripts/cancer_specific/` | the 2024 cancer-specific pipeline (TCGA preprocessing, limma-voom labels, ConcatedNet training, SHAP); superseded as listed below. `evaluation_lihc/` (2026-09) is exploratory and was never part of a manuscript |
| `data_download_2024/` | `data_download/` | TCGA downloads via TCGAbiolinks (`load_*.R`, `run.sh`), ENCODE/POSTAR regulator downloads (`load_regulator.sh`, five URL lists, `split_HeLa.R`) |
| `pretrained_2024/` | `pretrained/{LIHC,CESC}_params.json`, `gene_split/{LIHC,CESC}_*` | hyper-parameters and gene splits of the 2024 LIHC / CESC models (archival record; `cancer_specific/` uses chromosome-blocked folds instead) |
| `sample_data/LIHC_sample.zip` | `sample_data/` | demo inputs of the 2024 cancer-specific model |
| `dependencies_mini.yml` | repository root | reduced conda file of 2024; `pancancer/environment.yml` is the full one |
| `run_logs/slurm-*.out` | `pancancer_source/04_modeling/`, `05_interpret/` | server logs of the recovered pan-cancer project (plot steps) |

Not kept: `pretrained/{pancanatlas,pcawg}_params.json` (byte-identical to `../pancancer/config/*.json`) and
`pancancer_source/.DS_Store`.

## What the 2026 revision replaced (old cancer-specific files → `cancer_specific/`)

Taken from the repository audit of 2026-11-09. "Yes" means the step exists in the revised pipeline; it does not mean
the computation is identical (see the audit, Part 1 §3b, for the feature-definition differences that are still to be
decided before resubmission).

| old file (now under `legacy/`) | superseded? | replacement in `cancer_specific/` |
|---|---|---|
| `data_download_2024/load_expr.R` | Yes | `gdc/gdc_manifest.py` + `gdc_download.py` (STAR counts via the GDC API); TCGAbiolinks preprocessing replaced by pydeseq2 on raw counts in `01`, `11_voom_labels.R` approximates the old labels for the audit |
| `data_download_2024/load_cnv.R`, `load_methy.R`, `load_snv.R`, `run.sh` | Yes | `gdc/gdc_manifest2.py <P> cnv|methy|maf`, `gdc/gdc_download_all.sh` |
| `data_download_2024/load_regulator.sh` + URL lists, `split_HeLa.R` | No replacement | `cancer_specific/` reads their products (`encode_hepg2_*`, `encode_hela_*` pkl) from the 2024 Drive archives |
| `scripts_2024/01_bed_to_RNA_coord.R`, `02_to_sparse.py` | No replacement (live copy in `pancancer/`) | `cancer_specific/02` only re-reads the pkl |
| `scripts_2024/cancer_specific/data_observe.R` | Yes (4-way patient intersection dropped) | `03`, `04`, `05`, `13` use each modality's own patient set |
| `scripts_2024/cancer_specific/dea.R` | Yes | `01_de_tcga_only.py` (TCGA-only labels); ENCODE concordance filter dropped deliberately (`12` quantifies it); DMC → `05` |
| `scripts_2024/cancer_specific/data_merge.R`, `data_func.R` | Yes | `03_cnv_features.py`, `04_mutation_features.py`, `05_methylation_features.py` (mutation and CNV definitions differ, see audit §3b); `mapped_geneID.csv` and `DEG_info.csv` still read from the Drive archive |
| `scripts_2024/cancer_specific/get_encode_genes.py` | No replacement; nothing reads its output | gene sets come from `06` |
| `scripts_2024/cancer_specific/pretrain.py`, `eval.py` | Yes | `06_nested_cv_baselines.py`, `21`/`25`/`27` (CNN), `22c`, `09`, `29`/`29b` |
| `scripts_2024/cancer_specific/compute_shap.py`, `summarize_SHAP.R`, `shap_plot.R` | Yes | `30_interpretability.py`, `31_paper_figures.py`, `32`, `34` (3-class top-10 bar plots and `EG_heatmap.pdf` have no equivalent) |
| `scripts_2024/cancer_specific/individual_shap.py` | Dropped | per-gene attributions remain extractable from `exp5_interp/*.npy` |
| `scripts_2024/cancer_specific/evaluation_lihc/*` | Exploratory (2026-09); not in any manuscript | `21`/`25`/`27`, `29`, `06`; UMAP / ROC-OvR plots have no replacement; `full_run.sh` references a `split_data.R` that was never committed |
| `pretrained_2024/*` | Only used by the 2024 scripts | `cancer_specific/` uses `folds.csv` and transcribed constants |

Dropped on purpose in the revision (listed so that the decision is visible): the ENCODE HepG2/HeLa concordance filter
on the labels, the three-class figures (`upDEG/downDEG_top10EG.pdf`, `EG_heatmap.pdf`), the single-gene attribution
script, the UMAP and ROC-OvR plots of `evaluation_lihc/`.

Known defects of the archived code, left as they were: `scripts_2024/utils/data_tool.py` calls `sys.exit` without
importing `sys`; `scripts_2024/cancer_specific/data_observe.R` reads from a sibling checkout (`../../../DEomics/`);
`evaluation_lihc/network.py` uses `RNA_conv_kernel` for the DNA convolution layers; `README_2024.md` refers to
`get_HepG2_genes.py` (the file is `get_encode_genes.py`) and to `HepG2_bed_rna` (the download script writes
`hepg2_bed_rna` / `hepg2_bed_promoter`).
