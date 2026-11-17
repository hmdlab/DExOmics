# DExOmics pan-cancer study

This directory contains the pan-cancer component of **“An Interpretable Multi-Omics Integration Framework for Differential Expression Prediction in Cancer”**.

The workflow uses PanCanAtlas and PCAWG expression data together with positional TF, RBP, and miRNA binding features. An interactome CNN predicts cancer-type-specific expression patterns, followed by attribution analysis.

The core computational scripts were recovered from the original server project (`DEomics_Pancancer`) and reorganized under `pancancer/` during the 2026 repository restructuring. The pre-restructuring repository state is preserved in the Git tag `pre-restructure-2026`.

The pan-cancer study is independent of the revised workflow in `../cancer_specific/`.

## 1. Layout

```text
pancancer/
├── 01_bed_to_RNA_coord.R
├── 02_to_sparse.py
├── 03_expr_process/
│   ├── get_pancanatlas_exp.R
│   ├── get_pcawg_exp.R
│   ├── split_pancanatlas_data.R
│   └── split_pcawg_data.R
├── 04_modeling/
│   ├── train.py
│   ├── eval.py
│   ├── network.py
│   ├── calc_performance.R
│   ├── pancanatlas_visual.R
│   ├── pcawg_visual.R
│   └── utils/
├── 05_interpret/
│   ├── compute_shap.py
│   ├── summarize_SHAP.R
│   └── shap_plot.R
├── config/
│   ├── pancanatlas_params.json
│   ├── pcawg_params.json
│   └── gene_split/
├── environment.yml
├── R_requirements.txt
└── setup_workspace.sh
```

`<p>` below refers to either `pancanatlas` or `pcawg`.

## 2. Environment

The recovered workflow was developed with Python 3.8 and PyTorch 1.13.

```bash
conda env create -f pancancer/environment.yml
conda activate dexomics
```

The Python environment includes the versions used by the recovered project, including approximately:

```text
Python 3.8
PyTorch 1.13
SHAP 0.44
NumPy 1.24
pandas 2.0
scikit-learn 1.3
SciPy 1.10
```

The R scripts require the packages listed in `R_requirements.txt`.

The environment file lists the direct dependencies required by the pan-cancer workflow rather than the full platform-specific package export from the original server environment. The original environment snapshot
remains available through the `pre-restructure-2026` Git tag. On another operating system, recreating the environment from the main Python and R dependencies may be preferable to reproducing every build pin.

## 3. Workspace

The original pan-cancer code uses fixed relative paths for inputs and outputs. To keep datasets, checkpoints, and generated results outside the Git repository, prepare an external workspace:

```bash
bash pancancer/setup_workspace.sh /path/to/pancancer_workspace
```

If no path is supplied, the script uses:

```text
$PANCANCER_WORKSPACE
```

or, if that variable is unset:

```text
~/Data/DExOmics/pancancer
```

The setup script creates repository-root symlinks pointing to the external workspace:

```text
data/
results/
plots/
pancanatlas_model/
pcawg_model/
test_metrics_pancanatlas/
test_metrics_pcawg/
```

It also copies the tracked parameter files and fixed gene splits from `pancancer/config/` into:

```text
<workspace>/data/modeling/
```

The parameter files are copied rather than linked because `train.py` modifies them in place when recording inferred input dimensions.

The cancer-specific study does not use these paths and can use a separate `--data-root`.

## 4. Inputs

### 4.1 Supported starting point

For most users, the practical starting point is the processed input bundle rather than regeneration from raw source databases.

For each project (`pancanatlas` or `pcawg`), the modeling workflow requires:

```text
data/<p>_expid.txt

data/modeling/<p>_train.csv
data/modeling/<p>_val.csv
data/modeling/<p>_test.csv
data/modeling/<p>_params.json

results/rna_features/output_postar_rna.pkl
results/rna_features/output_postar_rna_feature_name.txt.gz

results/rna_features/output_targetscan_rna.pkl
results/rna_features/output_targetscan_rna_feature_name.txt.gz

results/promoter_features/output_gtrd_promoter.pkl
results/promoter_features/output_gtrd_promoter_feature_name.txt.gz
```

The train/validation/test splits and parameter files are tracked under:

```text
pancancer/config/
```

The remaining processed data were distributed separately with the original study and are not tracked by Git.

`<p>_expid.txt` contains the expression target matrix. The first column is `Name`, followed by cancer-type expression columns and a `MedianExp` column.

### 4.2 Raw-data regeneration

The repository also retains the upstream scripts used to construct the processed features:

```text
01_bed_to_RNA_coord.R
02_to_sparse.py
03_expr_process/
```

These scripts depend on external resources that are not fully reproduced by this repository, including:

- GENCODE gene annotation
- POSTAR3 RBP binding regions
- GTRD TF binding regions
- TargetScan miRNA target regions
- PanCanAtlas expression and clinical tables
- PCAWG expression and metadata tables
- regulator gene lists used by the split scripts
- a historical TCGA-LIHC expression object used for gene annotation

Therefore, regeneration from raw inputs should currently be considered an archival workflow rather than the primary supported entry point.

Historical preprocessing code can be inspected through:

```bash
git show pre-restructure-2026:<path>
```

## 5. Running the processed-data workflow

Commands below assume that the workspace has been prepared and the required processed inputs are available.

### Training and evaluation

From the repository root:

```bash
cd pancancer/04_modeling
```

Train PanCanAtlas:

```bash
python train.py ../../pancanatlas_model/reg0.001/ \
    -p pancanatlas \
    -bs 50 \
    -n 100 \
    -lr 0.001 \
    -step 30 \
    -reg 0.001
```

Evaluate:

```bash
python eval.py ../../pancanatlas_model/reg0.001/ \
    -p pancanatlas \
    -n 100 \
    -reg 0.001
```

Calculate per-cancer-type performance and generate figures:

```bash
Rscript calc_performance.R pancanatlas
Rscript pancanatlas_visual.R pancanatlas
```

For PCAWG, replace `pancanatlas` with `pcawg`:

```bash
python train.py ../../pcawg_model/reg0.001/ \
    -p pcawg \
    -bs 50 \
    -n 100 \
    -lr 0.001 \
    -step 30 \
    -reg 0.001

python eval.py ../../pcawg_model/reg0.001/ \
    -p pcawg \
    -n 100 \
    -reg 0.001

Rscript calc_performance.R pcawg
Rscript pcawg_visual.R pcawg
```

Output directory arguments must end with `/` because the original scripts construct file paths by string concatenation.

## 6. Attribution analysis

From:

```bash
cd pancancer/05_interpret
```

the recovered server version currently uses SHAP `GradientExplainer` (Expected Gradients):

```bash
python compute_shap.py ../../results/shap/ExpectedGrad_<p>/ -p <p>

Rscript summarize_SHAP.R \
    0.001 \
    <p> \
    ../../results/shap/ExpectedGrad_<p>/

Rscript shap_plot.R \
    <p> \
    ../../results/shap/ExpectedGrad_<p>/ \
    ../../plots/
```

Example for PanCanAtlas:

```bash
python compute_shap.py ../../results/shap/ExpectedGrad_pancanatlas/ \
    -p pancanatlas
```

## 7. Main outputs

Training creates:

```text
<p>_model/reg0.001/best_model.pth
<p>_model/reg0.001/epoch<N>.pth
<p>_model/reg0.001/feature_norm_stats.txt
```

Evaluation creates:

```text
test_metrics_<p>/evaluation.csv
test_metrics_<p>/actual.txt.gz
test_metrics_<p>/prediction.txt.gz
test_metrics_<p>/geneid.txt.gz
test_metrics_<p>/cor_tbl.txt
```

Visualization scripts write figures under:

```text
plots/
```

Attribution analysis writes per-output SHAP values under:

```text
results/shap/<method>_<p>/
```

and summary importance tables such as:

```text
RNA_importance_mean.txt
promoter_importance_mean.txt
```

Generated data, checkpoints, metrics, SHAP values, and plots are not tracked by Git.

## 8. Historical implementation differences

The recovered server project and the copy originally published in the repository are not identical.

The most important difference concerns the attribution method.

In the recovered version:

```text
shap.GradientExplainer
```

is active, corresponding to Expected Gradients.

In the pre-restructuring published copy:

```text
shap.DeepExplainer
```

is active.

The historical version can be inspected with:

```bash
git show pre-restructure-2026:scripts/pan_cancer/compute_shap.py
```

The two implementations write similarly named output files, so file names alone cannot establish which attribution method generated a historical result. This provenance question should be resolved before the attribution results are treated as final manuscript outputs.

Other implementation differences include:

- renamed data-loader and evaluation helper functions;
- additional training-loss and correlation plots in the recovered version;
- different argument signatures for `summarize_SHAP.R`;
- the recovered `01_bed_to_RNA_coord.R` contains explicit GTRD, TargetScan, and POSTAR3 source branches.

These differences are documented here rather than silently harmonized.

## 9. Current technical caveats

The recovered Python code retains its original device-handling behavior.

Several scripts move input tensors to `cuda:0` when CUDA is available, while model placement and NumPy conversion are not handled consistently. This may cause CPU/GPU device mismatches on CUDA systems.

The current code should therefore be treated as the recovered implementation rather than a fully modernized GPU-compatible package.

This behavior will be addressed separately from repository restructuring so that scientific and infrastructural changes remain distinguishable.

## 10. Basic checks

After creating the environment, the following checks can be performed without running the full analysis:

```bash
bash pancancer/setup_workspace.sh /tmp/pancancer_ws_check
```

Python imports:

```bash
cd pancancer/04_modeling

python -B -c \
'import network, utils.data_tool, utils.model_utils'

python -B train.py --help >/dev/null
python -B eval.py --help >/dev/null
```

Attribution imports:

```bash
cd ../05_interpret

python -B -c \
'import sys; sys.path.append("../04_modeling/"); import shap, network, utils.data_tool'

python -B compute_shap.py --help >/dev/null
```

R syntax check from the repository root:

```bash
Rscript -e \
'for (f in list.files("pancancer", pattern="[.]R$", recursive=TRUE, full.names=TRUE)) invisible(parse(file=f))'
```

With the processed input bundle and historical checkpoints available, the most informative reproducibility check is:

```text
eval.py
→ calc_performance.R
→ compare test_metrics_<p>/cor_tbl.txt with the original distributed metrics
```