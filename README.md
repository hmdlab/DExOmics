# DExOmics

Code for **“An Interpretable Multi-Omics Integration Framework for Differential Expression Prediction in Cancer”** and its 2026 revision.

The repository contains two independent analysis workflows:

| Directory | Study | Environment | Start here |
|---|---|---|---|
| [`pancancer/`](pancancer/README.md) | Pan-cancer analysis using PanCanAtlas and PCAWG. An interactome CNN predicts cancer-type-specific expression patterns from TF, RBP, and miRNA binding features, followed by SHAP-based interpretation. | Python 3.8, PyTorch 1.13, R | See `pancancer/README.md` |
| [`cancer_specific/`](cancer_specific/README.md) | Revised cancer-specific analysis using TCGA-LIHC, external validation in ICGC LIRI-JP, and frozen confirmation in TCGA-CESC. Includes chromosome-blocked validation, XGBoost and position-aware CNN models, pretraining, ensemble evaluation, and interpretation. | Python 3.12, PyTorch 2.x, R | `bash cancer_specific/run_all.sh data`, then `bash cancer_specific/run_all.sh all` |

The two studies are computationally independent and maintain separate environments and input conventions.

## Repository structure

```text
DExOmics/
├── README.md
├── LICENSE
├── pancancer/
│   ├── README.md
│   ├── environment.yml
│   ├── R_requirements.txt
│   ├── setup_workspace.sh
│   ├── config/
│   ├── 03_expr_process/
│   ├── 04_modeling/
│   └── 05_interpret/
└── cancer_specific/
    ├── README.md
    ├── environment.yml
    ├── requirements.txt
    ├── run_all.sh
    ├── gdc/
    └── analysis scripts
```

Large datasets, trained model checkpoints, intermediate results, and generated figures are intentionally kept outside the Git repository. Small configuration files and fixed gene splits required by the pan-cancer workflow are tracked under `pancancer/config/`.

Historical code from before the 2026 repository restructuring is preserved in the Git tag:

```bash
git checkout pre-restructure-2026
```

Individual historical files can also be inspected without changing branches, for example:

```bash
git show pre-restructure-2026:scripts/pan_cancer/compute_shap.py
```

## Reproducibility

The revised cancer-specific workflow has a single driver:

```bash
bash cancer_specific/run_all.sh --data-root /path/to/data all
```

The pan-cancer workflow retains the computational structure of the recovered 2024 project and requires a separate external workspace for data, checkpoints, and outputs. See [`pancancer/README.md`](pancancer/README.md) for the required processed inputs and commands.

## Citation

If you use this project, please cite the associated paper. Citation information will be added upon publication of the revision.

## License

See [`LICENSE`](LICENSE).