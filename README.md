Code for the paper **"An Interpretable Multi-Omics Integration Framework for Differential Expression Prediction in
Cancer"** and for its 2026 revision. The repository holds two independent studies, each with its own environment,
README, inputs and run commands:

| directory | study | environment | start here |
|---|---|---|---|
| [`pancancer/`](pancancer/README.md) | Pan-cancer model (PanCanAtlas and PCAWG): interactome CNN predicting per-cancer-type expression from TF / RBP / miRNA binding features, with SHAP attribution. Original 2024 code, unchanged. | Python 3.8, PyTorch 1.13 (`pancancer/environment.yml`), R | `bash pancancer/setup_workspace.sh`, then `pancancer/README.md` §4–5 |
| [`cancer_specific/`](cancer_specific/README.md) | Cancer-specific model (TCGA-LIHC, external replication in ICGC LIRI-JP, frozen TCGA-CESC confirmation): revised 2026 pipeline with TCGA-only labels, chromosome-blocked nested CV, XGBoost + positional CNN ensemble, interpretation. | Python 3.12, PyTorch 2.x (`cancer_specific/requirements.txt`), R for one step | `bash cancer_specific/run_all.sh data`, then `all` |
| [`legacy/`](legacy/README.md) | Archived 2024 code: the original cancer-specific pipeline, the published copy of the pan-cancer entry points, the 2024 download scripts, the 2024 README with the Google Drive links. Not runnable from this layout. | — | `legacy/README.md` |

Data, model checkpoints and generated outputs are kept outside the repository (`.gitignore` lists the names the two
studies use); Historical pre-restructuring code is preserved in the `pre-restructure-2026` Git tag.

Data, model checkpoints, and generated outputs are kept outside the repository. Small configuration files and gene splits required by the pan-cancer workflow are tracked under `pancancer/config/`. Each study's README says which processed inputs are needed and where the 2024 distributions are.

## Citation

If you find this project helpful, please cite the paper above (reference to be added on publication of the revision).

## License

See `LICENSE`.
