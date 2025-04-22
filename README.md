# DExOmics
The repository contains scripts and processed data for "Interpreting differential gene expression in cancer through AI-guided analysis of multi-omics data".

## 1. Dependencies
Tested Environment
```
OS: Ubuntu 24.04.1 LTS
Kernel: 6.8.0-51-generic
conda: 24.4.0
```
You can reconstruct the virtual environment following these command lines:
```
git clone https://github.com/hmdlab/DExOmics.git
cd DExOmics
conda env create -f dependencies_all.yml
conda activate dexomics
```

## 2. Data Sources
The downloading of the data can be conducted under `/data_download`.  The output data is be stored under `/data/TCGAdata`.
- Pancancer study
    - All data are stored under `/data/pancan_data.tar.gz`
- Cancer-specific study
    - Use the command `Rscript load_*.R [cancer_type]` to download each omics data of the TCGA-LIHC and TCGA-CESC.

    - The bed narrowPeak files of the TF-binding/RBP-binding data are stored in `.txt` files under `/data_download `, and run `bash load_regulator.sh` to download each of them.

    - Download [human.txt.gz](https://cloud.tsinghua.edu.cn/d/8133e49661e24ef7a915/) and put it into `/data`. Bed files of HeLa RBP-binding data can be extracted using `split_HeLa.R`


## 3. Proprocessing