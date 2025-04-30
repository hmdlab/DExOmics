# DExOmics
This repository provides code and processed data for the paper:  
**"Interpreting Differential Gene Expression in Cancer through AI-Guided Analysis of Multi-Omics Data"**.


## 1. Dependencies
**Tested Environment**\
OS: Ubuntu 24.04.1 LTS\
Kernel: 6.8.0-51-generic\
conda: 24.4.0

To recreate the virtual environment, run:

```bash
git clone https://github.com/hmdlab/DExOmics.git
cd DExOmics
conda env create -f dependencies_all.yml
conda activate dexomics
```

## 2. Data Sources
All data downloading scripts are provided under the `data_download/` directory.
### Pan-cancer study
- Create a directory:
    ```bash
    mkdir data
    ```
- Download the archive [pancan_data.tar.gz](https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk?usp=drive_link) and place it inside the `data` directory.

### Cancer-specific study
- Run the following command to download TCGA omics data for LIHC and CESC:
    ```bash
    Rscript load_*.R [cancer_type]
    ```
        Output files will be stored under `data/TCGAdata/`.

- Transcription factor (TF) and RNA-binding protein (RBP) binding peak files are defined in `.txt.gz` files under `data_download/`. To download them, run:
    ```bash
    bash load_regulator.sh
    ```

- Additionally, download [human.txt.gz](https://cloud.tsinghua.edu.cn/d/8133e49661e24ef7a915/files/?p=%2Fhuman.txt.gz&dl=1) from POSTAR3 and place it in the `data/` directory.
You can extract HeLa RBP-binding BED files using:
    ```bash
    Rscript split_HeLa.R
    ```


## 3. Proprocessing and Integration
### Mapping BED Features to RNA Coordinates
Convert BED-format genomic interactions to transcript-relative coordinates and sparse matrices:
```bash
mkdir data/promoter_features
Rscript 01_bed_to_RNA_coord.R -b "../data/HepG2_bed_rna" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "promoter" -o "../data/promoter_features/encode_hepg2_promoter" -s "ENCODE"
python 02_to_sparse.py ../data/promoter_features/encode_hepg2_promoter.txt

mkdir data/rna_features
Rscript 01_bed_to_RNA_coord.R -b "../data/HepG2_bed_rna" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "rna" -o "../data/rna_features/encode_hepg2_rna" -s "ENCODE"
python 02_to_sparse.py ../data/rna_features/encode_hepg2_rna.txt
```
>`Arguments:`\
`-b`: BED file directory\
`-n`: Bin size (genomic resolution)\
`-g`: Path to the GTF annotation\
`-t`: Type ("promoter" or "rna")\
`-o`: Output path\
`-s`: Data source name

You can also download preprocessed files:
- [promoter_features.tar.gz](https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk?usp=drive_link)
- [rna_features.tar.gz](https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk?usp=drive_link)

### TCGA Data Preprocessing and Integration
From `scripts/cancer_specific/`, run:
```bash
Rscript data_observe.R LIHC
Rscript dea.R LIHC hepg2
Rscript data_merge.R LIHC hepg2 TRUE    # TRUE to merge with ENCODE expression data
python get_HepG2_genes.py LIHC hepg2
```
>Replace arguments with the desired TCGA project and related cell line.
The processed data is also available as [TCGAprocessed.tar.gz](https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk?usp=drive_link).
Additionally, download the [Methylation Array Gene Annotation File](https://api.gdc.cancer.gov/v0/data/021a2330-951d-474f-af24-1acd77e7664f) and place it in `data/TCGAdata/`.



## 4. Analysis
### Pan-cancer study
Train and evaluate the model under `scripts/pan_cancer/`:
```bash
python pretrain.py ../../pancanatlas_model/ -p pancanatlas -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.001
python eval.py ../../pancanatlas_model/ -p pancanatlas -n 100 -reg 0.001
Rscript calc_performance.R pancanatlas
```

Interpret results using DeepLIFT:
```bash
python compute_shap.py ../../shap/DeepLIFT_pancanatlas/ -p pancanatlas
Rscript summarize_SHAP.R pancanatlas ../../shap/DeepLIFT_pancanatlas/
Rscript shap_plot.R pancanatlas ../../shap/DeepLIFT_pancanatlas/ ../../plots_pancanatlas/
```

### Cancer-specific study
Under `scripts/cancer_specific/`, train and evaluate the model:
```bash
python pretrain.py LIHC hepg2 ../../model_LIHC/concat/ -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.0001
python eval.py LIHC hepg2 ../../model_LIHC/concat/ -n 100 -reg 0.0001
```

Interpret using ExpectedGrad:
```bash
python compute_shap.py LIHC hepg2 ../../shap/ExpectedGrad_LIHC/
Rscript summarize_SHAP.R LIHC ../../shap/ExpectedGrad_LIHC/
Rscript shap_plot.R ../../shap/ExpectedGrad_LIHC/ ../../plots_LIHC/global/
```
>You can substitute `LIHC/hepg2` with other projects and cell lines (e.g., `CESC/hela`).
Pretrained models are available [here](https://drive.google.com/drive/folders/115VOsmUTsXhxcnQ6qf4_8ZRSEP29KyJO?usp=drive_link).

## 5. Citation
If you find this project helpful, please cite: