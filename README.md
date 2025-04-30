# DExOmics
This repository provides code and processed data for the paper:  
**"Interpreting Differential Gene Expression in Cancer through AI-Guided Analysis of Multi-Omics Data"**.


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
The downloading of the data can be conducted under `data_download`. 
- Pancancer study
    - Make a new directory by `mkdir data`.
    - Download [pancan_data.tar.gz](https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk?usp=drive_link) and put them under `data`.
- Cancer-specific study
    - Use the command `Rscript load_*.R [cancer_type]` to download each omics data of the TCGA-LIHC and TCGA-CESC. The output data is be stored under `data/TCGAdata`.

    - The bed narrowPeak files of the TF-binding/RBP-binding data are stored in `.txt.gz` files under `data_download`, and run `load_regulator.sh` to download each of them. 

    - Download [human.txt.gz](https://cloud.tsinghua.edu.cn/d/8133e49661e24ef7a915/files/?p=%2Fhuman.txt.gz&dl=1) from POSTAR3 and put it into `data`. Bed files of HeLa RBP-binding data can be extracted using `split_HeLa.R`.


## 3. Proprocessing and Integration
Genomic locations of the interaction data in bed files should be first mapped to local transcript locations, and the data should then be transfered to sparse matrices by using the following example commands:
```
mkdir data/promoter_features
Rscript 01_bed_to_RNA_coord.R -b "../data/HepG2_bed_rna" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "promoter" -o "../data/promoter_features/encode_hepg2_promoter" -s "ENCODE"
python 02_to_sparse.py ../data/promoter_features/encode_hepg2_promoter.txt

mkdir data/rna_features
Rscript 01_bed_to_RNA_coord.R -b "../data/HepG2_bed_rna" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "rna" -o "../data/rna_features/encode_hepg2_rna" -s "ENCODE"
python 02_to_sparse.py ../data/rna_features/encode_hepg2_rna.txt
```
Arguments: b - bed files directory; n - size of bins for the genomic features; g - path to the GTF file; t - input type; o - output file; s - data source.

> The processed data can be found in [promoter_features.tar.gz](https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk?usp=drive_link) and [rna_features.tar.gz](https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk?usp=drive_link). 

 For preprocessing of the TCGA omics data and integration, run the following under `scripts/cancer_specific`:
```
Rscript data_observe.R LIHC
Rscript dea.R LIHC hepg2
Rscript data_merge.R LIHC hepg2 TRUE    # arg3: whether or not merge with encode expression data
python get_HepG2_genes.py LIHC hepg2
```
> Replace the arguments with expected TCGA cancer project and related cell line. The preprocessed and integrated data of cancer-specific study is stored in [TCGAprocessed.tar.gz](https://drive.google.com/drive/folders/1etIOFisUnMDNoQ5UAiMHyz3Mo2n49dAk?usp=drive_link). In addition, the [Methylation Array Gene Annotation File](https://api.gdc.cancer.gov/v0/data/021a2330-951d-474f-af24-1acd77e7664f) should be downloaded for mapping and put under `data/TCGAdata`.

## 4. Analysis
As for pancaner study, the training and evaluation of the model can be done under `scripts/pan_cancer` using:
```
python pretrain.py ../../pancanatlas_model/ -p pancanatlas -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.001
python eval.py ../../pancanatlas_model/ -p pancanatlas -n 100 -reg 0.001
Rscript calc_performance.R pancanatlas
```

Interpretation using DeepLIFT and visualization can be done using:
```
python compute_shap.py ../../shap/DeepLIFT_pancanatlas/ -p pancanatlas
Rscript summarize_SHAP.R pancanatlas ../../shap/DeepLIFT_pancanatlas/
Rscript shap_plot.R pancanatlas ../../shap/DeepLIFT_pancanatlas/ ../../plots_pancanatlas/
```


As for cancer-specific pipeline, run the following unsder `scripts/cancer_specific`:
```
python pretrain.py LIHC hepg2 ../../model_LIHC/concat/ -bs 50 -n 100 -lr 0.001 -step 30 -reg 0.0001
python eval.py LIHC hepg2 ../../model_LIHC/concat/ -n 100 -reg 0.0001
```

Here's an example for interpreting the LIHC model using ExpectedGrad and visualizing the results:
```
python compute_shap.py LIHC hepg2 ../../shap/ExpectedGrad_LIHC/
Rscript summarize_SHAP.R LIHC ../../shap/ExpectedGrad_LIHC/
Rscript shap_plot.R ../../shap/ExpectedGrad_LIHC/ ../../plots_LIHC/global/
```
> Replace the arguments with other expected project (eg. pcawg, CESC) and related cell line (eg. hela), and the trained models are stored [here](https://drive.google.com/drive/folders/115VOsmUTsXhxcnQ6qf4_8ZRSEP29KyJO?usp=drive_link).

## 5. Citation
If you find this project helpful, please cite: