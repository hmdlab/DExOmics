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
The downloading of the data can be conducted under `/data_download`. 
- Pancancer study
    - All data are stored in [pancan_data.tar.gz](https://drive.google.com/drive/folders/14v4aZD8GmAYYpuaPXOEyj2PEa_GojN9G?usp=drive_link).
- Cancer-specific study
    - Use the command `Rscript load_*.R [cancer_type]` to download each omics data of the TCGA-LIHC and TCGA-CESC. The output data is be stored under `/data/TCGAdata`.

    - The bed narrowPeak files of the TF-binding/RBP-binding data are stored in `.txt` files under `/data_download `, and run `bash load_regulator.sh` to download each of them.

    - Download [human.txt.gz](https://cloud.tsinghua.edu.cn/d/8133e49661e24ef7a915/files/?p=%2Fhuman.txt.gz&dl=1) and put it into `/data`. Bed files of HeLa RBP-binding data can be extracted using `split_HeLa.R`.


## 3. Proprocessing and Integration
Genomic locations of the interaction data in bed files should be first mapped to local transcript locations, and the data should be then be transfered to sparse matrices by using the following exmaple commands:
```
mkdir data/promoter_features
Rscript 01_bed_to_RNA_coord.R -b "../data/HepG2_bed_rna" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "promoter" -o "../data/promoter_features/encode_hepg2_promoter" -s "ENCODE"

mkdir data/rna_features
Rscript 01_bed_to_RNA_coord.R -b "../data/HepG2_bed_rna" -n 100 -g "../data/pancan_data/references_v8_gencode.v26.GRCh38.genes.gtf" -t "rna" -o "../data/rna_features/encode_hepg2_rna" -s "ENCODE"
```
Arguments:
- b: bed files directory
- n: size of bins for the genomic features
- g: path to the GTF file
- t: input type
- o: output file
- s: data source

The processed data can be found in [promoter_features.tar.gz](https://drive.google.com/drive/folders/14v4aZD8GmAYYpuaPXOEyj2PEa_GojN9G?usp=drive_link) and [rna_features.tar.gz](https://drive.google.com/drive/folders/14v4aZD8GmAYYpuaPXOEyj2PEa_GojN9G?usp=drive_link)

 For preprocessing of the TCGA omics data and integraion, run the following under `/scripts/cancer_specific`:
```
Rscript data_observe.R LIHC
Rscript dea.R LIHC hepg2
Rscript data_merge.R LIHC hepg2 TRUE    # arg3: wether or not merge with encode expression data
python get_HepG2_genes.py LIHC hepg2
```
> Replace the arguments with expected TCGA cancer project and realted cell line. The preprocessed data of cancer-specific study is stored in [TCGAprocessed.tar.gz](https://drive.google.com/drive/folders/14v4aZD8GmAYYpuaPXOEyj2PEa_GojN9G?usp=drive_link)