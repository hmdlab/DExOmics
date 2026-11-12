import numpy as np
import pandas as pd 
import argparse
import os

# Create the parser
parser = argparse.ArgumentParser(description="Get ENCODE genes")
# Add arguments
parser.add_argument('tcga_cancer', type=str, help='The cancer type from TCGA to be trained')
parser.add_argument('encode_cell_line', type=str, help='The cell line from ENCODE to be trained')
# Parse the arguments
args = parser.parse_args()
tcga_cancer = args.tcga_cancer
encode_cell_line = args.encode_cell_line

mRNA_data_loc = '../../data/rna_features/'
promoter_data_loc = '../../data/promoter_features/'

# Load data
mRNA_data = pd.read_pickle(mRNA_data_loc+'encode_'+encode_cell_line+'_rna.pkl')
def split_ID(x):
    return x.split(".")[0]
mRNA_data['Name'] = mRNA_data['Name'].apply(split_ID)
# Deal with replicates
mRNA_data.drop_duplicates(subset=['Name'], inplace=True) # inplace=True means directly change the original data frame

promoter_data = pd.read_pickle(promoter_data_loc+'encode_'+encode_cell_line+'_promoter.pkl')
promoter_data['Name'] = promoter_data['Name'].apply(split_ID)
# Deal with replicates
promoter_data.drop_duplicates(subset=['Name'], inplace=True)

# Save data
# check if outloc path exist
outloc = '../../data/modeling/'+tcga_cancer+'/'
if not os.path.exists(outloc):
    os.makedirs(outloc)
mRNA_data['Name'].to_csv(outloc+'mRNA_genes.csv')
promoter_data['Name'].to_csv(outloc+'promoter_genes.csv')