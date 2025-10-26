import os
import sys
import pickle
import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
sys.path.append('../')
from network import *
from utils.data_tool import *

# Create the parser
parser = argparse.ArgumentParser(description="SHAP computation for individual genes")
# Add arguments
parser.add_argument('tcga_cancer', type=str, help='The cancer type from TCGA to be trained')
parser.add_argument('encode_cell_line', type=str, help='The cell line from ENCODE to be trained')
parser.add_argument('gene_name', type=str, help='Name of the gene to be explored')
parser.add_argument('outloc', type=str, help='The directory to store the output')
# Parse the arguments
args = parser.parse_args()
# Access the arguments
tcga_cancer = args.tcga_cancer
encode_cell_line = args.encode_cell_line
gene_name = args.gene_name
outloc = args.outloc
if not os.path.exists(outloc):
    os.makedirs(outloc)


# Data preparation
gene_mapping = pd.read_csv('../../data/TCGAprocessed/'+tcga_cancer+'/'+'mapped_geneID.csv')
mRNA_data_loc = '../../data/rna_features/'
promoter_data_loc = '../../data/promoter_features/'
merged_tcga_file = '../../data/TCGAprocessed/'+tcga_cancer+'/'+'merged_tcga_encode.csv'
mRNA_data, promoter_data, mRNA_feature_name, promoter_feature_name, merged_tcga_data = load_ml_data(merged_tcga_file, 
                                                                                                    mRNA_data_loc, 
                                                                                                    promoter_data_loc, 
                                                                                                    encode_cell_line)

# Extract individual gene
gene_id = gene_mapping[gene_mapping['gene_name'] == gene_name]
if gene_id.empty:
    print(f"[SKIP] gene_name {gene_name} not found.")
    sys.exit(0)
gene_id = gene_id['Gene'].values[0]
single_tcga = merged_tcga_data[merged_tcga_data['Gene'] == gene_id].set_index('Gene')
single_mRNA = mRNA_data[mRNA_data['Name'] == gene_id]
if single_mRNA.empty:
    print(f"[SKIP] gene_id {gene_id} has no mRNA features; skip.")
    sys.exit(0)
single_promoter = promoter_data[promoter_data['Name'] == gene_id]
if single_promoter.empty:
    print(f"[SKIP] gene_id {gene_id} has no promoter features; skip.")
    sys.exit(0)

# Process data for deep learning
single_X_mRNA = np.zeros((1, single_mRNA.values[0,1].shape[1], single_mRNA.values[0,1].shape[0]))
single_X_mRNA[0,0:single_mRNA.values[0,1].shape[1],:] = single_mRNA.values[0,1].transpose()
single_X_mRNA = torch.from_numpy(single_X_mRNA).float()
single_X_promoter = np.zeros((1, single_promoter.values[0,1].shape[1], single_promoter.values[0,1].shape[0]))
single_X_promoter[0,0:single_promoter.values[0,1].shape[1],:] = single_promoter.values[0,1].transpose()
single_X_promoter = torch.from_numpy(single_X_promoter).float()

# Data split --------------------------------------------------
data = pd.read_csv('../../data/TCGAprocessed/'+tcga_cancer+'/'+'merged_tcga_encode.csv')
data['DEclass'] = data.apply(encode_label, axis=1)     # encode target
data = data.drop(columns = ['DElabel'], axis = 1)    # delete DElabel

test_file = '../../pretrained/gene_split/'+tcga_cancer+'_test.csv'
val_file = '../../pretrained/gene_split/'+tcga_cancer+'_val.csv'
train_file = '../../pretrained/gene_split/'+tcga_cancer+'_train.csv'
test = pd.read_csv(test_file,sep="\t",header=0).values[:,0]
val = pd.read_csv(val_file,sep="\t",header=0).values[:,0]
train = pd.read_csv(train_file,sep="\t",header=0).values[:,0]
try:
    gene_idx = [index for index, value in enumerate(train) if value == gene_id]
    train = list(filter(lambda x: x != gene_id, train))
    tcga_train_df = data.loc[data['Gene'].isin(train)].set_index('Gene')
except:
    print(gene_id+" is not in the TCGA training data set.")
    sys.exit(0)

tcga_test_df = data.loc[data['Gene'].isin(test)].set_index('Gene')
tcga_val_df = data.loc[data['Gene'].isin(val)].set_index('Gene')

# ENCODE data preparation
merged_tcga_file = '../../data/TCGAprocessed/'+tcga_cancer+'/'+'merged_tcga_encode.csv'
encode_Y_train, encode_Y_val, encode_Y_test, encode_X_mRNA_train, encode_X_mRNA_val, encode_X_mRNA_test, encode_X_promoter_train, encode_X_promoter_val, encode_X_promoter_test = prep_ml_data_split(
    merged_tcga_file=merged_tcga_file,
    mRNA_data_loc=mRNA_data_loc,
    promoter_data_loc=promoter_data_loc,
    cell_line=encode_cell_line,
    train_file=train_file,
    val_file=val_file,
    test_file=test_file,
    outloc='../../model_'+tcga_cancer+'/concat/')
encode_X_mRNA_train = encode_X_mRNA_train[encode_X_mRNA_train['Name'] != gene_id]
encode_X_promoter_train = encode_X_promoter_train[encode_X_promoter_train['Name'] != gene_id]
if gene_idx: 
    encode_Y_train = encode_Y_train[encode_Y_train['Gene'] != gene_id]


# Load model parameters
with open('../../pretrained/'+tcga_cancer+'_params.json') as f:
    params = json.load(f)


# Standardization 
scaler = StandardScaler()
train_scaled = scaler.fit_transform(tcga_train_df.iloc[:, 0:-1])
tcga_train_df.iloc[:, 0:-1] = train_scaled
single_tcga_scaled = scaler.transform(single_tcga.iloc[:, 0:-1])
single_tcga.iloc[:, 0:-1] = single_tcga_scaled

# Prepare tcga batch data
single_X_tcga = single_tcga.iloc[:, :-1]
try:
    single_X_tcga = torch.from_numpy(single_X_tcga).float()
except:
    single_X_tcga = single_X_tcga

# Extract the data and labels from the training and validation sets
tcga_train_data, tcga_train_labels = tcga_train_df.iloc[:, :-1], tcga_train_df.iloc[:, -1]


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load best model 
best_model = ConcatedNet(params)
best_model.load_state_dict(torch.load('../../model_'+tcga_cancer+'/concat/best_model.pth'))
best_model.eval()
# print("Old model structure:", best_model)

# Construct a single output model 
params['n_out'] = 1
new_model = ConcatedNet(params)

# Copy parameters except the last layer from model_A to model_B
with torch.no_grad():
    for (name_A, param_A), (name_B, param_B) in zip(best_model.named_parameters(), new_model.named_parameters()):
        if 'output_layer' not in name_A and 'output_layer' not in name_B:
            param_B.data.copy_(param_A.data)


# Get parameters for the last dens layer
last_weights = best_model.output_layer.weight.detach()
last_biases = best_model.output_layer.bias.detach()

# GradSHAP for each sample
for out_indx in range(0,last_weights.shape[0]):
# for out_indx in range(1):
    print("out_indx:", out_indx)
    
    # Set dens parameters
    last_layer = new_model.output_layer
    last_layer.weight.data = torch.nn.Parameter(last_weights[out_indx:out_indx+1,:].clone())
    last_layer.bias.data = torch.nn.Parameter(last_biases[out_indx:out_indx+1].clone())

    # Speficy output file names
    outfile1 = outloc+'/RNA_'+str(out_indx)+'.txt'
    outfile2 = outloc+'/DNA_'+str(out_indx)+'.txt'
    outfile3 = outloc+'/omics_'+str(out_indx)+'.txt'
                                                  

    # GradientExplainer --------------------------------------------------
    # Prepare background data
    background_test = [single_X_mRNA, single_X_promoter], single_X_tcga
    max_test_mRNA_len = single_X_mRNA.numpy().shape[1]
    max_test_promoter_len = single_X_promoter.numpy().shape[1]
    train_batch, test_batch = length_align(background_test,
                                           tcga_train_data,
                                           encode_X_mRNA_train.values[:,1], 
                                           encode_X_promoter_train.values[:,1],
                                           encode_Y_train.values, 
                                           sample_size=100, 
                                           max_test_mRNA=max_test_mRNA_len, 
                                           max_test_promoter=max_test_promoter_len, 
                                           shuffle=True,
                                           seed=42)

    x_mRNA_train = train_batch[0][0]
    x_mRNA_train = x_mRNA_train.view(x_mRNA_train.shape[0], 1, x_mRNA_train.shape[1], x_mRNA_train.shape[2]).to(device) # batch_size, channel, length, feature_num
    x_promoter_train = train_batch[0][1]
    x_promoter_train = x_promoter_train.view(x_promoter_train.shape[0], 1, x_promoter_train.shape[1], x_promoter_train.shape[2]).to(device)
    x_tcga_train = train_batch[1].to(device)
    # print("x_mRNA_train shape:", x_mRNA_train.shape)
    # print("x_promoter_train shape:", x_promoter_train.shape)
    # print("x_tcga_train shape:", x_tcga_train.shape)

    x_mRNA_test = test_batch[0][0]
    x_mRNA_test = x_mRNA_test.view(1, 1, x_mRNA_test.shape[1], x_mRNA_test.shape[2]).to(device) # batch_size, channel, length, feature_num
    x_promoter_test = test_batch[0][1]
    x_promoter_test = x_promoter_test.view(1, 1, x_promoter_test.shape[1], x_promoter_test.shape[2]).to(device)
    x_tcga_test = torch.from_numpy(test_batch[1].values).float().to(device)
    # print("x_mRNA_test shape:", x_mRNA_test.shape)
    # print("x_promoter_test shape:", x_promoter_test.shape)
    # print("x_tcga_test shape:", x_tcga_test.shape)


    # Take the random 100 training data as the baseline, and explain for the test data of each batch)
    e = shap.GradientExplainer(model=new_model, data=[x_mRNA_train, x_promoter_train, x_tcga_train])
    shap_values = e.shap_values([x_mRNA_test, x_promoter_test, x_tcga_test])
    # # shap_values = shap_values[0]
    # print("shap_values[0]_shape", shap_values[0].shape)
    # print("shap_values[1]_shape:", shap_values[1].shape)
    # print("shap_values[2]_shape:", shap_values[2].shape)
    # print("ROUND END ----------------------")
    

    # Export SHAP scores to text
    with open(outfile1, 'a') as f1:
        for j in range(shap_values[0].shape[0]): # for each gene in the batch
            seq_indx = x_mRNA_test[j,0,:,0] > 0 # mRNA[gene_indx,channel,length,feature_num]
            feature_vector = map(str, np.sum(shap_values[0][j,0,seq_indx,:],axis=0)) # sum up shap values along RNA locations for each RBP
            out_txt = str(out_indx)+'\t'+gene_name+'\t'+','.join(feature_vector)+'\n' # write each line in the outfile
            f1.write(out_txt)

    with open(outfile2, 'a') as f2:
        for j in range(shap_values[1].shape[0]):
            seq_indx = x_promoter_test[j,0,:,0] > 0
            feature_vector = map(str, np.sum(shap_values[1][j,0,seq_indx,:],axis=0))
            out_txt = str(out_indx)+'\t'+gene_name+'\t'+','.join(feature_vector)+'\n'
            f2.write(out_txt)
    
    with open(outfile3, 'a') as f3:
        for j in range(shap_values[2].shape[0]):
            feature_vector = map(str, shap_values[2][j,:])
            out_txt = str(out_indx)+'\t'+gene_name+'\t'+','.join(feature_vector)+'\n'
            f3.write(out_txt)

    # gzip text files
    os.system("gzip -f "+outfile1)
    os.system("gzip -f "+outfile2)
    os.system("gzip -f "+outfile3)
