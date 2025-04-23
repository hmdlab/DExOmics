import os
import sys
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
parser = argparse.ArgumentParser(description="Training")
# Add arguments
parser.add_argument('-reg', '--l2_reg', type=float, help='Regularization parameter')
parser.add_argument('tcga_cancer', type=str, help='The cancer type from TCGA to be trained')
parser.add_argument('encode_cell_line', type=str, help='The cell line from ENCODE to be trained')
parser.add_argument('outloc', type=str, help='The directory to store the output')
# Parse the arguments
args = parser.parse_args()
# Access the arguments
l2_reg = args.l2_reg
tcga_cancer = args.tcga_cancer
encode_cell_line = args.encode_cell_line
outloc = args.outloc

# Data loading --------------------------------------------------
data = pd.read_csv('../../data/TCGAprocessed/'+tcga_cancer+'/'+'merged_tcga_encode.csv')
data['DEclass'] = data.apply(encode_label, axis=1)     # encode target
data = data.drop(columns = ['DElabel'], axis = 1)    # delete DElabel
mRNA_data_loc = '../../data/rna_features/'
promoter_data_loc = '../../data/promoter_features/'
with open('../../pretrained/'+tcga_cancer+'/params.json') as f:
    params = json.load(f)

# Data split --------------------------------------------------
test_file = '../../pretrained/gene_split'+tcga_cancer+'_test.csv'
val_file = '../../pretrained/gene_split'+tcga_cancer+'_val.csv'
train_file = '../../pretrained/gene_split'+tcga_cancer+'_train.csv'
test = pd.read_csv(test_file,sep="\t",header=0).values[:,0]
val = pd.read_csv(val_file,sep="\t",header=0).values[:,0]
train = pd.read_csv(train_file,sep="\t",header=0).values[:,0]
tcga_train_df = data.loc[data['Gene'].isin(train)].set_index('Gene')
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
    outloc='../../model_'+tcga_cancer+'/concat/)

# Standardization 
# Create a StandardScaler object to normalize the training data
scaler = StandardScaler()

# Fit the scaler to the training data and transform the data
train_scaled = scaler.fit_transform(tcga_train_df.iloc[:, 0:-1])

# Transform back to pandas df
tcga_train_df.iloc[:, 0:-1] = train_scaled
test_scaled = scaler.transform(tcga_test_df.iloc[:, 0:-1])
val_scaled = scaler.transform(tcga_val_df.iloc[:, 0:-1])

# Transform back to pandas df
tcga_test_df.iloc[:, 0:-1] = test_scaled
tcga_val_df.iloc[:, 0:-1] = val_scaled

# Extract the data and labels from the training and validation sets
tcga_train_data, tcga_train_labels = tcga_train_df.iloc[:, :-1], tcga_train_df.iloc[:, -1]
tcga_test_data, tcga_test_labels = tcga_test_df.iloc[:, :-1], tcga_test_df.iloc[:, -1]


# Paramter for background distribution
med_mRNA_len = int(np.median(list(map(lambda x:x.shape[1],encode_X_mRNA_test.values[:,1]))))
med_promoter_len = int(np.median(list(map(lambda x:x.shape[1],encode_X_promoter_test.values[:,1]))))
gene_names_test = encode_X_mRNA_test.values[:,0]

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load best model 
best_model = ConcatedNet(params)
best_model.load_state_dict(torch.load('../../model_'+tcga_cancer+'/concat/reg'+str(l2_reg)+'/best_model.pth'))
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

# Assign batch size
batch_size = 280
# batch_size = len(tcga_test_df)

# Get parameters for the last dens layer
last_weights = best_model.output_layer.weight.detach()
last_biases = best_model.output_layer.bias.detach()
# print("old last weight", last_weights.shape)
# print("old last bias", last_biases.shape)

if not os.path.exists(outloc):
    os.makedirs(outloc)

# GradSHAP for each sample
for out_indx in range(0,last_weights.shape[0]):
# for out_indx in range(1):
    print("out_indx:", out_indx)
    
    # Set dens parameters
    last_layer = new_model.output_layer
    last_layer.weight.data = torch.nn.Parameter(last_weights[out_indx:out_indx+1,:].clone())
    last_layer.bias.data = torch.nn.Parameter(last_biases[out_indx:out_indx+1].clone())

    # Speficy output file names
    outfile1 = outloc+'RNA_'+str(out_indx)+'.txt'
    outfile2 = outloc+'DNA_'+str(out_indx)+'.txt'
    outfile3 = outloc+'omics_'+str(out_indx)+'.txt'

    test_steps, test_batches = batch_iter_GradSHAP(tcga_test_data, 
                                                    encode_X_mRNA_test.values[:,1], 
                                                    encode_X_promoter_test.values[:,1], 
                                                    encode_Y_test.values, 
                                                    batch_size=batch_size, 
                                                    med_mRNA_len=med_mRNA_len, 
                                                    med_promoter_len=med_promoter_len, 
                                                    shuffle=False)
                                                  

    # GradientExplainer --------------------------------------------------
    # Prepare background data
    for i in range(test_steps):
    # for i in range(1):
        print("test_steps:", i)
        background_test = next(iter(test_batches))
        max_test_mRNA_len = background_test[0][0].numpy().shape[1]
        max_test_promoter_len = background_test[0][1].numpy().shape[1]
        
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

        x_mRNA_test = test_batch[0][0]
        x_mRNA_test = x_mRNA_test.view(x_mRNA_test.shape[0], 1, x_mRNA_test.shape[1], x_mRNA_test.shape[2]).to(device) # batch_size, channel, length, feature_num
        x_promoter_test = test_batch[0][1]
        x_promoter_test = x_promoter_test.view(x_promoter_test.shape[0], 1, x_promoter_test.shape[1], x_promoter_test.shape[2]).to(device)
        x_tcga_test = test_batch[1].to(device)
        # print("x_mRNA_shape:", x_mRNA.shape)
        # print("x_promoter_shape:", x_promoter.shape)
        # print("x_tcga_shape:", x_tcga.shape)


        '''
        # Calculate with zero-reference
        # Reshape background
        xs_background=[]
        xs_background.append(torch.zeros(1, 1, background[0][0].shape[1], background[0][0].shape[2]))  # X_mRNA
        xs_background.append(torch.zeros(1, 1, background[0][1].shape[1], background[0][1].shape[2]))  # X_promoter
        xs_background.append(torch.median(background[1], dim=0).values.reshape(1, -1))  # X_tcga                   # X_tcga
        xs_background[0][0,0,0:med_mRNA_len,0]=1
        xs_background[1][0,0,0:med_promoter_len,0]=1
        print(xs_background[0].shape)
        print(xs_background[1].shape)
        print(xs_background[2].shape)
    
        # Compute SHAP scores
        e = shap.GradientExplainer(model=new_model, data=xs_background)
        shap_values = e.shap_values([x_mRNA, x_promoter, x_tcga])
        '''

        # Take the random 100 training data as the baseline, and explain for the test data of each batch)
        e = shap.GradientExplainer(model=new_model, data=[x_mRNA_train, x_promoter_train, x_tcga_train])
        shap_values = e.shap_values([x_mRNA_test, x_promoter_test, x_tcga_test])
        # # shap_values = shap_values[0]
        print("shap_values[0]_shape", shap_values[0].shape)
        print("shap_values[1]_shape:", shap_values[1].shape)
        print("shap_values[2]_shape:", shap_values[2].shape)
        print("ROUND END ----------------------")
        

        # Export SHAP scores to text
        with open(outfile1, 'a') as f1:
            for j in range(shap_values[0].shape[0]): # for each gene in the batch
                seq_indx = x_mRNA_test[j,0,:,0] > 0 # mRNA[gene_indx,channel,length,feature_num]
                feature_vector = map(str, np.sum(shap_values[0][j,0,seq_indx,:],axis=0)) # sum up shap values along RNA locations for each RBP
                out_txt = str(out_indx)+'\t'+gene_names_test[j+i*batch_size]+'\t'+','.join(feature_vector)+'\n' # write each line in the outfile
                f1.write(out_txt)

        with open(outfile2, 'a') as f2:
            for j in range(shap_values[1].shape[0]):
                seq_indx = x_promoter_test[j,0,:,0] > 0
                feature_vector = map(str, np.sum(shap_values[1][j,0,seq_indx,:],axis=0))
                out_txt = str(out_indx)+'\t'+gene_names_test[j+i*batch_size]+'\t'+','.join(feature_vector)+'\n'
                f2.write(out_txt)
        
        with open(outfile3, 'a') as f3:
            for j in range(shap_values[2].shape[0]):
                feature_vector = map(str, shap_values[2][j,:])
                out_txt = str(out_indx)+'\t'+gene_names_test[j+i*batch_size]+'\t'+','.join(feature_vector)+'\n'
                f3.write(out_txt)

    # gzip text files
    os.system("gzip -f "+outfile1)
    os.system("gzip -f "+outfile2)
    os.system("gzip -f "+outfile3)
