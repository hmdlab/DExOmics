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
parser = argparse.ArgumentParser(description="Evaluation")
# Add arguments
parser.add_argument('outloc', type=str, help='The directory to store the output')
parser.add_argument('-p', '--project', type=str, help='The project name in smaller case of the pancancer')
# Parse the arguments
args = parser.parse_args()
# Access the arguments
outloc = args.outloc
project = args.project

# check if outloc path exist
if not os.path.exists(outloc):
    os.makedirs(outloc)
    
# fix seed
seed_value = 42

torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Evaluate the best model on the test set --------------------------------------------------------------------------

# Data loading -------------------------------------------------------------------
deg_data = pd.read_csv('../../data/pancan_data/transcriptome/'+project+'_expid.txt', sep='\t')
mRNA_data_loc = '../../data/rna_features/'
promoter_data_loc = '../../data/promoter_features/'
params_file = '../../pretrained/'+project+'_params.json'

# Data split -------------------------------------------------------------------
train_names = '../../pretrained/gene_split/'+project+'_train.csv'
val_names = '../../pretrained/gene_split/'+project+'_val.csv'
test_names = '../../pretrained/gene_split/'+project+'_test.csv'


# ENCODE data preparation
deg_data_file = '../../data/pancan_data/transcriptome/'+project+'_expid.txt'
y_train, y_val, y_test, x_mRNA_train, x_mRNA_val, x_mRNA_test, x_promoter_train, x_promoter_val, x_promoter_test = prep_ml_data_split(
    deg_data_file=deg_data_file,
    mRNA_data_loc=mRNA_data_loc,
    promoter_data_loc=promoter_data_loc,
    train_file=train_names,
    val_file=val_names,
    test_file=test_names,
    outloc=outloc)
y_train_numeric = y_train.drop(columns=['Name'])
y_val_numeric = y_val.drop(columns=['Name'])
y_test_numeric = y_test.drop(columns=['Name'])

# Paramter for background distribution
med_mRNA_len = int(np.median(list(map(lambda x:x.shape[1],x_mRNA_test.values[:,1]))))
med_promoter_len = int(np.median(list(map(lambda x:x.shape[1],x_promoter_test.values[:,1]))))
gene_names_test = x_mRNA_test.values[:,0]

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load best model 
with open(params_file) as f:
    params = json.load(f) 

best_model = interactome_CNN(params)
best_model.load_state_dict(torch.load('../../'+project+'_model'+'/best_model.pth'))
best_model.eval()
# print("Old model structure:", best_model)

# Construct a single output model 
params['n_out'] = 1
new_model = interactome_CNN(params)

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

if not os.path.exists(outloc):
    os.makedirs(outloc)

# print(last_weights.shape[0])

# GradSHAP for each sample
for out_indx in range(last_weights.shape[0]):
# for out_indx in range(1):
    print("out_indx:", out_indx)
    
    # Set dens parameters
    last_layer = new_model.output_layer
    last_layer.weight.data = torch.nn.Parameter(last_weights[out_indx:out_indx+1,:].clone())
    last_layer.bias.data = torch.nn.Parameter(last_biases[out_indx:out_indx+1].clone())

    # Speficy output file names
    outfile1 = outloc+'RNA_'+str(out_indx)+'.txt'
    outfile2 = outloc+'DNA_'+str(out_indx)+'.txt'

    test_steps, test_batches = batch_iter_GradSHAP(x_mRNA_test.values[:,1], 
                                                   x_promoter_test.values[:,1], 
                                                   y_test_numeric.values, 
                                                   batch_size=batch_size,
                                                   med_mRNA_len=med_mRNA_len,
                                                   med_promoter_len=med_promoter_len,
                                                   shuffle=False)
                                                  

    # Prepare background data
    for i in range(test_steps):
        '''
        # DeepExplainer --------------------------------------------------
        background = next(iter(test_batches))
        
        xs_background=[]
        xs_background.append(torch.zeros(1, 1, background[0][0].shape[1], background[0][0].shape[2]))  # X_mRNA
        xs_background.append(torch.zeros(1, 1, background[0][1].shape[1], background[0][1].shape[2]))  # X_promoter
        xs_background[0][0,0,0:med_mRNA_len,0]=1
        xs_background[1][0,0,0:med_promoter_len,0]=1
        print(xs_background[0].shape)
        print(xs_background[1].shape)

        mRNA_test = background[0][0]
        mRNA_test = mRNA_test.view(mRNA_test.shape[0], 1, mRNA_test.shape[1], mRNA_test.shape[2]).to(device)
        promoter_test = background[0][1]
        promoter_test = promoter_test.view(promoter_test.shape[0], 1, promoter_test.shape[1], promoter_test.shape[2]).to(device)
    
        # Compute SHAP scores
        e = shap.DeepExplainer(model=new_model, data=xs_background)
        shap_values = e.shap_values([mRNA_test, promoter_test])
        '''
        
        # GradientExplainer --------------------------------------------------
        print("test_steps:", i)
        background_test = next(iter(test_batches))
        print("background_test shape:", background_test[0][0].shape)
        print(type(background_test[0][0]))
        max_test_mRNA_len = background_test[0][0].shape[1]
        max_test_promoter_len = background_test[0][1].shape[1]
        
        train_batch, test_batch = length_align(background_test,
                                                x_mRNA_train.values[:,1], 
                                                x_promoter_train.values[:,1],
                                                sample_size=100, 
                                                max_test_mRNA=max_test_mRNA_len, 
                                                max_test_promoter=max_test_promoter_len, 
                                                shuffle=True)
        
        mRNA_train = train_batch[0]
        # print("x_mRNA_train shape:", mRNA_train.shape)
        # print("x_mRNA_train type:", type(mRNA_train))
        mRNA_train = mRNA_train.view(mRNA_train.shape[0], 1, mRNA_train.shape[1], mRNA_train.shape[2]).to(device) # batch_size, channel, length, feature_num
        promoter_train = train_batch[1]
        promoter_train = promoter_train.view(promoter_train.shape[0], 1, promoter_train.shape[1], promoter_train.shape[2]).to(device)

        mRNA_test = test_batch[0][0]
        # print("x_mRNA_test shape:", mRNA_test.shape)
        # print("x_mRNA_test type:", type(mRNA_test))
        mRNA_test = mRNA_test.view(mRNA_test.shape[0], 1, mRNA_test.shape[1], mRNA_test.shape[2]).to(device) # batch_size, channel, length, feature_num
        promoter_test = test_batch[0][1]
        promoter_test = promoter_test.view(promoter_test.shape[0], 1, promoter_test.shape[1], promoter_test.shape[2]).to(device)
        # print("x_mRNA_shape:", x_mRNA.shape)
        # print("x_promoter_shape:", x_promoter.shape)


        # Take the random 100 training data as the baseline, and explain for the test data of each batch)
        e = shap.GradientExplainer(model=new_model, data=[mRNA_train, promoter_train])
        shap_values = e.shap_values([mRNA_test, promoter_test])
        # # shap_values = shap_values[0]
        print("shap_values[0]_shape", shap_values[0].shape)
        print("shap_values[1]_shape:", shap_values[1].shape)
        print("ROUND END ----------------------")
        

        # Export SHAP scores to text
        with open(outfile1, 'a') as f1:
            for j in range(shap_values[0].shape[0]): # for each gene in the batch
                seq_indx = mRNA_test[j,0,:,0] > 0 # mRNA[gene_indx,channel,length,feature_num]
                feature_vector = map(str, np.sum(shap_values[0][j,0,seq_indx,:],axis=0)) # sum up shap values along RNA locations for each RBP
                out_txt = str(out_indx)+'\t'+gene_names_test[j+i*batch_size]+'\t'+','.join(feature_vector)+'\n' # write each line in the outfile
                f1.write(out_txt)

        with open(outfile2, 'a') as f2:
            for j in range(shap_values[1].shape[0]):
                seq_indx = promoter_test[j,0,:,0] > 0
                feature_vector = map(str, np.sum(shap_values[1][j,0,seq_indx,:],axis=0))
                out_txt = str(out_indx)+'\t'+gene_names_test[j+i*batch_size]+'\t'+','.join(feature_vector)+'\n'
                f2.write(out_txt)

    # gzip text files
    os.system("gzip -f "+outfile1)
    os.system("gzip -f "+outfile2)

