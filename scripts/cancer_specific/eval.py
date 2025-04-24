import os
import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.append('../')
from network import *
from utils.data_tool import encode_label
from utils.model_utils import *

# Create the parser
parser = argparse.ArgumentParser(description="Evaluation")
# Add arguments
parser.add_argument('tcga_cancer', type=str, help='The cancer type from TCGA to be trained')
parser.add_argument('encode_cell_line', type=str, help='The cell line from ENCODE to be trained')
parser.add_argument('outloc', type=str, help='The directory to store the output')
parser.add_argument('-n', '--epoch', type=int, help='Epoch', default=100)
parser.add_argument('-reg', '--l2_reg', type=float, help='Regularization parameter', default=0.01)
# Parse the arguments
args = parser.parse_args()
# Access the arguments
tcga_cancer = args.tcga_cancer
encode_cell_line = args.encode_cell_line
outloc = args.outloc
epoch = args.epoch
l2_reg = args.l2_reg

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

# Data loading 
data = pd.read_csv('../../data/TCGAprocessed/'+tcga_cancer+'/'+'merged_tcga_encode.csv')
data['DEclass'] = data.apply(encode_label, axis=1)     # encode target
data = data.drop(columns = ['DElabel'], axis = 1)    # delete DElabel
mRNA_data_loc = '../../data/rna_features/'
promoter_data_loc = '../../data/promoter_features/'
with open('../../pretrained/'+tcga_cancer+'/params.json') as f:
    params = json.load(f) 

# Data split 
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
    outloc=outloc)

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
tcga_val_data, tcga_val_labels = tcga_val_df.iloc[:, :-1], tcga_val_df.iloc[:, -1]
tcga_test_data, tcga_test_labels = tcga_test_df.iloc[:, :-1], tcga_test_df.iloc[:, -1]

# Batch initialization -------------------------------------------------------------------
train_steps, train_batches = batch_iter(tcga_train_data,
                                        encode_X_mRNA_train.values[:,1],
                                        encode_X_promoter_train.values[:,1],
                                        encode_Y_train.values,
                                        batch_size=len(tcga_train_data),
                                        shuffle=False)

val_steps, val_batches = batch_iter(tcga_val_data,
                                    encode_X_mRNA_val.values[:,1],
                                    encode_X_promoter_val.values[:,1],
                                    encode_Y_val.values,
                                    batch_size=len(tcga_val_data),
                                    shuffle=False)

test_steps, test_batches = batch_iter(tcga_test_data,
                                      encode_X_mRNA_test.values[:,1],
                                      encode_X_promoter_test.values[:,1],
                                      encode_Y_test.values,
                                      batch_size=len(tcga_test_data),
                                      shuffle=False)


# Pass the class weights to the loss function
criterion = nn.CrossEntropyLoss()

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluation on val set
best_model = ConcatedNet(params)
best_model.load_state_dict(torch.load(outloc+"/best_model.pth"))
val_auc_roc, val_accuracy, val_f1, val_precision, val_loss = evaluate(best_model, val_steps, val_batches, criterion, device)
print('Best model on Val: AUC-ROC={:.4f}, ACC={:.4f}, F1={:.4f}, Precision={:.4f}, Loss={:.4f}'.format(val_auc_roc, val_accuracy, val_f1, val_precision, val_loss))


# Evaluation on test set
test_AUCs = []
test_ACCs = []
with open('../../metrics/'+tcga_cancer+'_full_test_metrics.csv', 'w') as file:
    print('Testing for reg'+str(l2_reg), file=file)
    for i in range(epoch):
        model = ConcatedNet(params)
        model.load_state_dict(torch.load(outloc+"epoch"+str(i+1)+".pth"))
        # Evaluate the model on the test data
        test_auc_roc, test_accuracy, test_f1, test_precision, test_loss = evaluate(model, test_steps, test_batches, criterion, device)
        test_AUCs.append(test_auc_roc)
        test_ACCs.append(test_accuracy)
        print('Test {}/{}: AUC-ROC={:.4f}, ACC={:.4f}, F1={:.4f}, Precision={:.4f}, Loss={:.4f}'.format(i+1, epoch, test_auc_roc, test_accuracy, test_f1, test_precision, test_loss),
              file=file)
        
    # Performance of the best model with the highest val ACC
    test_auc_roc, test_accuracy, test_f1, test_precision, test_loss = evaluate(best_model, test_steps, test_batches, criterion, device)
    print('Best model on Test: AUC-ROC={:.4f}, ACC={:.4f}, F1={:.4f}, Precision={:.4f}, Loss={:.4f}'.format(test_auc_roc, test_accuracy, test_f1, test_precision, test_loss), file=file)