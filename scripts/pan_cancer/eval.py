import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
sys.path.append('../')
from network import *
from utils.data_tool import *
from utils.model_utils import *

# Create the parser
parser = argparse.ArgumentParser(description="Evaluation")
# Add arguments
parser.add_argument('outloc', type=str, help='The directory to store the output')
parser.add_argument('-p', '--project', type=str, help='The project name in smaller case of the pancancer')
parser.add_argument('-n', '--epoch', type=int, help='Epoch', default=100)
parser.add_argument('-reg', '--l2_reg', type=float, help='Regularization parameter', default=0.01)
# Parse the arguments
args = parser.parse_args()
# Access the arguments
outloc = args.outloc
project = args.project
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

# Batch initialization -------------------------------------------------------------------
val_steps, val_batches = batch_iter(x_mRNA_val.values[:,1],
                                    x_promoter_val.values[:,1],
                                    y_val_numeric.values,
                                    batch_size=len(y_val_numeric),
                                    shuffle=False)

test_steps, test_batches = batch_iter(x_mRNA_test.values[:,1],
                                      x_promoter_test.values[:,1],
                                      y_test_numeric.values,
                                      batch_size=len(y_test_numeric),
                                      shuffle=False)

# Load hyper-parameters -------------------------------------------------------------------
with open(params_file) as f:
    params = json.load(f) 
    
# Loss function
criterion = nn.MSELoss()

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluation on val set
best_model = interactome_CNN(params)
best_model.load_state_dict(torch.load(outloc+"best_model.pth"))
val_pcor, val_loss, val_actual, val_pred= evaluate(best_model, val_steps, val_batches, criterion, device)
print('Best model on Val: PCOR={:.4f}, LOSS={:.4f}'.format(val_pcor, val_loss))

# Evaluation on test set
test_data_dir = '../../metrics/'
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

test_pcors = []
test_losses = []
with open(test_data_dir+project+'_evaluation.csv', 'w') as file:
    print('Testing for reg'+str(l2_reg), file=file)
    for i in range(epoch):
        model = interactome_CNN(params)
        model.load_state_dict(torch.load(outloc+"epoch"+str(i+1)+".pth"))
        # Evaluate the model on the test data
        test_pcor, test_loss, test_actual, test_pred = evaluate(model, test_steps, test_batches, criterion, device)
        test_pcors.append(test_pcor)
        test_losses.append(test_loss)
        print('Test {}/{}: PCOR={:.4f}, LOSS={:.4f}'.format(i+1, epoch, test_pcor, test_loss),
              file=file)
        
    # Performance of the best model with the highest val ACC
    test_pcor, test_loss, test_actual, test_pred = evaluate(best_model, test_steps, test_batches, criterion, device)
    print('Best model on Test: POCR={:.4f}, LOSS={:.4f}'.format(test_pcor, test_loss), file=file)


# Save actual and predicted gene expression values to text files
np.savetxt(os.path.join(test_data_dir, project, '_actual.txt'), test_actual, delimiter='\t')
np.savetxt(os.path.join(test_data_dir, project, '_prediction.txt'), test_pred, delimiter='\t')

# Save gene IDs
x_mRNA_test['Name'].to_csv(os.path.join(test_data_dir, project, '_geneid.txt'), header=False, index=False, sep='\t')

# Optionally gzip the output files
os.system(f"gzip {os.path.join(test_data_dir, project, '_actual.txt')}")
os.system(f"gzip {os.path.join(test_data_dir, project, '_prediction.txt')}")
os.system(f"gzip {os.path.join(test_data_dir, project, '_geneid.txt')}")
