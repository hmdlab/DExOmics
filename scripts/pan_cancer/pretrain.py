import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
sys.path.append('../')
from network import interactome_CNN
from utils.data_tool import *
from utils.model_utils import *

# Create the parser
parser = argparse.ArgumentParser(description="Training")
# Add arguments
parser.add_argument('outloc', type=str, help='The directory to store the output')
parser.add_argument('-p', '--project', type=str, help='The project name in smaller case of the pancancer')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=10)
parser.add_argument('-n', '--epoch', type=int, help='Epoch', default=100)
parser.add_argument('-lr', '--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('-step', '--step_size', type=int, help='Step size for reducing the learning rate', default=50)
parser.add_argument('-reg', '--l2_reg', type=float, help='Regularization parameter', default=0.01)
# Parse the arguments
args = parser.parse_args()
# Access the arguments
outloc = args.outloc
project = args.project
batch_size = args.batch_size
epochs = args.epoch
lr = args.lr
step_size = args.step_size
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


# Data loading -------------------------------------------------------------------
deg_data = pd.read_csv('../../data/pancan_data/transcriptome/'+project+'_expid.txt', sep='\t')
mRNA_data_loc = '../../data/pancan_data/rna_features/'
promoter_data_loc = '../../data/pancan_data/promoter_features/'
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
train_steps, train_batches = batch_iter(x_mRNA_train.values[:,1],
                                        x_promoter_train.values[:,1],
                                        y_train_numeric.values,
                                        batch_size=batch_size,
                                        shuffle=True)

val_steps, val_batches = batch_iter(x_mRNA_val.values[:,1],
                                    x_promoter_val.values[:,1],
                                    y_val_numeric.values,
                                    batch_size=batch_size,
                                    shuffle=True)

# Load hyper-parameters -------------------------------------------------------------------
with open(params_file) as f:
    params = json.load(f) 

# Paramters for network structure
params['n_feature_mRNA']=x_mRNA_train.values[:,1][0].shape[0]
params['n_feature_promoter']=x_promoter_train.values[:,1][0].shape[0]
params['n_out'] = y_train.values[:,1:].shape[1]

# Save updated parameters back to the original file
with open(params_file, 'w') as f:
    json.dump(params, f, indent=4)

# Set up MLP model, loss function and optimizer -------------------------------------------------------------------
# Set up the MLP model[[]]
model = interactome_CNN(params)

# Loss function
criterion = nn.MSELoss()

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=lr) # ajust learning rate (lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train and evaluate the MLP model on the validation set -------------------------------------------------------------------
num_epochs = epochs
train_losses = []
val_losses = []
val_pcors = []
epoch_idx = 0
best_val_loss = float('inf')
for epoch in range(num_epochs):
    epoch_idx += 1
    model.train() # Set up training mode
    running_loss = 0.0 # Initialize loss
    y_train_true = []
    y_train_scores = []
    
    data_iter = iter(train_batches)
    for i in range(train_steps): # Iterations
        X = next(data_iter)
        x_mRNA = X[0][0]
        x_mRNA = x_mRNA.view(x_mRNA.shape[0], 1, x_mRNA.shape[1], x_mRNA.shape[2]).to(device) # batch_size*channel*height*width
        x_promoter = X[0][1]
        x_promoter = x_promoter.view(x_promoter.shape[0], 1, x_promoter.shape[1], x_promoter.shape[2]).to(device)
        labels = X[1].to(device)
        optimizer.zero_grad()
        outputs = model(x_mRNA, x_promoter) # =model.forward(inputs)
        y_train_true.extend(labels.cpu().numpy())
        y_train_scores.extend(outputs.detach().numpy())
        loss = criterion(outputs, labels)
        l2_loss = model.l2_loss() * l2_reg
        loss_sum = loss + l2_loss
        if epoch >= 1:
            loss_sum.backward() # Backpropogation
            optimizer.step() # Update parameters
                
        # Update the running loss and predictions
        running_loss += loss.item()
        
    # Calculate overall loss and AUC-ROC of an epoch
    epoch_loss = running_loss / train_steps
    
    # Append the epoch loss to the lists of training losses
    train_losses.append(epoch_loss)
    
    # Update the learning rate scheduler
    scheduler.step()
    
    # Print the epoch number and loss
    print('Epoch {}/{}: train loss={:.4f}'.format(epoch+1, num_epochs, epoch_loss))
    
    # Evaluate the finally updated model from training on the val set
    val_pcor, val_loss, val_actual, val_pred = evaluate(model, val_steps, val_batches, criterion, device)
    
    # Append performance metrics of the val set
    val_pcors.append(val_pcor)
    val_losses.append(val_loss)
    
    print('Epoch {}/{}: val pcor={:.4f}, val loss={:.4f}'.format(epoch+1, num_epochs, val_pcor, val_loss))
    
    # Save the latest best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), outloc+'best_model.pth')
    
    # Save the model of the epoch
    torch.save(model.state_dict(), outloc+'epoch'+str(epoch_idx)+'.pth')    

print('Best Val PCOR: {:.4f}'.format(max(val_pcors)))