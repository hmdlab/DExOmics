import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
sys.path.append('../')
from network import *
from utils.data_tool import *
from utils.model_utils import *

# Create the parser
parser = argparse.ArgumentParser(description="Training")
# Add arguments
parser.add_argument('tcga_cancer', type=str, help='The cancer type from TCGA to be trained')
parser.add_argument('encode_cell_line', type=str, help='The cell line from ENCODE to be trained')
parser.add_argument('outloc', type=str, help='The directory to store the output')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=10)
parser.add_argument('-n', '--epochs', type=int, help='Epoch', default=100)
parser.add_argument('-lr', '--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('-step', '--step_size', type=int, help='Step size for reducing the learning rate', default=50)
parser.add_argument('-reg', '--l2_reg', type=float, help='Regularization parameter', default=0.01)
# Parse the arguments
args = parser.parse_args()
# Access the arguments
tcga_cancer = args.tcga_cancer
encode_cell_line = args.encode_cell_line
outloc = args.outloc
batch_size = args.batch_size
epochs = args.epochs
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
data = pd.read_csv('../../data/TCGAprocessed/'+tcga_cancer+'/'+'merged_tcga_encode.csv')
data['DEclass'] = data.apply(encode_label, axis=1)     # encode target
data = data.drop(columns = ['DElabel'], axis = 1)    # delete DElabel
mRNA_data_loc = '../../data/rna_features/'
promoter_data_loc = '../../data/promoter_features/'
params_file = '../../pretrained/'+tcga_cancer+'_params.json'


# Data split -------------------------------------------------------------------
train_names = '../../pretrained/gene_split'+tcga_cancer+'_train.csv'
val_names = '../../pretrained/gene_split'+tcga_cancer+'_val.csv'
test_names = '../../pretrained/gene_split'+tcga_cancer+'_test.csv'
train = pd.read_csv(train_names,sep="\t",header=0).values[:,0]
val = pd.read_csv(val_names,sep="\t",header=0).values[:,0]

# TCGA data preparation
tcga_train_df = data.loc[data['Gene'].isin(train)].set_index('Gene')
tcga_val_df = data.loc[data['Gene'].isin(val)].set_index('Gene')

# ENCODE data preparation
merged_tcga_file = '../../data/TCGAprocessed/'+tcga_cancer+'/'+'merged_tcga_encode.csv'
encode_Y_train, encode_Y_val, encode_Y_test, encode_X_mRNA_train, encode_X_mRNA_val, encode_X_mRNA_test, encode_X_promoter_train, encode_X_promoter_val, encode_X_promoter_test = prep_ml_data_split(
    merged_tcga_file=merged_tcga_file,
    mRNA_data_loc=mRNA_data_loc,
    promoter_data_loc=promoter_data_loc,
    cell_line=encode_cell_line,
    train_file=train_names,
    val_file=val_names,
    test_file=test_names,
    outloc=outloc)
print('Train length:', len(encode_Y_train))
print('Val length:', len(encode_Y_val))
print('Test length:', len(encode_Y_test))
print('mRNA features:', encode_X_mRNA_train.values[:,1][0].shape)
print('promoter features:', encode_X_promoter_train.values[:,1][0].shape)
print('Train mRNA genes:', len(encode_X_mRNA_train['Name']))
print('Train promoter genes:', len(encode_X_promoter_train['Name']))

# TCGA Standardization -------------------------------------------------------------------
# Create a StandardScaler object to normalize the training data
scaler = StandardScaler()
# Fit the scaler to the training data and transform the data
train_scaled = scaler.fit_transform(tcga_train_df.iloc[:, 0:-1])
# Transform back to pandas df
tcga_train_df.iloc[:, 0:-1] = train_scaled
# Apply scaler to validation data
val_scaled = scaler.transform(tcga_val_df.iloc[:, 0:-1])
# Transform back to pandas df
tcga_val_df.iloc[:, 0:-1] = val_scaled
print("Standardization Done")

# Assign X and y
# Extract the data and labels from the training and validation sets
tcga_train_data, tcga_train_labels = tcga_train_df.iloc[:, :-1], tcga_train_df.iloc[:, -1]
tcga_val_data, tcga_val_labels = tcga_val_df.iloc[:, :-1], tcga_val_df.iloc[:, -1]
# print(tcga_train_labels.value_counts())
# print(tcga_val_labels.value_counts())


# Batch initialization -------------------------------------------------------------------
train_steps, train_batches = batch_iter(tcga_train_data,
                                        encode_X_mRNA_train.values[:,1],
                                        encode_X_promoter_train.values[:,1],
                                        encode_Y_train.values,
                                        batch_size=batch_size,
                                        shuffle=True)

val_steps, val_batches = batch_iter(tcga_val_data,
                                    encode_X_mRNA_val.values[:,1],
                                    encode_X_promoter_val.values[:,1],
                                    encode_Y_val.values,
                                    batch_size=batch_size,
                                    shuffle=True)

print("X and y Done")

# Load hyper-parameters -------------------------------------------------------------------
with open(params_file) as f:
    params = json.load(f) 

# Paramters for network structure
params['n_feature_mRNA']=encode_X_mRNA_train.values[:,1][0].shape[0]
params['n_feature_promoter']=encode_X_promoter_train.values[:,1][0].shape[0]
params['tcga_input_size']=tcga_train_df.shape[1]-1

# Save updated parameters back to the original file
with open(params_file, 'w') as f:
    json.dump(params, f, indent=4)


# Set up model, loss function and optimizer -------------------------------------------------------------------
model = ConcatedNet(params)

# Define the class weights as a tensor
weights = compute_class_weight(class_weight = "balanced",
                               classes = np.unique(tcga_train_labels),
                               y = tcga_train_labels)
class_weights = torch.FloatTensor(weights)

# Pass the class weights to the loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=lr) # ajust learning rate (lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Model Setup Done")

# Train and evaluate the MLP model on the validation set -------------------------------------------------------------------
num_epochs = epochs
best_val_AUC = float('-inf')
train_losses = []
val_losses = []
val_auc_rocs = []
val_accs = []
val_f1s = []
val_precisions = []
epoch_idx = 0
for epoch in range(num_epochs):
    epoch_idx += 1

    model.train() # Set up training mode
    running_loss = 0.0 # Initialize loss
    y_train_true = []
    y_train_scores = []
    
    data_iter = iter(train_batches)
    for i in range(train_steps): # Iterations
        X = next(data_iter)
        # Correct the dimension of the tensor
        x_mRNA = X[0][0]
        x_mRNA = x_mRNA.view(x_mRNA.shape[0], 1, x_mRNA.shape[1], x_mRNA.shape[2]).to(device) # batch_size*channel*height*width
        x_promoter = X[0][1]
        x_promoter = x_promoter.view(x_promoter.shape[0], 1, x_promoter.shape[1], x_promoter.shape[2]).to(device)
        x_tcga = X[1].to(device)
        labels = X[2].to(device)
        optimizer.zero_grad()
        outputs = model(x_mRNA, x_promoter, x_tcga) # =model.forward(inputs)
        softmax = nn.Softmax(dim=1)
        scores = softmax(outputs)
        y_train_true.extend(labels.cpu().numpy())
        y_train_scores.extend(scores.detach().numpy())
        loss = criterion(outputs, labels)
        l2_loss = model.l2_loss() * l2_reg
        loss_sum = loss + l2_loss
        # if epoch != 0:
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
    val_auc_roc, val_accuracy, val_f1, val_precision, val_loss = evaluate(model, val_steps, val_batches, criterion, device)
    val_ave = (val_auc_roc + val_accuracy) / 2

    
    # Append performance metrics of the val set
    val_accs.append(val_accuracy)
    val_auc_rocs.append(val_auc_roc)
    val_f1s.append(val_f1)
    val_precisions.append(val_precision)
    val_losses.append(val_loss)
    
    print('Epoch {}/{}: AUC-ROC={:.4f}, ACC={:.4f}, F1={:.4f}, Precision={:.4f}'.format(epoch+1, num_epochs, val_auc_roc, val_accuracy, val_f1, val_precision))
    
    # Save the latest best model 
    if val_auc_roc > best_val_AUC:
        best_val_AUC = val_auc_roc
        torch.save(model.state_dict(), outloc+'best_model.pth')
        
    # Save the model of the epoch
    torch.save(model.state_dict(), outloc+'epoch'+str(epoch_idx)+'.pth')    

print('Best Val AUC-ROC: {:.4f}'.format(max(val_auc_rocs)))
print('Best Val ACC: {:.4f}'.format(max(val_accs)))
print('Best Val F1: {:.4f}'.format(max(val_f1s)))
print('Best Val Precision: {:.4f}'.format(max(val_precisions)))