import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import shap
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_score
from utils.data_tool import *



# Define a custom dataset class for loading tabular data
class CustomDataset(Dataset):
    """Transform numpy array to pytorch tensor"""
    def __init__(self, data, labels, regression=False):
        # Transform the data from numpy version to pytorch tensor
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = torch.from_numpy(data.values).float()
            
        if regression:
            self.labels = torch.from_numpy(labels.values).float()
        else:
            self.labels = torch.from_numpy(labels.values).long()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]




# Pearson correlation coefficient
def pcor(y_true, y_pred):
    mx = torch.mean(y_true)
    my = torch.mean(y_pred)
    xm = y_true - mx
    ym = y_pred - my
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den
    r = torch.clamp(r, min=-1.0, max=1.0)
    return r



def evaluate(model, data_steps, data_batches, criterion, device):
    """This function is for model prediction and evaluation"""
    model.eval() # Set model to evaluation mode
    y_labels = []
    y_preds = []
    running_loss = 0
    val_pcor = 0

    with torch.no_grad():
        data_iter = iter(data_batches)
        for i in range(data_steps):
            X = next(data_iter)
            x_mRNA = X[0][0]
            x_mRNA = x_mRNA.view(x_mRNA.shape[0], 1, x_mRNA.shape[1], x_mRNA.shape[2]).to(device) # batch_size*channel*height*width
            x_promoter = X[0][1]
            x_promoter = x_promoter.view(x_promoter.shape[0], 1, x_promoter.shape[1], x_promoter.shape[2]).to(device)
            labels = X[1].to(device)
            outputs = model(x_mRNA, x_promoter)
            # Assign the class with the largest probability
            y_preds.extend(outputs.detach().numpy())
                
            loss = criterion(outputs, labels)
            
            # Update the running loss and predictions
            running_loss += loss.item()
            val_pcor += pcor(outputs, labels)
            y_labels.extend(labels.cpu().numpy())
    
    # Validation loss
    epoch_loss = running_loss / data_steps
    epoch_pcor = val_pcor / data_steps    
    y_labels = np.vstack(y_labels)
    y_preds = np.vstack(y_preds)

    return epoch_pcor, epoch_loss, y_labels, y_preds




def sanity_evaluate(model, data_steps, data_batches, criterion, device, level='low'):
    """This function is for model prediction and evaluation"""
    model.eval() # Set model to evaluation mode
    y_labels = []
    y_probs= []
    y_preds = []
    running_loss = 0

    with torch.no_grad():
        data_iter = iter(data_batches)
        for i in range(data_steps):
            X = next(data_iter)
            x_mRNA = X[0][0]
            x_mRNA = x_mRNA.view(x_mRNA.shape[0], 1, x_mRNA.shape[1], x_mRNA.shape[2]).to(device) # batch_size*channel*height*width
            x_promoter = X[0][1]
            x_promoter = x_promoter.view(x_promoter.shape[0], 1, x_promoter.shape[1], x_promoter.shape[2]).to(device)
            labels = X[2].to(device)
            outputs = model(x_mRNA, x_promoter)
            softmax = nn.Softmax(dim=1)
            scores = softmax(outputs)
            # Assign the class with the largest probability
            pred = torch.argmax(scores, dim=1)
            y_preds.extend(pred.detach().numpy())
                
            loss = criterion(outputs, labels)
            
            # Update the running loss and predictions
            running_loss += loss.item()
            
            y_labels.extend(labels.cpu().numpy())
            y_probs.extend(scores.detach().numpy())
    
    # Validation loss
    epoch_loss = running_loss / data_steps

    # Evaluation metrics
    if level == 'low':
        return epoch_loss
    else:
        auc_roc = roc_auc_score(y_labels, y_probs, average='weighted', multi_class='ovo')
        accuracy = accuracy_score(y_labels, y_preds)
        f1 = f1_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
        precision = precision_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
            
        return auc_roc, accuracy, f1, precision, epoch_loss




# def compute_DeepLIFT(model, neuron_of_interest, X_mRNA, X_promoter, outloc):
#     # Input
#     # model: a full path to a directory of the feature extraction model
#     # neuron_of_interest: the index of the neuron of interst to trace back from the last layer
#     # X_mRNA: mRNA annoation data
#     # X_promoter: Promoter annoatation data
#     # outloc: location to store the output
    
#     # Output
#     #.{outloc}/{best_model}/DeepLIFT/DNA_{sample index}.txt.gz: DeepLIFT scores of DNA regulators for each sample. The sample index corresponds to the column index of gene expression data, which starts from 0. DeepLIFT scores of regulators were sparated with commas. Its order is identical to the one appeared in {outloc}/feature_norm_stats.txt   
#     #.{outloc}/{best_model}/DeepLIFT/RNA_{sample index}}.txt.gz: DeepLIFT scores of RNA regulators for each sample. The sample index corresponds to the column index of gene expression data, which starts from 0. DeepLIFT scores of regulators were sparated with commas. Its order is identical to the one appeared in {outloc}/feature_norm_stats.txt 

#     # Get parameters for the last dens layer
#     dens_parameter = model.layers[-1].get_weights()
#     print(dens_parameter)
#     print(dens_parameter[0].shape)
#     print(dens_parameter[1].shape)

#     # Construct a single output model 
#     layer_output = model.layers[-1].output
#     # Select the output of the certain neuron
#     neuron_output = layer_output[neuron_of_interest]
#     new_model = Model(inputs=model.input, outputs=neuron_output)
    
#     # Paramter for background distribution
#     med_mRNA_len = int(np.median(list(map(lambda x: x.shape[1], X_mRNA.values[:, 1]))))      # median mRNA length
#     med_promoter_len = int(np.median(list(map(lambda x:x.shape[1],X_promoter.values[:,1]))))       # median promoter length
#     gene_names = X_mRNA.values[:,0]
    
#     # DeepLIFT score for each sample
#     for out_indx in range(dens_parameter[1].shape[0]):         # get the bias
#         print(out_indx)

#         # Set dens parameters
#         new_model.layers[-1].set_weights([dens_parameter[0][:,out_indx:(out_indx+1)],dens_parameter[1][out_indx:(out_indx+1)]])

#         if not os.path.exists(outloc+'/encode_DeepLIFT/'):
#             os.makedirs(outloc+'/encode_DeepLIFT/')
        
#         # Speficy output file names
#         outfile_name_at1 = outloc+'/encode_DeepLIFT/RNA_'+str(out_indx)+'.txt'
#         outfile_name_at2 = outloc+'/encode_DeepLIFT/DNA_'+str(out_indx)+'.txt'

#         # Batching testing data
#         batch_size = 256*4
#         steps, batches = batch_iter_DeepLIFT(X_mRNA.values[:,1],
#                                              X_promoter.values[:,1],
#                                              batch_size,
#                                              med_mRNA_len,
#                                              med_promoter_len,
#                                              shuffle=False)

#         for i in range(steps):
            
#             xs, ys = batches.next()

#             # Reshape background
#             xs_background=[]
#             xs_background.append(np.zeros((1,xs[0].shape[1],xs[0].shape[2])))           # xs[0] is the mRNA feature, xs[0].shape[0]=instances, xs[0].shape[1]=max_mRNA_len, xs[0].shape[2]=n_feature_mRNA
#             xs_background.append(np.zeros((1,xs[1].shape[1],xs[1].shape[2])))           # xs[1] is the promoter feature
#             xs_background[0][0,0:med_mRNA_len,0] = 1
#             xs_background[1][0,0:med_promoter_len,0] = 1

#             # Compute DeepLIFT scores
#             explainer = shap.DeepExplainer(new_model, xs_background)
#             shap_values = explainer.shap_values(xs)
#             print('shape_values_size:', shap_values.shape)
#             shap_values = shap_values[0]

#             # Export DeepLIFT scores to text
#             with open(outfile_name_at1, 'a') as f_handle:
#                 input_k=0
#                 for j in range(shap_values[input_k].shape[0]):
#                     seq_indx=xs[input_k][j,:,0]>0
#                     feature_vector=map(str, np.sum(shap_values[input_k][j,seq_indx,:],axis=0))
#                     out_txt=str(out_indx)+'\t'+gene_names[j+i*batch_size]+'\t'+','.join(feature_vector)+'\n'
#                     f_handle.write(out_txt)

#             with open(outfile_name_at2, 'a') as f_handle:
#                 input_k=1
#                 for j in range(shap_values[input_k].shape[0]):
#                     seq_indx=xs[input_k][j,:,0]>0
#                     feature_vector=map(str, np.sum(shap_values[input_k][j,seq_indx,:],axis=0))
#                     out_txt=str(out_indx)+'\t'+gene_names[j+i*batch_size]+'\t'+','.join(feature_vector)+'\n'
#                     f_handle.write(out_txt)

#         # gzip text files
#         os.system("gzip "+outfile_name_at1)
#         os.system("gzip "+outfile_name_at2)