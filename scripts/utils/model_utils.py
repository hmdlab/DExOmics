import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_score
from utils.data_tool import *
from utils.data_tool import *



def model_prob(model, steps, batches, device, data_type="full"):
    y_true = []
    y_prob = []
    y_pred = []
    data_iter = iter(batches)
    for i in range(steps):
        X = next(data_iter)
        x_mRNA = X[0][0]
        x_mRNA = x_mRNA.view(x_mRNA.shape[0], 1, x_mRNA.shape[1], x_mRNA.shape[2]).to(device) # batch_size*channel*height*width
        x_promoter = X[0][1]
        x_promoter = x_promoter.view(x_promoter.shape[0], 1, x_promoter.shape[1], x_promoter.shape[2]).to(device)
        if data_type == "full":
            x_tcga = X[1].to(device)
            outputs = model(x_mRNA, x_promoter, x_tcga)           
        elif data_type == "encode":
            outputs = model(x_mRNA, x_promoter)
        labels = X[2].to(device) 
        softmax = nn.Softmax(dim=1)
        scores = softmax(outputs)
        y_prob.extend(scores.detach().numpy())
        pred = torch.argmax(scores, dim=1)
        y_pred.extend(pred.detach().numpy())
        y_true.extend(labels.cpu().numpy())
    return np.array(y_true), np.array(y_prob), np.array(y_pred)



# Define the evaluation function for binary classification
def evaluate(model, data_steps, data_batches, criterion, device):
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
            x_tcga = X[1].to(device)
            labels = X[2].to(device)
            outputs = model(x_mRNA, x_promoter, x_tcga)
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
    auc_roc = roc_auc_score(y_labels, y_probs, average='weighted', multi_class='ovr')
    accuracy = accuracy_score(y_labels, y_preds)
    f1 = f1_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
    precision = precision_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
        
    return auc_roc, accuracy, f1, precision, epoch_loss




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


    

def pancan_evaluate(model, data_steps, data_batches, criterion, device):
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
    

    
def tcga_model_prob(model, dataloader, device):
    y_true = []
    y_prob = []
    y_pred = []
    with torch.no_grad():
        batch_num = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            softmax = nn.Softmax(dim=1)
            scores = softmax(outputs)
            # Assign the class with the largest probability
            pred = torch.argmax(scores, dim=1)
            y_pred.extend(pred.detach().numpy())            
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(scores.detach().numpy())
    return np.array(y_true), np.array(y_prob), np.array(y_pred)



def tcga_evaluate(model, dataloader, criterion, device, bi_class=True):
    """This function is for model prediction and evaluation"""
    model.eval() # Set model to evaluation mode
    y_true = []
    y_probs = []
    y_preds = []
    running_loss = 0
    
    with torch.no_grad():
        batch_num = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            softmax = nn.Softmax(dim=1)
            if bi_class:
                scores = softmax(outputs)[:, 1]
            else:
                scores = softmax(outputs)
                # Assign the class with the largest probability
                pred = torch.argmax(scores, dim=1)
                y_preds.extend(pred.detach().numpy())
                
            loss = criterion(outputs, labels)
            
            # Update the running loss and predictions
            running_loss += loss.item()
            batch_num += 1
            
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(scores.detach().numpy())
            
    
    # Validation loss
    epoch_loss = running_loss / batch_num
    
    if bi_class:     
        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y_true, y_probs)

        # Calculate AP (average precision)
        AP = average_precision_score(y_true, y_probs)

        # Set different thresholds
        thresholds = np.linspace(0, 1, 101)
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            f1_scores.append(f1)

        # Set the threshold to the value that produces the largest accuracy for prediction
        y_pred = (y_probs >= thresholds[f1_scores.index(max(f1_scores))]).astype(bool)

        # Calculate ACC and F1
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        T = thresholds[f1_scores.index(max(f1_scores))]
        
    else:
        auc_roc = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
        
        # Assign the class with the largest probability
        scores = torch.argmax(scores, dim=1)
        accuracy = accuracy_score(y_true, y_preds)
        f1 = f1_score(y_true, y_preds, average='weighted', zero_division=np.nan)
        precision = precision_score(y_true, y_preds, average='weighted', zero_division=np.nan)
        
    if bi_class:
        return auc_roc, AP, accuracy, f1, T, epoch_loss
    else:
        return auc_roc, accuracy, f1, precision, epoch_loss



def encode_evaluate(model, data_steps, data_batches, criterion, device):
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
    auc_roc = roc_auc_score(y_labels, y_probs, average='weighted', multi_class='ovr')
    accuracy = accuracy_score(y_labels, y_preds)
    f1 = f1_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
    precision = precision_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
        
    return auc_roc, accuracy, f1, precision, epoch_loss
