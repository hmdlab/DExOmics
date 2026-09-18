import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_score, recall_score, confusion_matrix
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



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # CE without class weights
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # p_t
        pt = torch.exp(-ce_loss)

        # focal term
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


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
        y_prob.extend(scores.detach().cpu().numpy())
        pred = torch.argmax(scores, dim=1)
        y_pred.extend(pred.detach().cpu().numpy())
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
            y_preds.extend(pred.detach().cpu().numpy())
                
            loss = criterion(outputs, labels)
            
            # Update the running loss and predictions
            running_loss += loss.item()
            
            y_labels.extend(labels.cpu().numpy())
            y_probs.extend(scores.detach().cpu().numpy())
    
    # Validation loss
    epoch_loss = running_loss / data_steps

    # Evaluation metrics
    '''
    auc_roc = roc_auc_score(y_labels, y_probs, average='weighted', multi_class='ovr')
    accuracy = accuracy_score(y_labels, y_preds)
    f1 = f1_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
    pr_auc = average_precision_score(y_labels, y_probs, average='macro')
    '''
    auc_roc = roc_auc_score(y_labels, y_probs, average='macro', multi_class='ovr')
    accuracy = accuracy_score(y_labels, y_preds)
    f1 = f1_score(y_labels, y_preds, average='macro', zero_division=np.nan)
    pr_auc = average_precision_score(y_labels, y_probs, average='macro')
    recalls = recall_score(y_labels, y_preds,average=None)
    cm = confusion_matrix(y_labels, y_preds)


    return auc_roc, accuracy, f1, pr_auc, recalls, cm, epoch_loss
    


def extract_embeddings(model, data_steps, data_batches, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        data_iter = iter(data_batches)
        for _ in range(data_steps):
            X = next(data_iter)
            x_mRNA = X[0][0]
            x_mRNA = x_mRNA.view(
                x_mRNA.shape[0],
                1,
                x_mRNA.shape[1],
                x_mRNA.shape[2]
            ).to(device)

            x_promoter = X[0][1]
            x_promoter = x_promoter.view(
                x_promoter.shape[0],
                1,
                x_promoter.shape[1],
                x_promoter.shape[2]
            ).to(device)

            x_tcga = X[1].to(device)
            y = X[2]

            _, embedding = model(
                x_mRNA,
                x_promoter,
                x_tcga,
                return_embedding=True
            )

            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(y.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    return embeddings, labels



    
        
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
            y_pred.extend(pred.detach().cpu().numpy())            
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(scores.detach().cpu().numpy())
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
                y_preds.extend(pred.detach().cpu().numpy())
                
            loss = criterion(outputs, labels)
            
            # Update the running loss and predictions
            running_loss += loss.item()
            batch_num += 1
            
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(scores.detach().cpu().numpy())
            
    
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
            f1_scores.append(f1_score(y_true, y_pred))

        # Set the threshold to the value that produces the largest accuracy for prediction
        y_pred = (y_probs >= thresholds[f1_scores.index(max(f1_scores))]).astype(bool)

        # Calculate ACC and F1
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        T = thresholds[f1_scores.index(max(f1_scores))]
        
    else:
        auc_roc = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
        
        # Assign the class with the largest probability
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
            y_preds.extend(pred.detach().cpu().numpy())
                
            loss = criterion(outputs, labels)
            
            # Update the running loss and predictions
            running_loss += loss.item()
            
            y_labels.extend(labels.cpu().numpy())
            y_probs.extend(scores.detach().cpu().numpy())
    
    # Validation loss
    epoch_loss = running_loss / data_steps

    # Evaluation metrics
    auc_roc = roc_auc_score(y_labels, y_probs, average='weighted', multi_class='ovr')
    accuracy = accuracy_score(y_labels, y_preds)
    f1 = f1_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
    precision = precision_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
        
    return auc_roc, accuracy, f1, precision, epoch_loss




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
            y_preds.extend(pred.detach().cpu().numpy())
                
            loss = criterion(outputs, labels)
            
            # Update the running loss and predictions
            running_loss += loss.item()
            
            y_labels.extend(labels.cpu().numpy())
            y_probs.extend(scores.detach().cpu().numpy())
    
    # Validation loss
    epoch_loss = running_loss / data_steps

    # Evaluation metrics
    if level == 'low':
        return epoch_loss
    else:
        auc_roc = roc_auc_score(y_labels, y_probs, average='weighted', multi_class='ovr')
        accuracy = accuracy_score(y_labels, y_preds)
        f1 = f1_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
        precision = precision_score(y_labels, y_preds, average='weighted', zero_division=np.nan)
            
        return auc_roc, accuracy, f1, precision, epoch_loss
    

