import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_score



class ConcatedNet(nn.Module):
    """Define the CNN network for the ENCODE part data"""
    def __init__(self, params, dropout_prob=0.5):
        super(ConcatedNet, self).__init__()

        # ------------------------------------- ENCODE part -------------------------------------
        # Define mRNA model
        self.RNA_layers = nn.Sequential()
        self.RNA_layers.add_module('rna_conv1', 
                                   nn.Conv2d(in_channels=int(params['mRNA_in_channels']),   # in_channels is the input depth
                                             out_channels=int(params['RNA_n_channel_1st']),
                                             kernel_size=(1, int(params['n_feature_mRNA']))))
        if params['ConvRelu'] == 'Yes':
            self.RNA_layers.add_module('rna_relu', nn.ReLU())
        for i in range(int(params['RNA_n_ConvLayer']) - 1):
            self.RNA_layers.add_module(f"rna_conv{i+2}",
                                       nn.Conv2d(in_channels=int(params['RNA_n_channel_1st']),
                                                 out_channels=int(params['RNA_n_channel_1st']),
                                                 kernel_size=(int(params['RNA_conv_kernel']), 1)))
            if params['ConvRelu'] == 'Yes':
                self.RNA_layers.add_module('rna_relu', nn.ReLU())


        # Define promoter model
        self.DNA_layers = nn.Sequential()
        self.DNA_layers.add_module('dna_conv1', 
                                   nn.Conv2d(in_channels=int(params['promoter_in_channels']),   # in_channels is the input depth
                                             out_channels=int(params['DNA_n_channel_1st']),
                                             kernel_size=(1, int(params['n_feature_promoter']))))
        if params['ConvRelu'] == 'Yes':
            self.DNA_layers.add_module('dna_relu', nn.ReLU())
        for i in range(int(params['DNA_n_ConvLayer']) - 1):
            self.DNA_layers.add_module(f"dna_conv{i+2}",
                                       nn.Conv2d(in_channels=int(params['DNA_n_channel_1st']),
                                                 out_channels=int(params['DNA_n_channel_1st']),
                                                 kernel_size=(int(params['DNA_conv_kernel']), 1)))
            if params['ConvRelu'] == 'Yes':
                self.DNA_layers.add_module('dna_relu', nn.ReLU())


        # Concate two vectors to form dense layers
        self.fc_layers = nn.Sequential()
        for i in range(int(params['last_ConvFClayer'])):
            self.fc_layers.add_module(f"fc{i+1}",
                                      nn.Linear(in_features=int(params['encode_last_n_channel']),
                                                out_features=int(params['encode_last_n_channel'])))
            if params['FullRelu'] == 'Yes':
                self.fc_layers.add_module(f"relu_fc{i+1}", nn.ReLU())
            # self.fc_layers.add_module(f"bn{i+1}", nn.BatchNorm1d(int(params['encode_last_n_channel'])))
        
        # ------------------------------------- TCGA part -------------------------------------
        self.tcga_fc1 = nn.Linear(int(params['tcga_input_size']), int(params['tcga_hidden_size']))
        self.tcga_fc2 = nn.Linear(int(params['tcga_hidden_size']), int(params['tcga_hidden_size']))
        self.tcga_fc3 = nn.Linear(int(params['tcga_hidden_size']), int(params['tcga_hidden_size']))
        self.output_layer = nn.Linear(int(params['last_n_channel']), int(params['n_out']))
        self.dropout = nn.Dropout(dropout_prob)
        self.init_weights()

    def forward(self, x_rna, x_dna, x_tcga):
        x_rna = self.RNA_layers(x_rna)
        # Eliminate the dimension of the changing height (due to different max mRNA len in each batch)
        x_rna = torch.sum(x_rna, dim=2)
        x_rna = torch.flatten(x_rna, 1)
        x_dna = self.DNA_layers(x_dna)
        x_dna = torch.sum(x_dna, dim=2)
        x_dna = torch.flatten(x_dna, 1)
        x_encode_combined = torch.cat((x_rna, x_dna), dim=1)
        x_encode_combined = self.fc_layers(x_encode_combined)
        x_encode_combined = self.dropout(x_encode_combined)

        x_tcga = F.relu(self.tcga_fc1(x_tcga))
        x_tcga = F.relu(self.tcga_fc2(x_tcga))
        x_tcga = F.relu(self.tcga_fc3(x_tcga))
        x_tcga = self.dropout(x_tcga)
        x = torch.cat((x_tcga, x_encode_combined), dim=1)
        x = self.output_layer(x)
        return x
    
    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)
    
    def l2_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(torch.square(param))
        return l2_loss



class tcga_MLP(nn.Module):
    """Define the MLP network"""
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(tcga_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.init_weights()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)
    
    def l2_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(torch.square(param))
        return l2_loss



class interactome_CNN(nn.Module):
    """Define the CNN network for the ENCODE part data"""
    def __init__(self, params):
        super(interactome_CNN, self).__init__()

        # Define mRNA model
        self.RNA_layers = nn.Sequential()
        self.RNA_layers.add_module('rna_conv1', 
                                   nn.Conv2d(in_channels=int(params['mRNA_in_channels']),   # in_channels is the input depth
                                             out_channels=int(params['RNA_n_channel_1st']),
                                             kernel_size=(1, int(params['n_feature_mRNA']))))
        if params['ConvRelu'] == 'Yes':
            self.RNA_layers.add_module('rna_relu', nn.ReLU())
        for i in range(int(params['RNA_n_ConvLayer']) - 1):
            self.RNA_layers.add_module(f"rna_conv{i+2}",
                                       nn.Conv2d(in_channels=int(params['RNA_n_channel_1st']),
                                                 out_channels=int(params['RNA_n_channel_1st']),
                                                 kernel_size=(int(params['RNA_conv_kernel']), 1)))
            if params['ConvRelu'] == 'Yes':
                self.RNA_layers.add_module('rna_relu', nn.ReLU())


        # Define promoter model
        self.DNA_layers = nn.Sequential()
        self.DNA_layers.add_module('dna_conv1', 
                                   nn.Conv2d(in_channels=int(params['promoter_in_channels']),   # in_channels is the input depth
                                             out_channels=int(params['DNA_n_channel_1st']),
                                             kernel_size=(1, int(params['n_feature_promoter']))))
        if params['ConvRelu'] == 'Yes':
            self.DNA_layers.add_module('dna_relu', nn.ReLU())
        for i in range(int(params['DNA_n_ConvLayer']) - 1):
            self.DNA_layers.add_module(f"dna_conv{i+2}",
                                       nn.Conv2d(in_channels=int(params['DNA_n_channel_1st']),
                                                 out_channels=int(params['DNA_n_channel_1st']),
                                                 kernel_size=(int(params['DNA_conv_kernel']), 1)))
            if params['ConvRelu'] == 'Yes':
                self.DNA_layers.add_module('dna_relu', nn.ReLU())


        # Concate two vectors to form dense layers
        self.fc_layers = nn.Sequential()
        for i in range(int(params['last_ConvFClayer'])):
            self.fc_layers.add_module(f"fc{i+1}",
                                      nn.Linear(in_features=int(params['encode_last_n_channel']),
                                                out_features=int(params['encode_last_n_channel'])))
            if params['FullRelu'] == 'Yes':
                self.fc_layers.add_module(f"relu_fc{i+1}", nn.ReLU())
            # self.fc_layers.add_module(f"bn{i+1}", nn.BatchNorm1d(int(params['encode_last_n_channel'])))
        self.output_layer = nn.Linear(int(params['encode_last_n_channel']), int(params['n_out']))
        self.init_weights()

    def forward(self, x_rna, x_dna):
        x_rna = self.RNA_layers(x_rna)
        # Eliminate the dimension of the changing height (due to different max mRNA len in each batch)
        x_rna = torch.sum(x_rna, dim=2)
        x_rna = torch.flatten(x_rna, 1)
        x_dna = self.DNA_layers(x_dna)
        x_dna = torch.sum(x_dna, dim=2)
        x_dna = torch.flatten(x_dna, 1)
        x_encode_combined = torch.cat((x_rna, x_dna), dim=1)
        # print("x_encode_combined:", x_encode_combined.shape)
        x_encode_combined = self.fc_layers(x_encode_combined)
        x = self.output_layer(x_encode_combined)
        return x
    
    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)
    
    def l2_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(torch.square(param))
        return l2_loss
