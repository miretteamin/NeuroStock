from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops ,
)
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import scipy.sparse as sp
from torch import nn
import pandas as pd
# import networkx as nx
from typing import List , Dict,Tuple
from torch_geometric.nn import SAGEConv, to_hetero
import torch

import matplotlib.pyplot as plt
import torch_geometric.nn as  gnn
from torch_geometric.data  import Data
from torch_geometric.data import HeteroData

from torch.nn import functional as F
import torch
from torch.optim.adamw import AdamW
import time
from argparse import ArgumentParser
import numpy as np
import random
from tqdm import tqdm
generator = torch.manual_seed(232)
np.random.seed(232)
torch.cuda.manual_seed(232)
torch.cuda.manual_seed_all(232)
random.seed(232)
from lstm import LSTM

class GConv(nn.Module):

    def __init__(self, emb_dim:int=64, num_layers:int=2, type = 'sage', encode:bool=True, concat_out:bool=False, device='cpu', dropout=0.2):

        super(GConv,self).__init__()
        self.num_layers = num_layers
        self.gconv_layers = []
        self.norm_layers = []
        self.encode = encode
        for _ in range(num_layers):
            if type == 'sage':
                self.gconv_layers.append(gnn.SAGEConv(emb_dim, emb_dim, dropout=dropout, project=True).to(device)) # project=True ()
            elif type == 'gat':
                self.gconv_layers.append(gnn.GATConv(emb_dim, emb_dim, dropout=dropout, project=True, add_self_loops = False).to(device)) # project=True ()
            elif type == 'transformer_conv':
                self.gconv_layers.append(gnn.TransformerConv(emb_dim, emb_dim, heads=2, concat=False, dropout=dropout, add_self_loops = True).to(device)) # project=True ()
            if self.encode:
                self.norm_layers.append(nn.LayerNorm(emb_dim).to(device))
        self.gconv_layers = nn.ModuleList(self.gconv_layers)
        self.norm_layers = nn.ModuleList(self.norm_layers)

        self.concat_out = concat_out

    def forward(self, x, edge_index):

        outs = []
        if self.encode:
            outs.append(self.norm_layers[0](self.gconv_layers[0](x, edge_index)))
        else:
            outs.append(self.gconv_layers[0](x, edge_index))
        for i in range(1,self.num_layers):
            if self.encode:
                outs.append(self.norm_layers[i](self.gconv_layers[i](outs[-1], edge_index)))
            else:
                outs.append(self.gconv_layers[i](outs[-1], edge_index))
        if self.concat_out:
            return torch.cat(outs, dim = -1)

        return outs[-1]


class NeuroStock(nn.Module):

    def __init__(self,
                num_timeseries_features=1,
                n_companies=310,
                company_emb_size=32,
                node_emb_size=64,
                article_emb_size=768,
                n_industries=14,
                n_gnn_layers=3,
                use_timeseries_only=False,
                type = 'sage',	
                graph_metadata:Tuple=None):
        super(NeuroStock, self).__init__()
        """
        company node representation will be a concatenation of its embedding and the output of the timeseries model (in this case it's an LSTM)
        """
        self.num_timeseries_features = num_timeseries_features
        self.n_companies = n_companies
        self.company_emb_size = company_emb_size
        self.node_emb_size = node_emb_size
        self.article_emb_size = article_emb_size
        self.n_industries = n_industries
        self.n_gnn_layers = n_gnn_layers
        self.use_timeseries_only = use_timeseries_only
        self.lstm = LSTM(input_size=num_timeseries_features,  hidden_size=company_emb_size, output_size=company_emb_size).to(torch.float)

        if graph_metadata is None:
            raise("You need to pass HeteroData.metadata()")
    
        self.company_embedding = nn.Embedding(n_companies, company_emb_size).to(torch.float)
        self.industry_embedding = nn.Embedding(n_industries, node_emb_size).to(torch.float)
        self.project_article = nn.Linear(article_emb_size, node_emb_size).to(torch.float)

        # to_hetero transforms normal gnn aggregation layer to a heterogeneous aggregation layer
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.to_hetero_transformer.to_hetero
        self.g_conv = gnn.to_hetero(GConv(emb_dim=node_emb_size, num_layers=n_gnn_layers, type = type), graph_metadata).to(torch.float)

        self.classifier = nn.Sequential(nn.Dropout(0.2),nn.Linear(node_emb_size, 1)).to(torch.float)

    def forward(self, hetero_x:HeteroData):
        companies_timeseries = self.lstm(hetero_x["company_timeseries"][:,:, -2:-1].to(torch.float))
        # if self.use_timeseries_only:
        out = F.sigmoid(self.classifier(companies_timeseries))
        return out
        # print(hetero_x["company_timeseries"][:,:, -2:-1].to(torch.double).shape, hetero_x["company_timeseries"][:,:, -2:-1].to(torch.float).dtype)
        # company_timeseries is of shape (n_companies*batch_size, n_days, n_features)  the features are "open", "high", "low", "close", "volume"

    def compute_loss(self, out, target):
        loss = F.binary_cross_entropy(out.reshape(-1), target.float())
        return loss