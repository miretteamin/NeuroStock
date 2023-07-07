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
import wandb

from tqdm import tqdm
from torch_geometric.data import DataLoader as GraphDataLoader
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
import argparse
import wandb
import numpy as np
import random
import os
import datetime
from tqdm import tqdm
generator = torch.manual_seed(232)
np.random.seed(232)
torch.cuda.manual_seed(232)
torch.cuda.manual_seed_all(232)
random.seed(232)
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood, SoftmaxLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, RFFKernel
from torch.utils.data import TensorDataset, DataLoader


class LSTM(nn.Module):
    def __init__(self, input_size=1, num_layers=2,hidden_size=64, output_size=64, num_steps =15 ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.batch_norm1 = nn.BatchNorm1d(num_steps)

    def forward(self, x):
        # h0 = torch.zeros(2, x.size(0), 100).to(device) # num_layers * num_directions, batch_size, hidden_size
        # c0 = torch.zeros(2, x.size(0), 100).to(device)
        x = self.batch_norm1(x)
        out, _ = self.lstm1(x)
        # out, _ = self.lstm2(out)
        out = F.relu(self.fc1(out[:, -1, :]))
        return out



class GConv(nn.Module):

    def __init__(self, emb_dim:int=64, num_layers:int=2, encode:bool=False, concat_out:bool=False, type="gin", device='cpu', dropout=0.2):

        super(GConv,self).__init__()
        self.num_layers = num_layers
        self.gconv_layers = []
        self.norm_layers = []
        self.encode = encode
        for _ in range(num_layers):
            if type=="transformer":
              self.gconv_layers.append(gnn.TransformerConv(emb_dim, emb_dim, heads=2, concat=False, dropout=dropout, add_self_loops = True).to(device)) # project=True ()
            if type=="gat":
              self.gconv_layers.append(gnn.GATConv(emb_dim, emb_dim, dropout=dropout, add_self_loops = True).to(device)) # project=True ()
            if type=="gin":
              self.gconv_layers.append(gnn.GINConv(nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.PReLU())).to(device)) # project=True ()
            if type=="sage":
              self.gconv_layers.append(gnn.SAGEConv(emb_dim, emb_dim, aggr=["mean", "max"], project=True).to(device)) # project=True ()
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



class NeuroStockBloom(nn.Module):

  def __init__(self,
               num_timeseries_features=1,
               n_companies=617,
               company_emb_size=32,
               node_emb_size=64,
               article_emb_size=768,
               n_industries=14,
               n_gnn_layers=2,
               type="gin",
               use_timeseries_only=False,
               graph_metadata:Tuple=None):
    super(NeuroStockBloom, self).__init__()
    """
    company node representation will be a sum of its embedding and the output of the timeseries model (in this case it's an LSTM)
    """
    self.num_timeseries_features = num_timeseries_features
    self.n_companies = n_companies
    self.company_emb_size = company_emb_size
    self.node_emb_size = node_emb_size
    self.article_emb_size = article_emb_size
    self.n_industries = n_industries
    self.type = type
    self.n_gnn_layers = n_gnn_layers
    self.use_timeseries_only = use_timeseries_only
    self.lstm = LSTM(input_size=num_timeseries_features,  hidden_size=company_emb_size, output_size=company_emb_size).to(torch.float)

    if graph_metadata is None:
      raise("You need to pass HeteroData.metadata()")
    self.company_embedding = nn.Embedding(n_companies, company_emb_size).to(torch.float)
    self.project_article = nn.Linear(article_emb_size, node_emb_size).to(torch.float)

    # to_hetero transforms normal gnn aggregation layer to a heterogeneous aggregation layer
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.to_hetero_transformer.to_hetero
    self.g_conv = gnn.to_hetero(GConv(emb_dim=node_emb_size, type=type, num_layers=n_gnn_layers), graph_metadata).to(torch.float)
    self.classifier = nn.Linear(node_emb_size, 2).to(torch.float)

  def forward(self, hetero_x:HeteroData, return_representations=False):
    companies_timeseries = self.lstm(hetero_x["company_timeseries"][:,:, -2:-1].to(torch.float))
    if self.use_timeseries_only:
      out = self.classifier(companies_timeseries)
      if return_representations:
        return out, companies_timeseries
      return out

    hetero_x["sentence"].x = self.project_article(hetero_x["sentence"].x.to(torch.float))
    companies = self.company_embedding(hetero_x["company"].x)
    # print(hetero_x["company_timeseries"][:,:, -2:-1].to(torch.double).shape, hetero_x["company_timeseries"][:,:, -2:-1].to(torch.float).dtype)
    # company_timeseries is of shape (n_companies*batch_size, n_days, n_features)  the features are "open", "high", "low", "close", "volume"
    hetero_x["company"].x = companies_timeseries + companies  #companies are in shape (n_companies*batch_size, node_emb_size)

    for k in hetero_x.edge_index_dict.keys():
      hetero_x[k].edge_index = hetero_x[k].edge_index.to(torch.int64)
    graph = self.g_conv(hetero_x.x_dict, hetero_x.edge_index_dict)
    out = self.classifier(graph["company"])
    if return_representations:
      return out, graph["company"]
    return out

  def compute_loss(self, out, target):
    loss = F.cross_entropy(out, target)
    return loss


def get_output(neurostock:NeuroStockBloom, points:List[HeteroData], target_name="target"):
  data_loader = GraphDataLoader(
                points,
                batch_size=1, shuffle=False)
  neurostock.eval()
  device = next(neurostock.parameters()).device
  valid_losses = []
  # continue
  valid_outs = []
  valid_targets = []
  with torch.no_grad():
      for i, batch  in enumerate(data_loader):
        # if i > 2: break
        batch = batch.to(device)
        out = neurostock(batch)
        # print(out.squeeze(-1).unsqueeze(0))
        # print(batch["target"].shape)
        # break
        valid_outs.append(out.unsqueeze(0).cpu().detach())
        valid_targets.append(batch[target_name].unsqueeze(0).cpu().detach())
        loss = neurostock.compute_loss(out, batch[target_name])
        valid_losses.append(loss.item())
  valid_outs = torch.cat(valid_outs).numpy()
  valid_targets = torch.cat(valid_targets).numpy()
  return valid_outs, valid_targets



