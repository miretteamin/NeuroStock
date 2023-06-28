


import wandb

from tqdm import tqdm
from .bloomberg_model import NeuroStock
from .bloomberg_model import train_gp
from .bloomberg_model import get_gp_model
from .bloomberg_model import get_output_representations
from .bloomberg_model import get_output
from torch_geometric.data import DataLoader as GraphDataLoader
from torch.utils.data import TensorDataset, DataLoader
from .day_graphs import DayGraphs

import torch
import numpy as np
import pandas as pd
import json

with open("../config_files/bloomberg_train_config.json", "r") as f:
    train_config = json.load(f)

all_points = []
for i in range(1,4):
  dataset = DayGraphs(f"./drive/MyDrive/bloomberg_graph_std_trick{i}")
  for k in range(len(dataset)):
    all_points.append(dataset[k])

wandb.finish()
wandb.init(
    project=train_config["project_name"],
    name=train_config["run_name"],
    config=train_config,
    mode=train_config["wandb_mode"])
# warmup_steps=
neurostock = NeuroStock(
    node_emb_size=train_config["node_emb_size"],
    company_emb_size=train_config["node_emb_size"],
    gnn_msg_aggr=train_config["gnn_msg_aggr"],
    use_timeseries_only=train_config["use_timeseries_only"],
    graph_metadata=all_points[0].metadata())

neurostock.to('cuda')
device = next(neurostock.parameters()).device
optimizer =  torch.optim.AdamW(neurostock.parameters(),
                               lr=train_config["lr"],
                               weight_decay=train_config["weight_decay"] )
accs = []
accum_companies = []
gp_accs = []
gp_vars = []
# lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,  total_iters=warmup_steps)

# if train_config["reset_parameters"] and (split_index + 1) % train_config["reset_parameters_freq"]  == 0 :
#   for layer in neurostock.children():
#     if hasattr(layer, 'reset_parameters'):
#         layer.reset_parameters()

train_loader = GraphDataLoader(
    all_points[train_config["start_day"]:train_config["start_day"]+train_config["train_size"]],
    batch_size=train_config["train_batch_size"], shuffle=True)
test_loader = GraphDataLoader(
    all_points[train_config["start_day"]+train_config["train_size"]:train_config["start_day"]+train_config["train_size"]+train_config["test_interval"]],
    batch_size=1, shuffle=False)
best_val_acc = 0
for e in tqdm(range(train_config["n_epochs"])):
    train_losses= []
    neurostock.train()
    train_outs = []
    train_targets = []
    for batch  in train_loader:
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        batch = batch.to(device)
        out = neurostock(batch)
        train_outs.append(out.cpu().detach().unsqueeze(0))
        train_targets.append(batch["target"].cpu().detach().unsqueeze(0))
        loss = neurostock.compute_loss(out, batch["target"])
        optimizer.zero_grad()
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

    neurostock.eval()
    valid_losses = []
    # continue
    valid_outs = []
    valid_targets = []
    with torch.no_grad():

        for i, batch  in enumerate(test_loader):
          batch = batch.to(device)
          out = neurostock(batch)
          valid_outs.append(out.cpu().detach().unsqueeze(0))
          valid_targets.append(batch["target"].cpu().detach().unsqueeze(0))
          loss = neurostock.compute_loss(out, batch["target"])
          valid_losses.append(loss.item())

    valid_outs = torch.cat(valid_outs).numpy()
    valid_targets = torch.cat(valid_targets).numpy()
    acc_per_company = (valid_outs.argmax(-1) == valid_targets).mean(axis=0)
    high_acc_companies = np.where(acc_per_company >= 0.7)[0]

    valid_acc = (valid_outs.argmax(-1) == valid_targets).mean()
    train_outs = torch.cat(train_outs).numpy()
    train_targets = torch.cat(train_targets).numpy()
    train_acc = (train_outs.argmax(-1) == train_targets).mean()
    high_confidence_accs = []
    for out, target in zip(valid_outs, valid_targets):
      predicted_label = out.argmax(-1)
      confidence = out.max(-1)
      predicted_label = predicted_label[confidence.argsort()[-train_config["highest_conf_k"]:]]
      target = target[confidence.argsort()[-train_config["highest_conf_k"]:]]
      high_confidence_accs.append((target ==predicted_label).mean())

    wandb.log({
        "train_loss" : np.mean(train_losses),
        "valid_loss" : np.mean(valid_losses),
        "valid_acc" : valid_acc,
        "high_confidence_acc" : np.mean(high_confidence_accs),
        "valid_n_high" : (acc_per_company >= 0.7).sum(),
        "train_acc" : train_acc,
    })
    if np.mean(high_confidence_accs) > best_val_acc :
      best_val_acc = np.mean(high_confidence_accs)
      torch.save(neurostock, train_config["model_file_name"])

neurostock = torch.load(train_config["model_file_name"])
# representations, target = get_output_representations(neurostock,
#                                                       all_points[train_config["start_day"]+train_config["train_size"]:])

# gp_train_x, gp_train_y = representations[:train_config["gp_train_days"]*616], target[:train_config["gp_train_days"]*616]
# gp_valid_x, gp_valid_y = gp_train_x[-train_config["gp_val_days"]*616:], gp_train_y[-train_config["gp_val_days"]*616:]
# gp_train_x, gp_train_y = gp_train_x[:-train_config["gp_val_days"]*616], gp_train_y[:-train_config["gp_val_days"]*616]

# gp_test_x, gp_test_y = representations[train_config["gp_train_days"]*616:], target[train_config["gp_train_days"]*616:]
# gp_model, likelihood = get_gp_model(gp_train_x, gp_train_y)
# train_dataset = TensorDataset(gp_train_x, likelihood.transformed_targets.T)

# gp_uncertainty_acc, gp_vars = train_gp(gp_model, likelihood, train_dataset, gp_valid_x, gp_valid_y, gp_test_x, gp_test_y, n_epochs=train_config["gp_epochs"], least_var_k=train_config["highest_conf_k"])



long_range_outputs, long_range_true = get_output(neurostock, all_points[train_config["start_day"]+train_config["train_size"]+train_config["test_interval"]:])

for out, target in zip(long_range_outputs, long_range_true):
  predicted_label = out.argmax(-1)
  confidence = out.max(-1)
  predicted_label = predicted_label[confidence.argsort()[-train_config["highest_conf_k"]:]]
  target = target[confidence.argsort()[-train_config["highest_conf_k"]:]]
  high_confidence_accs.append((target ==predicted_label).mean())


torch.save(neurostock, train_config["model_file_name"])
wandb.save(train_config["model_file_name"])
torch.save(optimizer.state_dict(),"./neurostock_optimizer_state_dict.pt")
wandb.save("./neurostock_optimizer_state_dict.pt")
# wandb.save("./best_gp.pt")
# wandb.save("./best_likelihoood.pt")

wandb.log({
           "long_range_acc" : (long_range_outputs.argmax(-1) == long_range_true).mean(),
           "long_range_conf_ac" : np.mean(high_confidence_accs),
           })
wandb.finish()