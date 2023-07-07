import wandb

from tqdm import tqdm
from bloomberg_model import  NeuroStockBloom
from gnn import NeuroStockMultiClass 
from bloomberg_model import get_output
from torch_geometric.data import DataLoader as GraphDataLoader
from day_graphs import DayGraphs
from sklearn.metrics import precision_score, f1_score, recall_score
import torch
from typing import Dict
from typing import Any
import numpy as np
import os

def get_model_metrics(neurostock: NeuroStockBloom, test_loader:GraphDataLoader, config:Dict[str,Any])-> Dict[str, float]:
    valid_losses = []
    # continue
    device = next(neurostock.parameters()).device

    valid_outs = []
    valid_targets = []
    with torch.no_grad():

        for i, batch  in enumerate(test_loader):
            batch = batch.to(device)
            out = neurostock(batch)
            valid_outs.append(out.cpu().detach().unsqueeze(0))
            valid_targets.append(batch[config["gnn_model"]["target_name"]].cpu().detach().unsqueeze(0))
            loss = neurostock.compute_loss(out, batch[config["gnn_model"]["target_name"]])
            valid_losses.append(loss.item())

    valid_outs = torch.cat(valid_outs).numpy()
    valid_targets = torch.cat(valid_targets).numpy()

    valid_acc = (valid_outs.argmax(-1) == valid_targets).mean()
    high_confidence_accs = []
    precisions= []
    biased_precision_high_conf = []
    precisions_conf =[]
    f1s_conf =[]
    f1s =[]
    for out, target in zip(valid_outs, valid_targets):
        predicted_label = out.argmax(-1)
        confidence = out.max(-1)
        increased_pred_conf = confidence[predicted_label == 1]
        increased_target = target[predicted_label == 1]
        increased_target = increased_target[increased_pred_conf.argsort()[-config['gnn_model']["highest_conf_increased_k"]:]]
        predicted_label = predicted_label[confidence.argsort()[-config['gnn_model']["highest_conf_k"]:]]
        high_conf_target = target[confidence.argsort()[-config['gnn_model']["highest_conf_k"]:]]
        high_confidence_accs.append((high_conf_target == predicted_label).mean())
        biased_precision_high_conf.append(precision_score(increased_target, np.ones_like(increased_target)))

        precisions.append(precision_score(target, out.argmax(-1)))
        f1s.append(f1_score(target, out.argmax(-1)))
        precisions_conf.append(precision_score(high_conf_target, predicted_label))
        f1s_conf.append(f1_score(high_conf_target, predicted_label))

    return {
        "valid_loss" : np.mean(valid_losses),
        "valid_acc" : valid_acc,
        "high_confidence_acc" : np.mean(high_confidence_accs),
        "biased_precision_high_conf" : np.mean(biased_precision_high_conf),
        "precision" : np.mean(precisions),
        "precision_high_conf" : np.mean(precisions_conf),
        "f1" : np.mean(f1s),
        "f1_high_conf" : np.mean(f1s_conf),
    }
     
def train_model(config):

    all_points = []
    dataset = DayGraphs(config['graphs']['graph_path'])
    for k in range(len(dataset)):
        all_points.append(dataset[k])

    wandb.finish()
    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        config=config["gnn_model"],
        mode=config["wandb"]["mode"])
    # warmup_steps=
    if config["data"]["dataset"] == "bloomberg":    
        neurostock = NeuroStockBloom(
            node_emb_size=config['gnn_model']["node_emb_size"],
            company_emb_size=config['gnn_model']["node_emb_size"],
            type=config['gnn_model']["type"],
            use_timeseries_only=config['gnn_model']["use_timeseries_only"],
            graph_metadata=all_points[0].metadata())
    else:
        neurostock = NeuroStockMultiClass(num_timeseries_features=config["gnn_model"]["num_timeseries_features"], n_companies=config["gnn_model"]["n_companies"], 
                            node_emb_size=config["gnn_model"]["node_emb_size"], company_emb_size=config["gnn_model"]["company_emb_size"], 
                            article_emb_size=config["gnn_model"]["article_emb_size"], n_industries=config["gnn_model"]["n_industries"], 
                            n_gnn_layers=config["gnn_model"]["n_gnn_layers"], use_timeseries_only=config["gnn_model"]["use_timeseries_only"],
                            type = config["gnn_model"]["type"], lstm = config["gnn_model"]["lstm"], 
                            graph_metadata=all_points[0].metadata())
        
    save_dir= os.path.dirname(config["gnn_model"]["best_model_save_path"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)              
    neurostock.to('cuda')
    device = next(neurostock.parameters()).device
    optimizer =  torch.optim.AdamW(neurostock.parameters(),
                                lr=config['gnn_model']["lr"],
                                weight_decay=config['gnn_model']["weight_decay"] )


    train_loader = GraphDataLoader(
        all_points[config['gnn_model']["start_day"]:config['gnn_model']["start_day"]+config['gnn_model']["train_size"]],
        batch_size=config['gnn_model']["train_batch_size"], shuffle=True)
    test_loader = GraphDataLoader(
        all_points[config['gnn_model']["start_day"]+config['gnn_model']["train_size"]:config['gnn_model']["start_day"]+config['gnn_model']["train_size"]+config['gnn_model']["test_interval"]],
        batch_size=1, shuffle=False)
    
    best_val_metric = 0
    best_val_metrics = {}
    for e in tqdm(range(config['gnn_model']["n_epochs"])):
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
            train_targets.append(batch[config["gnn_model"]["target_name"]].cpu().detach().unsqueeze(0))
            loss = neurostock.compute_loss(out, batch[config["gnn_model"]["target_name"]])
            optimizer.zero_grad()
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

        neurostock.eval()
        train_outs = torch.cat(train_outs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_acc = (train_outs.argmax(-1) == train_targets).mean()
        
        metrics = get_model_metrics(neurostock, test_loader, config)
        metrics["train_loss"] = np.mean(train_losses)
        metrics["train_acc"] = train_acc
        if "std" in config["gnn_model"]["target_name"]:
            eval_metric_name = "precision"
        else:
            eval_metric_name = "valid_acc" 
        wandb.log(metrics)
        if metrics[eval_metric_name] > best_val_metric :
            best_val_metric = metrics[eval_metric_name]
            best_val_metrics = metrics
            os.makedirs("/".join(config['gnn_model']["best_model_save_path"].split("/")[:-1]), exist_ok=True)
            torch.save(neurostock, config['gnn_model']["best_model_save_path"])

    wandb.log({"final_"+k : v for k,v in best_val_metrics.items()})
    neurostock = torch.load(config['gnn_model']["best_model_save_path"])
    wandb.finish()
    return neurostock