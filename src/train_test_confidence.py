import wandb

from tqdm import tqdm
from .bloomberg_model import  NeuroStockBloom
from .gnn import NeuroStockMultiClass 
from .bloomberg_model import get_output
from torch_geometric.data import DataLoader as GraphDataLoader
from .day_graphs import DayGraphs

import torch
import numpy as np


def train_model(train_config):

    all_points = []
    if train_config["data"]["dataset"] == "bloomberg":    
        for i in range(1,4):
            dataset = DayGraphs(f"{train_config['graphs']['bloomberg_graph_path']}/bloomberg_graph_{i}")
            for k in range(len(dataset)):
                all_points.append(dataset[k])
    else :
        dataset = DayGraphs(train_config['graphs']['graph_path'])
        for k in range(len(dataset)):
            all_points.append(dataset[k])

    wandb.finish()
    wandb.init(
        project=train_config["wandb"]["project"],
        name=train_config["wandb"]["run_name"],
        config=train_config["gnn_model"],
        mode=train_config["wandb"]["mode"])
    # warmup_steps=
    if train_config["data"]["dataset"] == "bloomberg":    
        neurostock = NeuroStockBloom(
            node_emb_size=train_config['gnn_model']["node_emb_size"],
            company_emb_size=train_config['gnn_model']["node_emb_size"],
            gnn_msg_aggr=train_config['gnn_model']["gnn_msg_aggr"],
            use_timeseries_only=train_config['gnn_model']["use_timeseries_only"],
            graph_metadata=all_points[0].metadata())
    else:
        neurostock = NeuroStockMultiClass(num_timeseries_features=train_config["gnn_model"]["num_timeseries_features"], n_companies=train_config["gnn_model"]["n_companies"], 
                            node_emb_size=train_config["gnn_model"]["node_emb_size"], company_emb_size=train_config["gnn_model"]["company_emb_size"], 
                            article_emb_size=train_config["gnn_model"]["article_emb_size"], n_industries=train_config["gnn_model"]["n_industries"], 
                            n_gnn_layers=train_config["gnn_model"]["n_gnn_layers"], use_timeseries_only=train_config["gnn_model"]["use_timeseries_only"],
                            type = train_config["gnn_model"]["type"], lstm = train_config["gnn_model"]["lstm"], 
                            graph_metadata=all_points[0].metadata())
                  
    neurostock.to('cuda')
    device = next(neurostock.parameters()).device
    optimizer =  torch.optim.AdamW(neurostock.parameters(),
                                lr=train_config['gnn_model']["lr"],
                                weight_decay=train_config['gnn_model']["weight_decay"] )


    train_loader = GraphDataLoader(
        all_points[train_config['gnn_model']["start_day"]:train_config['gnn_model']["start_day"]+train_config['gnn_model']["train_size"]],
        batch_size=train_config['gnn_model']["train_batch_size"], shuffle=True)
    test_loader = GraphDataLoader(
        all_points[train_config['gnn_model']["start_day"]+train_config['gnn_model']["train_size"]:train_config['gnn_model']["start_day"]+train_config['gnn_model']["train_size"]+train_config['gnn_model']["test_interval"]],
        batch_size=1, shuffle=False)
    
    best_val_acc = 0
    for e in tqdm(range(train_config['gnn_model']["n_epochs"])):
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

        valid_acc = (valid_outs.argmax(-1) == valid_targets).mean()
        train_outs = torch.cat(train_outs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_acc = (train_outs.argmax(-1) == train_targets).mean()
        high_confidence_accs = []
        for out, target in zip(valid_outs, valid_targets):
            predicted_label = out.argmax(-1)
            confidence = out.max(-1)
            predicted_label = predicted_label[confidence.argsort()[-train_config['gnn_model']["highest_conf_k"]:]]
            target = target[confidence.argsort()[-train_config['gnn_model']["highest_conf_k"]:]]
            high_confidence_accs.append((target ==predicted_label).mean())

        wandb.log({
            "train_loss" : np.mean(train_losses),
            "valid_loss" : np.mean(valid_losses),
            "valid_acc" : valid_acc,
            "high_confidence_acc" : np.mean(high_confidence_accs),
            "train_acc" : train_acc,
        })
        if np.mean(high_confidence_accs) > best_val_acc :
            best_val_acc = np.mean(high_confidence_accs)
            torch.save(neurostock, train_config['gnn_model']["best_model_save_path"])

    if  "use_test_set" in train_config["gnn_model"].keys():
        if train_config["gnn_model"]["use_test_set"]:
            neurostock = torch.load(train_config['gnn_model']["best_model_save_path"])
            long_range_outputs, long_range_true = get_output(neurostock, all_points[train_config['gnn_model']["start_day"]+train_config['gnn_model']["train_size"]+train_config['gnn_model']["test_interval"]:])

            for out, target in zip(long_range_outputs, long_range_true):
                predicted_label = out.argmax(-1)
                confidence = out.max(-1)
                predicted_label = predicted_label[confidence.argsort()[-train_config['gnn_model']["highest_conf_k"]:]]
                target = target[confidence.argsort()[-train_config['gnn_model']["highest_conf_k"]:]]
                high_confidence_accs.append((target ==predicted_label).mean())

            wandb.log({
                    "long_range_acc" : (long_range_outputs.argmax(-1) == long_range_true).mean(),
                    "long_range_conf_ac" : np.mean(high_confidence_accs),
                    })
    wandb.log({"best_val_acc": best_val_acc})

    torch.save(neurostock, train_config['gnn_model']["best_model_save_path"])
    wandb.save(train_config['gnn_model']["best_model_save_path"])
    # wandb.save("./best_gp.pt")
    # wandb.save("./best_likelihoood.pt")

    
    wandb.finish()
    return neurostock