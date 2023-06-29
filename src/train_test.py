from .gnn import NeuroStock
import torch
from tqdm.notebook  import tqdm
import numpy as np
import wandb
import os

def train_model(train_loader, test_loader, config):

    wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            mode=config["wandb"]["mode"])

    n_epochs = config["gnn_model"]["n_epochs"]
    
    first_batch = next(iter(train_loader))
    train_first_element = first_batch[0]

    neurostock = NeuroStock(num_timeseries_features=config["gnn_model"]["num_timeseries_features"], n_companies=config["gnn_model"]["n_companies"], 
                            node_emb_size=config["gnn_model"]["node_emb_size"], company_emb_size=config["gnn_model"]["company_emb_size"], 
                            article_emb_size=config["gnn_model"]["article_emb_size"], n_industries=config["gnn_model"]["n_industries"], 
                            n_gnn_layers=config["gnn_model"]["n_gnn_layers"], use_timeseries_only=config["gnn_model"]["use_timeseries_only"],
                            type = config["gnn_model"]["type"], lstm = config["gnn_model"]["lstm"], 
                            graph_metadata=train_first_element.metadata())
                            
    neurostock.to('cuda')
    device = next(neurostock.parameters()).device
    optimizer =  torch.optim.AdamW(neurostock.parameters(), lr=0.001)
    best_valid_acc = 0
    save_dir= os.path.dirname(config["gnn_model"]["best_model_save_path"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,  total_iters=warmup_steps)
    for e in tqdm(range(n_epochs)):
        train_losses= []
        neurostock.train()
        train_outs = []
        train_targets = []
        for batch  in train_loader:
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            batch = batch.to(device)
            out = neurostock(batch)
            train_outs.append(out.cpu().detach().reshape(-1))
            train_targets.append(batch["target"].cpu().detach().reshape(-1))
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
            for batch  in test_loader:
                batch = batch.to(device)
                out = neurostock(batch)
                valid_outs.append(out.reshape(-1).cpu().detach())
                valid_targets.append(batch["target"].reshape(-1).cpu().detach())
                loss = neurostock.compute_loss(out, batch["target"])
                valid_losses.append(loss.item())

        valid_outs = torch.cat(valid_outs).numpy()
        valid_targets = torch.cat(valid_targets).numpy()
        valid_acc = ((valid_outs >= 0.5) == valid_targets).mean()
        train_outs = torch.cat(train_outs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_acc = ((train_outs >= 0.5) == train_targets).mean()
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(neurostock, config["gnn_model"]["best_model_save_path"])  # Save the best model weights
    
        wandb.log({
                "train_loss" : np.mean(train_losses),
                "valid_loss" : np.mean(valid_losses),
                "valid_acc" : valid_acc,
                "train_acc" : train_acc
        })

    wandb.save(config['gnn_model']["best_model_save_path"])
    wandb.finish()
    
    torch.save(neurostock, config["gnn_model"]["last_model_save_path"])
    return neurostock