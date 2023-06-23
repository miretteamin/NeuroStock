from gnn import NeuroStock
import torch
from tqdm.notebook  import tqdm
import numpy as np
import wandb

def train_model(train_loader, testnode_emb_size, company_emb_size, graph_metadata, n_epochs=50):

    wandb.init(
        project="test_sage",
        name="test_titles",
        mode="online")
    
    neurostock = NeuroStock(node_emb_size=128, company_emb_size=64, graph_metadata=train[0].metadata())
    neurostock.to('cuda')
    device = next(neurostock.parameters()).device
    optimizer =  torch.optim.AdamW(neurostock.parameters(), lr=0.001)
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
        wandb.log({
                "train_loss" : np.mean(train_losses),
                "valid_loss" : np.mean(valid_losses),
                "valid_acc" : valid_acc,
                "train_acc" : train_acc
        })

    wandb.finish()
