# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:49:02 2026

Simple Autoencoder to visualize latent embeddings on Batch corrected data

@author: shabnam
"""

# General Module Import
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import matplotlib.cm as cm
import random
from pathlib import Path
import json
import argparse
import copy
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import GroupShuffleSplit
import os
import umap
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd 
import glob
import seaborn as sns
import random
from sklearn.metrics import silhouette_score
# Reproducibility
from utils.seed import set_seed
#Load data
from utils.load_data import load_data
# Data transformation
from utils.data_transformation import abundance_binning




# Study-level train and validation split - Specific to Autoencoder

def study_group_split(X: np.ndarray, study_name: np.ndarray, val_ratio: float = 0.10, seed: int = 42,):

    """
    Split samples into train and validation sets by study - there are around 3k samples

    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx_train, idx_val = next(gss.split(X, groups=study_name))

    n_studies_train = len(np.unique(study_name[idx_train]))
    n_studies_val   = len(np.unique(study_name[idx_val]))
    
    return X[idx_train], X[idx_val], idx_train, idx_val


# Autoencoder architecture

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(),nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,dropout:float):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        latent_dim: int = 64,dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256,128]
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim,dropout)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim,dropout)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


#Autoencoder training
def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    latent_dim: int = 64,
    hidden_dims: list = None,
    epochs: int = 30,
    batch_size: int = 256,
    lrate: float = 1e-3,
    wdecay: float = 0.01,
    dropout: float = 0.2,
   
    patience: int = 10,        # stop after this many epochs with no val improvement
    min_delta: float = 1e-6,   # minimum improvement to count as "better"
    checkpoint_path: str = None,  # where to save the best model weights (.pt file)
   
    device: str = None,
    verbose: bool = True,) -> tuple:
    
    if hidden_dims is None:
        hidden_dims = [512, 384]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
 
    scale = X_train.astype(np.float32).max()
    X_train_norm = X_train.astype(np.float32) / scale
    X_val_norm   = X_val.astype(np.float32)   / scale
 
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_norm)),
        batch_size=batch_size, shuffle=True,)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_norm)),
        batch_size=batch_size, shuffle=False,)
 
    model     = Autoencoder(X_train_norm.shape[1], hidden_dims, latent_dim, dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lrate, weight_decay=wdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.MSELoss()
 
    train_losses, val_losses = [], []
 
  
    best_val_loss   = float("inf")   # best validation MSE seen so far
    best_epoch      = 1              # epoch at which best val loss occurred
    best_state_dict = None           # in-memory copy of the best model weights
    epochs_no_improve = 0            # counter: how many epochs without improvement
 
    for epoch in range(1, epochs + 1):
 
       
        model.train()
        epoch_train = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_train += loss.item() * batch.size(0)
        epoch_train /= len(train_loader.dataset)
 
       
        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x_hat, _ = model(batch)
                epoch_val += criterion(x_hat, batch).item() * batch.size(0)
        epoch_val /= len(val_loader.dataset)
 
        train_losses.append(epoch_train)
        val_losses.append(epoch_val)
        scheduler.step()
 
        if epoch_val < best_val_loss - min_delta:
            best_val_loss   = epoch_val
            best_epoch      = epoch
            epochs_no_improve = 0
 
          
            
            best_state_dict = copy.deepcopy(model.state_dict())
 
            
            if checkpoint_path:
                torch.save(best_state_dict, checkpoint_path)
        else:
            epochs_no_improve += 1
 
        
        
        if epochs_no_improve >= patience:
            
            break
 
  
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n  Loaded best model weights from epoch {best_epoch}.")
 
    return model, train_losses, val_losses, best_epoch, best_val_loss
 


# Extract embeddings

@torch.no_grad()  #Decorator

def extract_embeddings(model: Autoencoder,X_binned: np.ndarray,batch_size: int = 256,device: str = None,) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scale  = X_binned.astype(np.float32).max()
    X_norm = X_binned.astype(np.float32) / scale
    loader = DataLoader(TensorDataset(torch.tensor(X_norm)),batch_size=batch_size, shuffle=False,)

    model.eval()
    embeddings = []
    for (batch,) in loader:
        _, z = model(batch.to(device))
        embeddings.append(z.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# Plotting

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: str  = None,):
    """
    Plot training and validation loss (MSE) curves, and mean-absolute-error (MAE) curves as a proxy for reconstruction accuracy.
    """
    plt.figure(figsize=(8,5))
    
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training and Validation Loss")
    
    plt.legend()
    plt.grid(True)
    

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# MAE for reconstruction to monitor loss convergence only - No relevance to research question

def compute_mae_per_epoch(
    model: Autoencoder,
    X_binned: np.ndarray,
    scale: float,
    batch_size: int = 256,
    device: str = None,) -> float:
    """Return mean absolute error in original bin units for one split."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    X_norm = X_binned.astype(np.float32) / scale
    loader = DataLoader(
        TensorDataset(torch.tensor(X_norm)),
        batch_size=batch_size, shuffle=False,)
    model.eval()
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            # Convert back to bin scale before computing MAE
            mae = (x_hat - batch).abs().mean().item() * scale
            total_mae += mae * batch.size(0)
            n += batch.size(0)
    return total_mae / n


# Main function

def main():
    set_seed(42)
    begin = time.time()

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    data_dir   = project_root / "CMD_data"
    output_dir = project_root / "results"

    parser = argparse.ArgumentParser(description="Simple Autoencoder training")
    parser.add_argument("--train_csv",   type=str, default=str(data_dir / "X_gut_train.csv"), help="Training file")
    parser.add_argument("--labels_csv",  type=str, default=str(data_dir / "metadata_gut.csv"), help="Labels / metadata file")
    parser.add_argument("--output_dir",  type=str, default=str(output_dir / "Simple_AE_Binned"/"disease_covariate"/"dim_final"/"gut_only"), help="Directory to save model outputs")
    parser.add_argument("--bins",        type=int, default=40, help="Number of quantile bins")
    parser.add_argument("--epochs",      type=int, default=40, help="Number of training epochs")
    parser.add_argument("--dropout",      type=float, default=0.2, help="Dropout")
    parser.add_argument("--wdecay",      type=float, default=0.01, help="Weight decay")
    parser.add_argument("--latent_dim",  type=int, default=256, help="Latent space dimensionality")
    parser.add_argument("--batch_size",  type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lrate",          type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--study_col",   type=str, default="study_name", help="Metadata column containing study identifiers")
    parser.add_argument("--val_ratio",   type=float, default=0.10, help="Fraction of studies held out for validation")
    parser.add_argument("--patience",    type=int,   default=10,   help="Early stopping patience (epochs)")
    parser.add_argument("--min_delta",   type=float, default=1e-6, help="Minimum val-loss improvement to reset patience")
    
    args = parser.parse_args()

   
    run_name = f"epoch{args.epochs}_bins{args.bins}_lr{args.lrate}_drop{args.dropout}"
    result_path = Path(args.output_dir) / run_name
    result_path.mkdir(parents=True, exist_ok=True)

    # Load data
    X_path = Path(args.train_csv)
    y_path = Path(args.labels_csv)
   
    abundance, metadata = load_data(X_path, y_path)
   

    # Binned matrix
    
    X_binned   = abundance_binning(abundance, bins=args.bins)
    study_name  = metadata[args.study_col].values
    print(f" Binned matrix: {X_binned.shape}  |  "
          f"unique studies: {len(np.unique(study_name))}")

    # Study split
    
    X_train, X_val, idx_train, idx_val = study_group_split(X_binned, study_name, val_ratio=args.val_ratio, seed=42,)

    # Training
    
    checkpoint_file = str(result_path / "best_simple_ae.pt")
 
    model, train_losses, val_losses, best_epoch, best_val_mse = train_autoencoder(
        X_train, X_val,
        latent_dim      = args.latent_dim,
        hidden_dims     = [256, 128],
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lrate           = args.lrate,
        wdecay          = args.wdecay,
        dropout         = args.dropout,
        patience        = args.patience,
        min_delta       = args.min_delta,
        checkpoint_path = checkpoint_file,  
        device          = str(device),
        verbose         = True,)

    #MAE
    
    scale = X_train.astype(np.float32).max()
    train_mae = compute_mae_per_epoch(model, X_train, scale, device=str(device))
    val_mae   = compute_mae_per_epoch(model, X_val,   scale, device=str(device))
    print(f"  Final train MAE: {train_mae:.4f} bins  |  val MAE: {val_mae:.4f} bins")

    
    # Plot
    
    plot_training_curves(train_losses, val_losses,save_path=str(result_path / "training_curves.png"),)
    
    metrics = {
        "run_name":       run_name,
        "best_epoch":     best_epoch,
        "total_epochs_run": len(train_losses),
        "best_val_mse":   best_val_mse,
        "final_train_mse": train_losses[-1],
        "final_val_mse":   val_losses[-1],
        
        "hyperparameters": {
            "bins":       args.bins,
            "epochs":     args.epochs,
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,},}
    metrics_path = result_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Extract embeddings
    all_embeddings = extract_embeddings(model, X_binned, device=str(device))
    np.save(result_path / "embeddings.npy", all_embeddings)
    
    # Save model
    torch.save(model.state_dict(), result_path / "simple_ae.pt")
   

    elapsed = time.time() - begin
    


if __name__ == "__main__":
    main()
