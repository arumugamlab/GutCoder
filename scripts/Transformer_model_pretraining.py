# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 14:02:43 2026

@author: shabnam
"""

import time

# Module import
import argparse
from pathlib import Path
import numpy as np
import pandas as pd 
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Dict
import torch.nn.functional as F
import json
import umap
import random
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
from utils.seed import set_seed
from utils.load_data import load_data



# Embedding strategy
class BinnedCMDDataset(Dataset):
    def __init__(self, bin_mat: np.ndarray, n_species: int=2316, max_len: int = 512, cls_bin: int=0):
        self.bin_mat = bin_mat
        self.max_len = max_len
        self.cls_bin = cls_bin
        self.n_species = bin_mat.shape[1]
        self.cls_id = n_species + 1
        self.pad_id = 0
        self.cls_id = self.n_species + 1
    # returns number of samples
    def __len__(self):
        return self.bin_mat.shape[0]
    # loads and returns a sample from the dataset at the given index. Defining what happens to 1 sample.
    def __getitem__(self, idx):
        row = self.bin_mat[idx] # This vector has the length of species 
        # keep non-zero species only (detected taxa)
        non_zero = np.flatnonzero(row > 0)
        # Unlikely but what if there are no detected species in a sample, then return empty arrays or else extract bins
        if non_zero.size == 0:
            species = np.array([], dtype=np.int64)
            bins = np.array([], dtype=np.int64)
        else:
            bins_non_zero = row[non_zero].astype(np.int64)
            
            # Arbitrary order?
            species = non_zero + 1
            species = species.astype(np.int64)
            bins = bins_non_zero
        # Truncate to consider attention cost
        keep = min(species.shape[0], self.max_len - 1)
        species = species[:keep]
        bins = bins[:keep]# Prepend CLS token so it comes first in the sequence but you still give an id like other tokens
        # The last id is reserved for CLS 
        cls_species = np.array([self.n_species ], dtype=np.int64)
        cls_bins = np.array([self.cls_bin], dtype=np.int64)
        species = np.concatenate([cls_species, species], axis=0)
        bins = np.concatenate([cls_bins, bins], axis=0)

        current_len = species.shape[0]
        pad_len = self.max_len - current_len
        if pad_len > 0:
            species = np.concatenate([species, np.full((pad_len,), self.pad_id, dtype=np.int64)], axis=0)
            bins    = np.concatenate([bins,    np.zeros((pad_len,), dtype=np.int64)], axis=0)

        # Attention mask: no padding yet, tells which ones are real (1) and which ones are padding (0). At this point, all tokens are valid.
        # Return an array of ones with the same shape and type as a given array - here species 
        attention_mask = (species != self.pad_id).astype(np.int64)

        # Return in tensors, so dataloader can be applied
        return {"species_ids": torch.from_numpy(species).long(),"bin_ids": torch.from_numpy(bins).long(), "attention_mask": torch.from_numpy(attention_mask).bool()}
    

# Sample sequence creation - Embedding layer

'''I have dictionary containing species_id, bin_id and attention mask. Now I want to use the embedding strategy.
According to BiomeGPT model,
1. Species_id are converted to a species embedding through pytorch's embedding
2. Abundance_ids go through a lightweight MLP and abundance embedding is created
3. Layernorm is applied to both these embeddings so they are at the same scale
4. Additively combined so now we have a (species, abundance) embedding
5. Same thing is also done to the cls token which was prepended at the beginning

This way we create the sample sequence'''

class InputEmbeddings(nn.Module):
    # Construction: Define parameters. Only architecture
    def __init__(self, n_species: int = 2316, n_bins:int = 40, d_model: int = 128 ,mlp_hidden: int = 200, pad_id: int = 0):
        super().__init__()
        self.n_species = n_species
        self.n_bins = n_bins
        self.d_model = d_model
        self.mlp_hidden = mlp_hidden
        self.pad_id = pad_id
        self.cls_token_id = n_species + 1
        # Species embedding layer: table, +1 for <cls>
        self.species_emb = nn.Embedding(n_species + 2, d_model, padding_idx=0)
        # Abundance embedding: shallow MLP with ReLU
        self.abund_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),)
        # Independent LayerNorms
        self.ln_species = nn.LayerNorm(d_model)
        self.ln_abund = nn.LayerNorm(d_model)

    # Computation : Data flow, so input tensor to output tensor
    def forward(self,species_ids: torch.LongTensor,abundance_bins: torch.LongTensor,pad_id: int = 0,) -> Tuple [torch.Tensor, torch.Tensor]:
        attention_mask = species_ids.ne(0)
        batch, length = species_ids.shape
        device = species_ids.device
        valid_mask = species_ids.ne(self.pad_id)
        e_species = self.species_emb(species_ids)
        # Abundance embeddings (MLP)
        a = abundance_bins.float().unsqueeze(-1)       
        e_abund = self.abund_mlp(a)
        # Independent LayerNorm
        e_species = self.ln_species(e_species)
        e_abund = self.ln_abund(e_abund)
        # Final embeddings
        t = e_species + e_abund
        t = t * valid_mask.unsqueeze(-1)
        return t, attention_mask

# Output here would be a tensor of shape (B,L,d_model)

# Transformer Model
'''
Non-autoregressive masked attention strategy is applied. According to BiomeGPT,
1. Implemented using TransformerEncoderLayer module from Pytorch
2. 8 stacked transformer layers each containing 8 attention heads
3. Feedforward sublayer with hidden dimension of 512

Architecture:
Input - Sample sequence, Tensor with shape (B,L,d_model)
Multihead attention layer
Feedforward network

TransformerEncoderLayer from pytorch

Each TransformerEncoderLayer contains:

1. Multi-Head Self-Attention
2. Add & Norm
3. Feedforward Network (2-layer MLP)
4. Add & Norm
'''
class TransformerModel(nn.Module):
    def __init__(self, n_species, n_bins,mask_ratio, dropout, d_model, nhead, nlayers, ff_dim, head_hidden, mlp_hidden,device = None):
        super().__init__()
        self.pad_id = 0
        self.device = device
        self.cls_id = n_species + 1
        self.n_bins = n_bins
        self.mask_ratio = mask_ratio
        self.nhead = nhead
        self.embed = InputEmbeddings(n_species=n_species, d_model=d_model, mlp_hidden=mlp_hidden)
        # TransformerEncoderLayer from PyTorch
        encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= ff_dim, dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers= nlayers)
        # After the transformer stack, each token's final layer embedding passed through 3-layer MLP head, outputs one value per abundance bin.
        # This gives the contextual embedding we need for the dimensions (512)
        out_dim = n_bins + 1 #Bins start from 0 to 100 - How do you decide the number of bins? 
        
        # 3layer MLP
        # Implementing vertical formatting to improve readability
        self.bin_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, out_dim),
            )
        
    # For each sample, choose 25% tokens to mask and this masking should be done to only True attention masks (without padding or cls), as you cannot mask <cls>
    def sample_mask_positions(self,species_ids: torch.LongTensor, bin_ids: torch.LongTensor,attention_mask: torch.Tensor,) -> torch.Tensor:
        batch, length = species_ids.shape
        mask_position = torch.zeros((batch, length), dtype=torch.bool, device=species_ids.device)
        # So masked_position is filled with False
        
        # Which positions are eligible to mask, not pad and cls tokens
        masking_eligible = (attention_mask &(species_ids != self.pad_id) & (species_ids != self.cls_id))
        # Masking applied to each sample
        for i in range(batch):
            true_indices = torch.where(masking_eligible[i])[0]
            
            # If total number of elements equals zero (not eligible), skip
            if true_indices.numel() == 0:
                continue
            # How many To be masked rounded off, atleast 1 is
            n = true_indices.numel()
            to_mask = int(math.ceil(self.mask_ratio * n))
            to_mask = max(1, to_mask)

            # leave at least 1 eligible unmasked
            to_mask = min(to_mask, n - 1)

            if to_mask <= 0:
                continue
            
            # Among these which ones to mask - random permutation to shuffle and select
            pick = true_indices[torch.randperm(true_indices.numel(), device=true_indices.device)[:to_mask]]
            mask_position[i, pick] = True # Mark these as masked, initially this was all false
        return mask_position
    
    # Attention principle: Attention principle: Unmasked can see unmasked. Masked can look at unmasked. Unmasked cannot look at masked. Masked cannot look at masked. 
    # Doing this returns attention_mask with shape (b*nhead,l,l) with 0 for allowed and -inf for blocked
    def attention_masking(self, mask_position: torch.Tensor, attention_mask: torch.Tensor,) -> torch.Tensor:
        
        batch, length = mask_position.shape
        device = mask_position.device
        
        # Mask position is matrix with masks with (batch,length) dimension
        # Masked cannot look at masked
        # In each sample, attention computes similarity score for each of L query tokens and each of L key tokens. So this gives an (L,L) matrix
        # For each sample and each query token, is the query token masked?
        q_masked = mask_position.unsqueeze(2) #(B,L,1)
        # For each sample and each key token (to which the query is compared), is the key token masked?
        k_masked = mask_position.unsqueeze(1)   #(B,1,L)
        # Unmasked to masked means q_unmasked and k_masked
        unmask_to_mask = (~q_masked) & k_masked
        
        # Identity matrix
        id_matrix = torch.eye(length, dtype=torch.bool, device=device).unsqueeze(0) #(1,L,L) Diagonal is true, off diagonal is false
        masked_to_masked = (q_masked & k_masked) & (~id_matrix) # True when token i is masked and j is masked and off diagonal is false
        # Block
        block = unmask_to_mask | masked_to_masked

        atten_mask = torch.zeros((batch, length, length), device=device)
        atten_mask[block] = float("-inf") # Whatever is blocked gets -inf

        # Expand for MultiheadAttention's (B*nhead, L, L)
        atten_mask = atten_mask.repeat_interleave(self.nhead, dim=0)
        return atten_mask
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Batch dictionary, keys are species_ids, bins, attention_mask (bool)
        
        species_ids = batch["species_ids"].to(self.device)
        bin_ids = batch["bin_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device).bool()

        
        # Choose masked positions (% of non-zero species)
        mask_position = self.sample_mask_positions(species_ids, bin_ids, attention_mask)

        # Fill masked bin values to 0, these are masked so we don't see the bin values
        bin_in = bin_ids.clone()
        bin_in[mask_position] = 0

        # Masked embeddings
        x = self.embed(species_ids, bin_in, attention_mask)[0]
        # Making sure only detected are present
        detected = attention_mask & (bin_ids > 0)
        detected[:, 0] = True

        # Padding mask for transformer:True where padding. Attention mask contains True for valid 
        src_key_padding_mask = ~detected 

        # Computing the multihead attention mask based on the rules
        atten_mask = self.attention_masking(mask_position, detected).bool()
        
        # Transformer forward pass - layer by layer computation, h is the output embeddings. So each token has a contextual embedding in dimension d_model
        h = self.transformer(
            x,
            mask=atten_mask,
            src_key_padding_mask=src_key_padding_mask)  
        
        # Extract sample-level representations
        cls_emb = h[:, 0, :] 
        
        # Output MLP Predicts integer bin values per token. MLP maps d_model to n_bins+1
        predict_bin = self.bin_head(h)  

        # Loss is computed only on masked tokens (not on cls and padded tokens)
        loss_masked = mask_position & (bin_ids > 0)
        # What if there is no masked token which has bin_d > 0
        if loss_masked.sum() == 0:
            loss = predict_bin.sum() * 0.0
        else:
            # MSE, each token's true bin is one-hot encoded so that predicted and target can be compared
            target = F.one_hot(bin_ids.clamp(0, self.n_bins).long(),num_classes=self.n_bins + 1).float()  # (B, L, C)
            mse = (predict_bin - target) ** 2  # (B, L, C)
            loss = mse[loss_masked].mean() #Final loss is a scalar

        
        return h, cls_emb, loss

 
def get_scheduler(
    optimizer:     torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs:  int,
    eta_min:       float = 1e-6,) -> SequentialLR:
    
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# Train only 1 epoch
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    total_grad_norm = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        h, cls_emb, loss = model(batch)
        loss.backward()
        
        # Check for exploding gradients - if needed then can be clipped
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_grad_norm += grad_norm.item()

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(1, n_batches)
    avg_grad_norm = total_grad_norm / max(1, n_batches)

    return avg_loss, avg_grad_norm


def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        h, cls_emb, loss = model(batch)
        
        total_loss += float(loss.detach())
        n_batches += 1

    return total_loss / max(1, n_batches)


# Pretraining function block and saving best model
# Model checkpoint and early stopping
def pretrain(model, train_loader, val_loader, optimizer, device, epochs=30, patience=7, model_path=None, log_path=None, warmup_epochs=2):

    scheduler = get_scheduler(optimizer, warmup_epochs, epochs)

    best_val = float("inf")
    no_improvement_epoch = 0
    logging = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        tr, grad_norm = train_one_epoch(model, train_loader, optimizer, device)
        va = validate_one_epoch(model, val_loader, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logging["epoch"].append(epoch)
        logging["train_loss"].append(tr)
        logging["val_loss"].append(va)
        logging["lr"].append(current_lr)

        print(f"Epoch {epoch:02d} | train={tr:.6f} | val={va:.6f} | "
              f"lr={current_lr:.2e} | grad_norm={grad_norm:.4f}")

        if va < best_val:
            best_val = va
            no_improvement_epoch = 0
            torch.save(model.state_dict(), model_path)
        else:
            no_improvement_epoch += 1
            print(f"No improvement ({no_improvement_epoch}/{patience})")
        if no_improvement_epoch >= patience:
            print("Early stopping triggered")
            break
        if log_path is not None:
            with open(log_path, "w") as f:
                json.dump(logging, f, indent=4)

    return logging["train_loss"], logging["val_loss"]
    


# Plot loss curves for every epoch
def plot_loss_curves(train_losses, val_losses, title="Pretraining Loss Curves", save_path = None):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    
    plt.close()

# Extract cls embeddings
@torch.no_grad()
def extract_cls_embeddings(model, loader, device, cls_path=None):
    model.eval()
    cls_all = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        h, cls_emb, loss = model(batch)

        cls_all.append(cls_emb.detach().cpu().numpy())

    Z = np.concatenate(cls_all, axis=0)

    if cls_path is not None:
        np.save(cls_path, Z)

    return Z

# UMAP to visualize representations (sample-level)
def run_umap_full(Z: np.ndarray, y_labels, label_col: str = "body_site", title: str = "UMAP of CLS embeddings", save_umap_path: str | None = None, random_state: int = 42,):

    y_body = y_labels[label_col].astype(str).to_numpy()

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    Z_umap = reducer.fit_transform(Z) 

    # Plot
    plt.figure(figsize=(6, 5))
    labels = np.unique(y_body)

    for lab in labels:
        idx = (y_body == lab)
        plt.scatter(Z_umap[idx, 0], Z_umap[idx, 1], s=8, alpha=0.8, label=lab)

    plt.title(f"{title} (colored by {label_col})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    # Save plot
    if save_umap_path is not None:
        p = Path(save_umap_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)

    plt.close()





# Main function

def main():
    
    set_seed(42)
    begin = time.time()
    
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "CMD_data"
    output_dir = project_root / "results"
    
    parser = argparse.ArgumentParser(description="Transformer model pretraining")
    parser.add_argument('--train_csv', type=str, default=str(data_dir/"X_train.csv"), help='Training file')
    parser.add_argument('--labels_csv', type=str, default=str(data_dir/"y_train.csv"), help='Labels file')
    parser.add_argument('--output_dir', type=str, default=str(output_dir/"Phase1_PT_gut"), help='Directory to save model outputs')
    parser.add_argument('--bins', type=int, default=40, help='Number of bins')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    
    parser.add_argument('--lrate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wdecay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--maskratio', type=float, default=0.25, help='Masking ratio')
    parser.add_argument('--nheads', type=int, default=4, help='Number of heads inside multihead attention')
    parser.add_argument('--nlayers', type=int, default=4, help='Number of layers')
    parser.add_argument('--hiddendim', type=int, default=256, help='Hidden dimension of 3layer MLP')
    parser.add_argument('--dmodel', type=int, default=256, help='Model dimension in Multihead attention')
    parser.add_argument('--ffdim', type=int, default=512, help='Hidden dimension of feedforward layer')
    parser.add_argument('--mlphidden', type=int, default=200, help='Lightweight MLP for abundance embedding')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum length of sequence')
    
    # parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()  # Parse command-line arguments
    train_path = Path(args.train_csv)
    y_path = Path(args.labels_csv)
    # labels_path = Path(args.labels_csv)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = f"dmodel{args.dmodel}_epoch{args.epochs}_bins{args.bins}_dropout_{args.dropout}_layers{args.nlayers}_heads{args.nheads}_lr{args.lrate}_mr{args.maskratio}_max{args.max_len}"
    result_path = Path(args.output_dir) / run_name
    result_path.mkdir(parents=True, exist_ok=True)

    
    save_path=result_path /"loss_curves.png"
    model_path=result_path /"best_model.pt"
    log_path=result_path /"pretrain_logs.json"
    cls_path=result_path /"cls_embeddings.npy"
    save_umap_path= result_path /"umap.png"


    X_train,y_labels = load_data(X_path,y_path)
    X_train_np=X_train.to_numpy()
    Xtrain_percent = fraction_convert(X_train_np)
    bin_mat = abundance_binning(Xtrain_percent, bins=args.bins)
    print(bin_mat.shape, bin_mat.dtype)
    
    dataset = BinnedCMDDataset(bin_mat)

    train_ratio = 0.9
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    g = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
    entire_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    
    model = TransformerModel(
        n_species = 2316,
        n_bins = args.bins,
        dropout = args.dropout,
        d_model = args.dmodel,
        nhead  = args.nheads,
        nlayers = args.nlayers,
        ff_dim = args.ffdim,
        mask_ratio = args.maskratio,
        head_hidden = args.hiddendim,
        mlp_hidden = args.mlphidden,
        device=device
        ).to(device)
    
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=0.01)
    
 


    train_losses, val_losses = pretrain(model, train_loader, val_loader, optimizer, device, epochs=args.epochs, patience=7, warmup_epochs = 2, model_path=model_path,log_path=log_path)
    
    plot_loss_curves(train_losses, val_losses, save_path= save_path)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    Z = extract_cls_embeddings(model, entire_loader, device,cls_path=cls_path)
    
    Z_umap = run_umap_full(Z,y_labels,label_col="disease",save_umap_path=save_umap_path)
    
    end = time.time()
    Time = end - begin
    print(f"Time: {Time}")

# Run main if script is executed directly
if __name__ == "__main__":
    main()
    
