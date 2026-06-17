# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:20:11 2026

@author: shab3
"""


from collections import defaultdict
import time
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from utils.seed import set_seed
from utils.load_data import load_data_cohort
from utils.class_resampling import oversampling_gaussian_noise
from utils.data_transformation import abundance_binning

# Define binned dataset - Tokenization
class BinnedCMDDataset(Dataset):
    def __init__(self, bin_mat, labels=None, n_species=2316, max_len=200, cls_bin=0):
        self.bin_mat   = bin_mat
        self.labels    = labels
        self.max_len   = max_len
        self.cls_bin   = cls_bin
        self.n_species = n_species
        self.pad_id    = 0

    def __len__(self):
        return self.bin_mat.shape[0]

    def __getitem__(self, idx):
        row      = self.bin_mat[idx]
        non_zero = np.flatnonzero(row > 0)

        if non_zero.size == 0:
            species = np.array([], dtype=np.int64)
            bins    = np.array([], dtype=np.int64)
        else:
            species = (non_zero + 1).astype(np.int64)
            bins    = row[non_zero].astype(np.int64)

        keep    = min(species.shape[0], self.max_len - 1)
        species = species[:keep]
        bins    = bins[:keep]

        species = np.concatenate([np.array([self.n_species], dtype=np.int64), species])
        bins    = np.concatenate([np.array([self.cls_bin],   dtype=np.int64), bins])

        pad_len = self.max_len - species.shape[0]
        if pad_len > 0:
            species = np.concatenate([species, np.full(pad_len, self.pad_id, dtype=np.int64)])
            bins    = np.concatenate([bins,    np.zeros(pad_len, dtype=np.int64)])

        attention_mask = (species != self.pad_id).astype(np.int64)

        out = {
            "species_ids":    torch.from_numpy(species).long(),
            "bin_ids":        torch.from_numpy(bins).long(),
            "attention_mask": torch.from_numpy(attention_mask).bool(),
        }
        if self.labels is not None:
            out["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return out

# Input embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, n_species=2316, n_bins=50, d_model=128, mlp_hidden=200, pad_id=0):
        super().__init__()
        self.pad_id      = pad_id
        self.species_emb = nn.Embedding(n_species + 2, d_model, padding_idx=0)
        self.abund_mlp   = nn.Sequential(
            nn.Linear(1, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, d_model)
        )
        self.ln_species = nn.LayerNorm(d_model)
        self.ln_abund   = nn.LayerNorm(d_model)

    def forward(self, species_ids, abundance_bins):
        valid_mask = species_ids.ne(self.pad_id)
        e_species  = self.ln_species(self.species_emb(species_ids))
        e_abund    = self.ln_abund(self.abund_mlp(abundance_bins.float().unsqueeze(-1)))
        return (e_species + e_abund) * valid_mask.unsqueeze(-1)

# Transformer backbone
class TransformerModel(nn.Module):
    def __init__(self, n_species, n_bins, dropout, d_model, nhead,
                 nlayers, ff_dim, head_hidden, mlp_hidden, device=None):
        super().__init__()
        self.pad_id  = 0
        self.device  = device
        self.cls_id  = n_species + 1
        self.n_bins  = n_bins
        self.nhead   = nhead

        self.embed = InputEmbeddings(n_species=n_species, d_model=d_model,
                                     mlp_hidden=mlp_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.bin_head = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, n_bins + 1),
        )

    def forward(self, batch):
        species_ids      = batch["species_ids"].to(self.device)
        bin_ids          = batch["bin_ids"].to(self.device)
        attention_mask   = species_ids.ne(self.pad_id)
        key_padding_mask = ~attention_mask

        if species_ids.dim() == 1:
            species_ids      = species_ids.unsqueeze(0)
            bin_ids          = bin_ids.unsqueeze(0)
            key_padding_mask = key_padding_mask.unsqueeze(0)

        x       = self.embed(species_ids, bin_ids)
        h       = self.transformer(x, src_key_padding_mask=key_padding_mask)
        cls_emb = h[:, 0, :]
        return h, cls_emb

# Classification head - MLP

class ClassificationHead(nn.Module):
    def __init__(self, d_model, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, cls_emb):
        return self.head(cls_emb)

# Finetuner integrating pretrained backbone and classifier head

class FineTuner(nn.Module):
    def __init__(self, backbone, d_model, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.backbone   = backbone
        self.classifier = ClassificationHead(d_model, hidden_dim, num_classes, dropout)

    def forward(self, batch):
        _, cls_emb = self.backbone(batch)
        return self.classifier(cls_emb)

# Freezing to avoid overfitting
def apply_freeze_strategy(model: FineTuner, strategy: str, n_layers: int,
                           n_unfreeze_layers: int = 1) -> None:
    """
    "full"
        Freeze the entire backbone (embed + all transformer layers).
        Only the ClassificationHead trains. Identical to the original script.

    "partial"
        Freeze the embedding layer and the first (n_layers - n_unfreeze_layers)
        transformer layers. The last n_unfreeze_layers transformer layers and
        the ClassificationHead train. This lets high-level representations adapt
        while preserving general low-level microbiome features.

    "none"
        Leave the entire backbone unfrozen. Everything trains.
        Highest risk of overfitting on small cohorts.

    """
    if strategy not in ("full", "partial", "none"):
        raise ValueError(f"freeze_strategy must be 'full', 'partial', or 'none'. Got: {strategy}")

    if strategy == "none":
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("  Freeze strategy: NONE — entire backbone trainable.")
        return

    if strategy == "full":
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("  Freeze strategy: FULL — backbone frozen, head only.")
        return

    
   
    for param in model.backbone.parameters():
        param.requires_grad = False

    # unfreeze the last n_unfreeze_layers transformer layers
    n_freeze = max(0, n_layers - n_unfreeze_layers)
    for layer_idx in range(n_freeze, n_layers):
        for param in model.backbone.transformer.layers[layer_idx].parameters():
            param.requires_grad = True

    frozen_layers   = list(range(n_freeze))
    trainable_layers = list(range(n_freeze, n_layers))
    

    
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")


# Learning rate scheduler
def get_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs,
                               eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = batch.pop("label").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("label")
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    probs      = torch.softmax(all_logits, dim=-1).numpy()
    preds      = probs.argmax(axis=-1)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, preds)),
        "f1":       float(f1_score(all_labels, preds, average="macro", zero_division=0)),
        "mcc":      float(matthews_corrcoef(all_labels, preds)),
    }

    if len(np.unique(all_labels)) < 2:
        metrics["auc"] = float("nan")
    elif num_classes == 2:
        metrics["auc"] = float(roc_auc_score(all_labels, probs[:, 1]))
    else:
        metrics["auc"] = float(
            roc_auc_score(all_labels, probs, multi_class="ovr", average="macro")
        )

    return metrics, probs, all_labels

# Build model with freezing
def _build_model(
    pretrained_checkpoint: str,
    backbone_kwargs:       Dict,
    d_model:               int,
    hidden_dim:            int,
    num_classes:           int,
    dropout:               float,
    device:                torch.device,
    freeze_strategy:       str  = "full",
    n_unfreeze_layers:     int  = 1,
) -> FineTuner:
    backbone = TransformerModel(**backbone_kwargs)
    backbone.load_state_dict(torch.load(pretrained_checkpoint, map_location=device))

    model = FineTuner(
        backbone    = backbone,
        d_model     = d_model,
        hidden_dim  = hidden_dim,
        num_classes = num_classes,
        dropout     = dropout,
    ).to(device)

    apply_freeze_strategy(
        model,
        strategy          = freeze_strategy,
        n_layers          = backbone_kwargs["nlayers"],
        n_unfreeze_layers = n_unfreeze_layers,
    )
    return model


def _train_and_eval(
    *,
    pretrained_checkpoint: str,
    backbone_kwargs:       Dict,
    train_bin:             np.ndarray,
    train_labels:          np.ndarray,
    val_bin:               np.ndarray,
    val_labels:            np.ndarray,
    d_model:               int,
    hidden_dim:            int,
    num_classes:           int,
    dropout:               float,
    lr:                    float,
    weight_decay:          float,
    epochs:                int,
    patience:              int,
    warmup_epochs:         int,
    batch_size:            int,
    sigma_aug:             float,
    max_len:               int,
    n_species:             int,
    device:                torch.device,
    freeze_strategy:       str = "full",
    n_unfreeze_layers:     int = 1,
    fold_seed:             int = 0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

    X_aug, y_aug = oversampling_gaussian_noise(
        train_bin.astype(np.float32), train_labels,
        sigma=sigma_aug, random_state=fold_seed
    )
    rng  = np.random.default_rng(fold_seed)
    perm = rng.permutation(len(y_aug))
    X_aug, y_aug = X_aug[perm], y_aug[perm]

    train_ds = BinnedCMDDataset(X_aug,   y_aug,      n_species=n_species, max_len=max_len)
    val_ds   = BinnedCMDDataset(val_bin, val_labels, n_species=n_species, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = _build_model(
        pretrained_checkpoint, backbone_kwargs,
        d_model, hidden_dim, num_classes, dropout, device,
        freeze_strategy   = freeze_strategy,
        n_unfreeze_layers = n_unfreeze_layers,
    )

    class_counts  = np.bincount(y_aug, minlength=num_classes).astype(float)
    class_weights = torch.tensor(
        1.0 / np.maximum(class_counts, 1), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = get_scheduler(optimizer, warmup_epochs, epochs)

    best_auc   = -1.0
    no_improve = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, device, num_classes)
        scheduler.step()

        auc_val = val_metrics["auc"]
        if not np.isnan(auc_val) and auc_val > best_auc:
            best_auc   = auc_val
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return evaluate(model, val_loader, device, num_classes)


# Inner grid search
def inner_grid_search(
    pretrained_checkpoint: str,
    backbone_kwargs:       Dict,
    bin_mat:               np.ndarray,
    y:                     np.ndarray,
    groups:                np.ndarray,
    param_grid:            Dict[str, List],
    d_model:               int,
    num_classes:           int,
    device:                torch.device,
    weight_decay:          float,
    epochs:                int,
    patience:              int,
    warmup_epochs:         int,
    batch_size:            int,
    sigma_aug:             float,
    n_species:             int,
    max_len:               int,
    freeze_strategy:       str = "full",
    n_unfreeze_layers:     int = 1,
    outer_fold_seed:       int = 0,
    verbose:               bool = False,
) -> Dict:
    keys    = list(param_grid.keys())
    combos  = list(product(*[param_grid[k] for k in keys]))
    configs = [dict(zip(keys, c)) for c in combos]

    inner_studies = np.array([
        s for s in np.unique(groups)
        if len(np.unique(y[groups == s])) >= 2
    ])

    if len(inner_studies) < 2:
        print("  Fewer than 2 inner studies — returning default config")
        return configs[0]

    print(f"\n  Inner grid search: {len(configs)} configs × {len(inner_studies)} inner folds")

    config_scores = []
    for cfg_idx, cfg in enumerate(configs):
        fold_aucs = []
        for inner_fold, held in enumerate(inner_studies):
            val_mask   = groups == held
            train_mask = ~val_mask

            val_bin      = bin_mat[val_mask]
            val_labels   = y[val_mask]
            if len(np.unique(val_labels)) < 2:
                continue

            try:
                metrics, _, _ = _train_and_eval(
                    pretrained_checkpoint = pretrained_checkpoint,
                    backbone_kwargs       = backbone_kwargs,
                    train_bin             = bin_mat[train_mask],
                    train_labels          = y[train_mask],
                    val_bin               = val_bin,
                    val_labels            = val_labels,
                    d_model               = d_model,
                    hidden_dim            = cfg["hidden_dim"],
                    num_classes           = num_classes,
                    dropout               = cfg["dropout"],
                    lr                    = cfg["lr"],
                    weight_decay          = weight_decay,
                    epochs                = epochs,
                    patience              = patience,
                    warmup_epochs         = warmup_epochs,
                    batch_size            = batch_size,
                    sigma_aug             = sigma_aug,
                    max_len               = max_len,
                    n_species             = n_species,
                    device                = device,
                    freeze_strategy       = freeze_strategy,
                    n_unfreeze_layers     = n_unfreeze_layers,
                    fold_seed             = outer_fold_seed * 1000 + inner_fold,
                )
                fold_auc = metrics["auc"]
                if not np.isnan(fold_auc):
                    fold_aucs.append(fold_auc)
                if verbose:
                    print(f"    cfg {cfg_idx+1}/{len(configs)} inner fold "
                          f"{inner_fold+1}/{len(inner_studies)} AUC={fold_auc:.4f} | {cfg}")
            except Exception as e:
                print(f"    cfg {cfg_idx+1} inner fold {inner_fold+1} FAILED: {e}")

        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        config_scores.append({"config": cfg, "mean_inner_auc": mean_auc,
                               "n_valid_folds": len(fold_aucs)})
        print(f"  Config {cfg_idx+1:02d}/{len(configs)} mean_inner_AUC={mean_auc:.4f}  {cfg}")

    valid = [s for s in config_scores if not np.isnan(s["mean_inner_auc"])]
    if not valid:
        print("  WARNING: all inner configs returned NaN — using default")
        return configs[0]

    best = max(valid, key=lambda s: s["mean_inner_auc"])
    print(f"\n  Best config: {best['config']}  (mean inner AUC={best['mean_inner_auc']:.4f})")
    return best["config"]


# Nested LOSO CV
def nested_loso_cv(
    pretrained_checkpoint: str,
    backbone_kwargs:       Dict,
    bin_mat:               np.ndarray,
    y:                     np.ndarray,
    groups:                np.ndarray,
    param_grid:            Dict[str, List],
    d_model:               int   = 256,
    num_classes:           int   = 2,
    device:                torch.device = torch.device("cpu"),
    epochs:                int   = 20,
    patience:              int   = 5,
    weight_decay:          float = 1e-2,
    warmup_epochs:         int   = 2,
    batch_size:            int   = 64,
    sigma_aug:             float = 0.1,
    n_species:             int   = 2316,
    max_len:               int   = 512,
    freeze_strategy:       str   = "full",
    n_unfreeze_layers:     int   = 1,
    log_path               = None,
    fold_plot_dir          = None,
    inner_verbose:         bool  = False,
) -> Dict:

    unique_studies = np.array([
        s for s in np.unique(groups)
        if len(np.unique(y[groups == s])) >= 2
    ])
    n_outer = len(unique_studies)

    fold_results       = []
    oof_probs_list     = []
    oof_preds_list     = []
    oof_labels_list    = []
    fold_val_indices   = []
    all_fold_histories = []
    best_state_global  = None

    print(f"\nNested LOSO CV: {n_outer} outer folds")
    print(f"Grid            : {param_grid}")
    print(f"Freeze strategy : {freeze_strategy}"
          + (f"  (unfreeze last {n_unfreeze_layers} layer(s))"
             if freeze_strategy == "partial" else ""))

    for fold, held_out_study in enumerate(unique_studies):
        val_mask   = groups == held_out_study
        train_mask = ~val_mask
        val_idx    = np.where(val_mask)[0]
        train_idx  = np.where(train_mask)[0]

    
 
        fold_val_indices.append(val_idx)

        best_cfg = inner_grid_search(
            pretrained_checkpoint = pretrained_checkpoint,
            backbone_kwargs       = backbone_kwargs,
            bin_mat               = bin_mat[train_idx],
            y                     = y[train_idx],
            groups                = groups[train_idx],
            param_grid            = param_grid,
            d_model               = d_model,
            num_classes           = num_classes,
            device                = device,
            weight_decay          = weight_decay,
            epochs                = epochs,
            patience              = patience,
            warmup_epochs         = warmup_epochs,
            batch_size            = batch_size,
            sigma_aug             = sigma_aug,
            n_species             = n_species,
            max_len               = max_len,
            freeze_strategy       = freeze_strategy,
            n_unfreeze_layers     = n_unfreeze_layers,
            outer_fold_seed       = fold,
            verbose               = inner_verbose,
        )

        hidden_dim = best_cfg["hidden_dim"]
        dropout    = best_cfg["dropout"]
        lr         = best_cfg["lr"]

        print(f"\n  Retraining outer fold {fold+1} with best config: {best_cfg}")

        X_train_aug, y_train_aug = oversampling_gaussian_noise(
            bin_mat[train_idx].astype(np.float32), y[train_idx],
            sigma=sigma_aug, random_state=fold
        )
        rng  = np.random.default_rng(fold)
        perm = rng.permutation(len(y_train_aug))
        X_train_aug = X_train_aug[perm]
        y_train_aug = y_train_aug[perm]

        train_ds = BinnedCMDDataset(X_train_aug,      y_train_aug, n_species=n_species, max_len=max_len)
        val_ds   = BinnedCMDDataset(bin_mat[val_idx], y[val_idx],  n_species=n_species, max_len=max_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        model = _build_model(
            pretrained_checkpoint, backbone_kwargs,
            d_model, hidden_dim, num_classes, dropout, device,
            freeze_strategy   = freeze_strategy,
            n_unfreeze_layers = n_unfreeze_layers,
        )

        class_counts  = np.bincount(y_train_aug, minlength=num_classes).astype(float)
        class_weights = torch.tensor(
            1.0 / np.maximum(class_counts, 1), dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        scheduler = get_scheduler(optimizer, warmup_epochs, epochs)

        best_auc   = -1.0
        no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        fold_history = []

        for epoch in range(1, epochs + 1):
            train_loss        = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics, _, _ = evaluate(model, val_loader, device, num_classes)
            scheduler.step()
            current_lr        = scheduler.get_last_lr()[0]

            fold_history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            print(f"  Epoch {epoch:02d} | loss={train_loss:.5f} | "
                  f"auc={val_metrics['auc']:.4f} | f1={val_metrics['f1']:.4f} | "
                  f"mcc={val_metrics['mcc']:.4f} | acc={val_metrics['accuracy']:.4f} | "
                  f"lr={current_lr:.2e}")

            auc_val = val_metrics["auc"]
            if not np.isnan(auc_val) and auc_val > best_auc:
                best_auc   = auc_val
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        final_metrics, fold_probs, fold_labels = evaluate(model, val_loader, device, num_classes)

        fold_results.append({
            "fold":        fold + 1,
            "study":       held_out_study,
            "best_config": best_cfg,
            **final_metrics,
        })

        oof_probs_list.append(fold_probs)
        oof_preds_list.append(fold_probs.argmax(axis=-1))
        oof_labels_list.append(fold_labels)
        all_fold_histories.append(fold_history)

        print(f"\n  Outer fold {fold+1} final | Study='{held_out_study}' | "
              f"AUC={final_metrics['auc']:.4f} | F1={final_metrics['f1']:.4f} | "
              f"MCC={final_metrics['mcc']:.4f} | Acc={final_metrics['accuracy']:.4f}")
        print(classification_report(fold_labels, fold_probs.argmax(axis=-1), zero_division=0))

        best_state_global = best_state
        if fold_plot_dir is not None:
            _plot_fold_curves(fold_history, fold + 1, fold_plot_dir)

    # ── Aggregate OOF ──────────────────────────────────────────────────────
    y_true_all = np.concatenate(oof_labels_list)
    y_pred_all = np.concatenate(oof_preds_list)
    y_prob_all = np.concatenate(oof_probs_list, axis=0)

    oof_auc = (roc_auc_score(y_true_all, y_prob_all[:, 1]) if num_classes == 2
               else roc_auc_score(y_true_all, y_prob_all, multi_class="ovr", average="macro"))
    oof_f1  = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    oof_mcc = matthews_corrcoef(y_true_all, y_pred_all)
    oof_acc = accuracy_score(y_true_all, y_pred_all)

    print(f"\n{'='*65}")
    print(f"Overall OOF Performance — strategy={freeze_strategy}")
    print(f"  AUC      : {oof_auc:.4f}")
    print(f"  F1       : {oof_f1:.4f}")
    print(f"  MCC      : {oof_mcc:.4f}")
    print(f"  Accuracy : {oof_acc:.4f}")
    print(classification_report(y_true_all, y_pred_all, zero_division=0))

    print("\nPer-study variance:")
    for metric in ["auc", "f1", "mcc", "accuracy"]:
        vals  = [r[metric] for r in fold_results]
        valid = [v for v in vals if not np.isnan(v)]
        n_nan = len(vals) - len(valid)
        note  = f" ({n_nan} NaN excluded)" if n_nan else ""
        print(f"  {metric.upper():10s}: {np.mean(valid):.4f} ± {np.std(valid, ddof=1):.4f}{note}")

    def _safe_mean(m):
        vals = [r[m] for r in fold_results if not np.isnan(r[m])]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    def _safe_std(m):
        vals = [r[m] for r in fold_results if not np.isnan(r[m])]
        return round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else float("nan")

    results = {
        "fold_results":                  fold_results,
        "fold_val_indices":              fold_val_indices,
        "oof_auc":                       round(float(oof_auc),  4),
        "oof_f1":                        round(float(oof_f1),   4),
        "oof_mcc":                       round(float(oof_mcc),  4),
        "oof_accuracy":                  round(float(oof_acc),  4),
        "overall_classification_report": classification_report(
            y_true_all, y_pred_all, zero_division=0, output_dict=True
        ),
        "overall_confusion_matrix":      confusion_matrix(y_true_all, y_pred_all),
        "y_true_all":                    y_true_all,
        "y_pred_all":                    y_pred_all,
        "y_prob_all":                    y_prob_all,
        "mean_auc":      _safe_mean("auc"),
        "std_auc":       _safe_std("auc"),
        "mean_f1":       _safe_mean("f1"),
        "std_f1":        _safe_std("f1"),
        "mean_mcc":      _safe_mean("mcc"),
        "std_mcc":       _safe_std("mcc"),
        "mean_accuracy": _safe_mean("accuracy"),
        "std_accuracy":  _safe_std("accuracy"),
        "fold_histories":  all_fold_histories,
        "best_state":      best_state_global,
        "freeze_strategy": freeze_strategy,
        "n_unfreeze_layers": n_unfreeze_layers,
    }

    if log_path is not None:
        json_safe = {
            k: v for k, v in results.items()
            if k not in ("y_true_all", "y_pred_all", "y_prob_all",
                         "overall_confusion_matrix", "fold_val_indices",
                         "best_state", "fold_histories")
        }
        with open(log_path, "w") as f:
            json.dump(json_safe, f, indent=4)
        print(f"  Results saved → {log_path}")

    return results



def _plot_fold_curves(history, fold, save_dir):
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_auc    = [h["auc"]        for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Fold {fold} — Train Loss")
    ax2.plot(epochs, val_auc, color="orange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
    ax2.set_title(f"Fold {fold} — Val AUC")
    plt.tight_layout()
    p = Path(save_dir); p.mkdir(parents=True, exist_ok=True)
    plt.savefig(p / f"fold_{fold}_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", cm_path=None):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm,      display_labels=class_names).plot(ax=ax1, colorbar=False, cmap="Blues")
    ConfusionMatrixDisplay(np.round(cm_norm, 2), display_labels=class_names).plot(ax=ax2, colorbar=False, cmap="Blues")
    ax1.set_title(f"{title} (Counts)")
    ax2.set_title(f"{title} (Normalised)")
    plt.tight_layout()
    if cm_path:
        Path(cm_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, fold_results, oof_probs, y,
                   fold_val_indices, title="ROC Curve", roc_path=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    for fold_idx, val_idx in enumerate(fold_val_indices):
        fold_auc = fold_results[fold_idx]["auc"]
        if np.isnan(fold_auc) or len(np.unique(y[val_idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[val_idx], oof_probs[val_idx, 1])
        ax.plot(fpr, tpr, color="steelblue", alpha=0.3, linewidth=1,
                label=f"Fold {fold_idx+1} (AUC={fold_auc:.3f})")
    fpr_oof, tpr_oof, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr_oof, tpr_oof, color="red", linewidth=2.5,
            label=f"Overall OOF (AUC={auc(fpr_oof, tpr_oof):.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    if roc_path:
        Path(roc_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()


def patch_transformer_for_attention(transformer_encoder):
    attn_weights_store = {}
    for i, layer in enumerate(transformer_encoder.layers):
        def make_patched_forward(layer_idx, layer_module):
            def patched_forward(src, src_mask=None, src_key_padding_mask=None, is_causal=False):
                src2, attn_w = layer_module.self_attn(
                    src, src, src,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                attn_weights_store[layer_idx] = attn_w.detach().cpu()
                src = src + layer_module.dropout1(src2)
                src = layer_module.norm1(src)
                src2 = layer_module.linear2(
                    layer_module.dropout(layer_module.activation(layer_module.linear1(src)))
                )
                src = src + layer_module.dropout2(src2)
                src = layer_module.norm2(src)
                return src
            return patched_forward
        layer.forward = make_patched_forward(i, layer)
    return attn_weights_store


def run_umap(Z, y_labels, label_col="disease", title="UMAP", umap_path=None, random_state=42):
    y_disease = y_labels[label_col].astype(str).to_numpy()
    reducer   = umap.UMAP(n_components=2, random_state=random_state)
    Z_umap    = reducer.fit_transform(Z)
    plt.figure(figsize=(6, 5))
    for lab in np.unique(y_disease):
        idx = y_disease == lab
        plt.scatter(Z_umap[idx, 0], Z_umap[idx, 1], s=8, alpha=0.8, label=lab)
    plt.title(title); plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    if umap_path:
        Path(umap_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(umap_path, dpi=300)
    plt.close()
    return Z_umap


# Main
def main():
    set_seed(42)
    begin = time.time()

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    data_dir     = project_root / "CMD_data" / "binary_healthy_X"
    output_dir   = project_root / "results"
    model_dir    = (output_dir /
                    "Phase2_PT_train/dmodel256_ff512_epoch10_bins40_dropout_0.3_layers2_heads2_lr1e-05_mr0.25_max512")
    CHECKPOINT   = model_dir / "best_gut_model.pt"

    parser = argparse.ArgumentParser(description="Transformer fine-tuning — Healthy vs T2D (partial freeze)")

    # Data
    parser.add_argument("--train_csv",             type=str,   default=str(data_dir / "X_T2D.csv"))
    parser.add_argument("--labels_csv",            type=str,   default=str(data_dir / "y_T2D.csv"))
    parser.add_argument("--groups_csv",            type=str,   default=str(data_dir / "cohort_T2D.csv"))
    parser.add_argument("--output_dir",            type=str,   default=str(output_dir / "Finetuned_T2D_nested" / "partial_freeze"))
    parser.add_argument("--pretrained_model_path", type=str,   default=str(CHECKPOINT))

    # Architecture
    parser.add_argument("--bins",             type=int,   default=40)
    parser.add_argument("--epochs",           type=int,   default=35)
    parser.add_argument("--wdecay",           type=float, default=0.01)
    parser.add_argument("--nheads",           type=int,   default=2)
    parser.add_argument("--nlayers",          type=int,   default=2)
    parser.add_argument("--dmodel",           type=int,   default=256)
    parser.add_argument("--ffdim",            type=int,   default=512)
    parser.add_argument("--mlphidden",        type=int,   default=200)
    parser.add_argument("--backbone_dropout", type=float, default=0.3)
    parser.add_argument("--max_len",          type=int,   default=512)

    # ── Freeze strategy ────────────────────────────────────────────────────
    parser.add_argument("--freeze_strategy",type=str,default="none",choices=["full", "partial", "none"],help=("Freezing strategy for the backbone.\n"
            "  full: freeze entire backbone, train head only (original behaviour)\n"
            "  partial: freeze embed + first N-k layers, train last k layers + head\n"
            "  none: train entire model end-to-end"),)
    parser.add_argument("--n_unfreeze_layers",
        type=int,
        default=1,
        help="Number of transformer layers to unfreeze from the top (only used with --freeze_strategy partial).",)

    # Grid search
    parser.add_argument("--grid_hidden_dims", type=str, default="64,32,128")
    parser.add_argument("--grid_dropouts",    type=str, default="0.2,0.3,0.5")
    parser.add_argument("--grid_lrs",         type=str, default="1e-5,1e-4,1e-3,5e-4")
    parser.add_argument("--inner_verbose",    action="store_true")

    args = parser.parse_args()

   
    strategy_tag = (f"freeze_{args.freeze_strategy}"
                    + (f"_top{args.n_unfreeze_layers}"
                       if args.freeze_strategy == "partial" else ""))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_grid = {
        "hidden_dim": [int(v)   for v in args.grid_hidden_dims.split(",")],
        "dropout":    [float(v) for v in args.grid_dropouts.split(",")],
        "lr":         [float(v) for v in args.grid_lrs.split(",")],
    }
    print(f"\nParam grid      : {param_grid}")
    print(f"Freeze strategy : {args.freeze_strategy}"
          + (f"  (unfreeze top {args.n_unfreeze_layers} layer(s))"
             if args.freeze_strategy == "partial" else ""))

    run_name = (
        f"dmodel{args.dmodel}_ffdim{args.ffdim}_epoch{args.epochs}_"
        f"bins{args.bins}_bdropout_{args.backbone_dropout}_"
        f"layers{args.nlayers}_heads{args.nheads}_max{args.max_len}_"
        f"{strategy_tag}_nested_loso"
    )
    result_path = output_dir / run_name
    result_path.mkdir(parents=True, exist_ok=True)

    X_train, y_labels, groups = load_data_cohort(Path(args.train_csv), Path(args.labels_csv), Path(args.groups_csv))
    X_train_np   = X_train.to_numpy()
    species_list = X_train.columns.tolist()
    label_map    = {"healthy": 0, "T2D": 1}
    y_np         = np.array([label_map[v] for v in y_labels["disease"].to_numpy()])
    groups_np    = groups["study_name"].to_numpy()
    n_species    = X_train_np.shape[1]

    print(f"\n  Samples  : {len(y_np)}")
    print(f"  Species  : {n_species}")
    print(f"  Cohorts  : {np.unique(groups_np).tolist()}")

    bin_mat = abundance_binning(X_train_np, bins=args.bins)

    backbone_kwargs = dict(
        n_species   = 2316,
        n_bins      = args.bins,
        dropout     = args.backbone_dropout,
        d_model     = args.dmodel,
        nhead       = args.nheads,
        nlayers     = args.nlayers,
        ff_dim      = args.ffdim,
        head_hidden = 256,
        mlp_hidden  = args.mlphidden,
        device      = device,
    )

   
    results = nested_loso_cv(
        pretrained_checkpoint = str(Path(args.pretrained_model_path)),
        backbone_kwargs       = backbone_kwargs,
        bin_mat               = bin_mat,
        y                     = y_np,
        groups                = groups_np,
        param_grid            = param_grid,
        d_model               = args.dmodel,
        num_classes           = 2,
        device                = device,
        epochs                = args.epochs,
        patience              = 5,
        weight_decay          = args.wdecay,
        warmup_epochs         = 2,
        batch_size            = 64,
        sigma_aug             = 0.15,
        n_species             = n_species,
        max_len               = args.max_len,
        freeze_strategy       = args.freeze_strategy,
        n_unfreeze_layers     = args.n_unfreeze_layers,
        log_path              = str(result_path / "cv_results.json"),
        fold_plot_dir         = str(result_path / "fold_curves"),
        inner_verbose         = args.inner_verbose,
    )
    # Save best model
    torch.save(results["best_state"], str(result_path / "best_finetuned_model.pt"))

    y_true_all  = results["y_true_all"]
    y_pred_all  = results["y_pred_all"]
    y_prob_all  = results["y_prob_all"]
    class_names = ["healthy", "T2D"]
    # Plots
    plot_confusion_matrix(y_true_all, y_pred_all, class_names,
                          title=f"Confusion Matrix ({strategy_tag})",
                          cm_path=str(result_path / "confusion_matrix.png"))

    plot_roc_curve(y_true_all, y_prob_all[:, 1],
                   results["fold_results"], y_prob_all, y_np,
                   results["fold_val_indices"],
                   title=f"ROC Curve — {strategy_tag} (nested LOSO)",
                   roc_path=str(result_path / "roc_plot.png"))

    # CLS embeddings
    full_ds     = BinnedCMDDataset(bin_mat, n_species=n_species, max_len=args.max_len)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False)

    best_hidden_dim = results["fold_results"][-1]["best_config"]["hidden_dim"]
    best_dropout    = results["fold_results"][-1]["best_config"]["dropout"]

    backbone = TransformerModel(**backbone_kwargs).to(device)
    finetune_model = FineTuner(backbone, args.dmodel, best_hidden_dim, 2, best_dropout).to(device)
    finetune_model.load_state_dict({k: v.to(device) for k, v in results["best_state"].items()})
    finetune_model.eval()

    attn_store             = patch_transformer_for_attention(finetune_model.backbone.transformer)
    cls_all                = []
    species_attention_sum  = defaultdict(float)
    species_presence_count = defaultdict(int)

    with torch.no_grad():
        for batch in full_loader:
            species_ids  = batch["species_ids"]
            batch_device = {k: v.to(device) for k, v in batch.items()}
            h, cls_emb   = finetune_model.backbone(batch_device)
            cls_all.append(cls_emb.detach().cpu().numpy())

            all_layer_attns    = torch.stack([attn_store[i] for i in range(args.nlayers)], dim=1)
            attn_from_cls_mean = all_layer_attns[:, :, :, 0, :].mean(dim=[1, 2])

            B = species_ids.shape[0]
            for b in range(B):
                for seq_pos in range(1, species_ids.shape[1]):
                    token_id = species_ids[b, seq_pos].item()
                    if token_id == 0:
                        break
                    species_name = species_list[token_id - 1]
                    species_attention_sum[species_name]  += attn_from_cls_mean[b, seq_pos].item()
                    species_presence_count[species_name] += 1

    species_mean_attention = {
        sp: species_attention_sum[sp] / species_presence_count[sp]
        for sp in species_attention_sum if species_presence_count[sp] > 0
    }
    attn_df = pd.DataFrame({
        "species":           list(species_mean_attention.keys()),
        "mean_attention":    list(species_mean_attention.values()),
        "n_samples_present": [species_presence_count[sp] for sp in species_mean_attention],
    }).sort_values("mean_attention", ascending=False).reset_index(drop=True)
    attn_df.to_csv(result_path / "species_attention_scores.csv", index=False)
    print(f"\nTop 10 species by attention:\n{attn_df.head(10).to_string(index=False)}")

    #Attention vs log2FC 
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    eps          = 1e-6
    mean_healthy = X_train_np[y_np == 0].mean(axis=0) + eps
    mean_disease = X_train_np[y_np == 1].mean(axis=0) + eps
    log2fc       = np.log2(mean_disease / mean_healthy)

    fc_df  = pd.DataFrame({"species": species_list, "log2fc": log2fc, "abs_log2fc": np.abs(log2fc)})
    merged = attn_df.merge(fc_df, on="species", how="inner")
    rho, pval = stats.spearmanr(merged["mean_attention"], merged["abs_log2fc"])
    _, pvals_corrected, _, _ = multipletests([pval], method="fdr_bh")
    pval_bh = pvals_corrected[0]
    print(f"\nAttention–|log2FC| Spearman: ρ={rho:.4f}, p={pval:.4e}, p_BH={pval_bh:.4e}")
    merged["pval_bh"] = pval_bh
    merged.to_csv(result_path / "attention_foldchange_correlation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged["abs_log2fc"], merged["mean_attention"], alpha=0.4, s=15, color="steelblue")
    ax.set_xlabel("|log₂ Fold-Change| (disease / healthy)")
    ax.set_ylabel("Mean CLS→Species Attention")
    ax.set_title(f"T2D ({strategy_tag}) — Attention vs |log₂FC|\nρ={rho:.3f}, p_BH={pval_bh:.2e}")
    ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(result_path / "attention_foldchange_scatter.png", dpi=300)
    plt.close()

    # UMAP
    Z = np.concatenate(cls_all, axis=0)
    np.save(result_path / "cls_embeddings.npy", Z)
    run_umap(Z, y_labels, label_col="disease",
             title=f"UMAP — {strategy_tag}",
             umap_path=str(result_path / "umap.png"))

    # Summary 
    summary = {
        "disease":          "T2D",
        "freeze_strategy":  args.freeze_strategy,
        "n_unfreeze_layers": args.n_unfreeze_layers,
        "n_samples":        len(y_np),
        "n_disease":        int(np.sum(y_np == 1)),
        "n_healthy":        int(np.sum(y_np == 0)),
        "n_species":        n_species,
        "n_cohorts":        len(np.unique(groups_np)),
        "oof_auc":          results["oof_auc"],
        "oof_f1":           results["oof_f1"],
        "oof_mcc":          results["oof_mcc"],
        "oof_accuracy":     results["oof_accuracy"],
        "mean_auc":         results["mean_auc"],
        "std_auc":          results["std_auc"],
    }
    pd.DataFrame([summary]).to_csv(result_path / "results_summary.csv", index=False)

    plt.figure(figsize=(6, 3))
    plt.barh(["T2D"], [results["oof_auc"]],
             xerr=[[0], [results["std_auc"]]], color="steelblue", capsize=4, alpha=0.85)
    plt.axvline(0.5, color="red", linestyle="--", label="Random (0.5)")
    plt.xlabel("OOF AUC")
    plt.title(f"Nested LOSO AUC — {strategy_tag}")
    plt.xlim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(result_path / "T2D_auc.png", dpi=300)
    plt.close()

    print(f"\nTotal time: {time.time() - begin:.1f}s")
    print(f"  OOF AUC : {results['oof_auc']:.4f}")
    print(f"  Results → {result_path}")


if __name__ == "__main__":
    main()
