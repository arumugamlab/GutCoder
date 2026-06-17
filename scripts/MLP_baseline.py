# -*- coding: utf-8 -*-


# Baseline MLP

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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import umap
from sklearn.metrics import (ConfusionMatrixDisplay,accuracy_score,auc,classification_report,confusion_matrix,f1_score,matthews_corrcoef,roc_auc_score,roc_curve)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from utils.seed import set_seed
from utils.load_data import load_data_mlp
from utils.data_transformation import abundance_binning
from utils.class_resampling import oversampling_gaussian_noise



# MLP
class MLP(nn.Module):
    """
    Plain feed-forward MLP classifier.

    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int,
        num_classes: int,
        n_layers:    int   = 3,
        dropout:     float = 0.3,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["x"])

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the penultimate hidden representation (before final linear)."""
        x = batch["x"]
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x


# Learning rate scheduler
def get_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1),
                               eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


#Training
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
        print("  WARNING: val fold has only one class — AUC set to NaN")
        metrics["auc"] = float("nan")
    elif num_classes == 2:
        metrics["auc"] = float(roc_auc_score(all_labels, probs[:, 1]))
    else:
        metrics["auc"] = float(
            roc_auc_score(all_labels, probs, multi_class="ovr", average="macro")
        )

    return metrics, probs, all_labels


def build_model(input_dim, hidden_dim, num_classes, n_layers, dropout, device):
    return MLP(input_dim=input_dim,hidden_dim=hidden_dim,num_classes=num_classes,n_layers=n_layers,dropout=dropout,).to(device)


def _train_and_eval(
    *,
    train_bin:    np.ndarray,
    train_labels: np.ndarray,
    val_bin:      np.ndarray,
    val_labels:   np.ndarray,
    input_dim:    int,
    hidden_dim:   int,
    num_classes:  int,
    n_layers:     int,
    dropout:      float,
    lr:           float,
    weight_decay: float,
    epochs:       int,
    patience:     int,
    warmup_epochs: int,
    batch_size:   int,
    sigma_aug:    float,
    device:       torch.device,
    fold_seed:    int = 0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

    X_aug, y_aug = oversampling_gaussian_noise(
        train_bin.astype(np.float32), train_labels,
        sigma=sigma_aug, random_state=fold_seed
    )
    rng  = np.random.default_rng(fold_seed)
    perm = rng.permutation(len(y_aug))
    X_aug, y_aug = X_aug[perm], y_aug[perm]

    train_ds = BinnedDataset(X_aug,   y_aug)
    val_ds   = BinnedDataset(val_bin, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = build_model(input_dim, hidden_dim, num_classes, n_layers, dropout, device)

    class_counts  = np.bincount(y_aug, minlength=num_classes).astype(float)
    class_weights = torch.tensor(
        1.0 / np.maximum(class_counts, 1), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    final_metrics, probs, labels = evaluate(model, val_loader, device, num_classes)
    return final_metrics, probs, labels


# Inner grid search
def inner_grid_search(
    bin_mat:       np.ndarray,
    y:             np.ndarray,
    groups:        np.ndarray,
    param_grid:    Dict[str, List],
    input_dim:     int,
    num_classes:   int,
    device:        torch.device,
    weight_decay:  float,
    epochs:        int,
    patience:      int,
    warmup_epochs: int,
    batch_size:    int,
    sigma_aug:     float,
    outer_fold_seed: int = 0,
    verbose:       bool  = False,
) -> Dict:
    keys    = list(param_grid.keys())
    combos  = list(product(*[param_grid[k] for k in keys]))
    configs = [dict(zip(keys, c)) for c in combos]

    inner_studies = np.array([
        s for s in np.unique(groups)
        if len(np.unique(y[groups == s])) >= 2
    ])

    if len(inner_studies) < 2:
        print("  WARNING: fewer than 2 inner studies — returning default config")
        return configs[0]

    print(f"\n  Inner grid search: {len(configs)} configs × "
          f"{len(inner_studies)} inner folds")

    config_scores = []

    for cfg_idx, cfg in enumerate(configs):
        fold_aucs = []
        for inner_fold, held in enumerate(inner_studies):
            val_mask   = groups == held
            train_mask = ~val_mask

            val_labels   = y[val_mask]
            if len(np.unique(val_labels)) < 2:
                continue

            try:
                metrics, _, _ = _train_and_eval(
                    train_bin    = bin_mat[train_mask],
                    train_labels = y[train_mask],
                    val_bin      = bin_mat[val_mask],
                    val_labels   = val_labels,
                    input_dim    = input_dim,
                    hidden_dim   = cfg["hidden_dim"],
                    num_classes  = num_classes,
                    n_layers     = cfg["n_layers"],
                    dropout      = cfg["dropout"],
                    lr           = cfg["lr"],
                    weight_decay = weight_decay,
                    epochs       = epochs,
                    patience     = patience,
                    warmup_epochs= warmup_epochs,
                    batch_size   = batch_size,
                    sigma_aug    = sigma_aug,
                    device       = device,
                    fold_seed    = outer_fold_seed * 1000 + inner_fold,
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
        print(f"  Config {cfg_idx+1:02d}/{len(configs)} "
              f"mean_inner_AUC={mean_auc:.4f}  {cfg}")

    valid = [s for s in config_scores if not np.isnan(s["mean_inner_auc"])]
    if not valid:
        print("  WARNING: all inner configs returned NaN — using default")
        return configs[0]

    best = max(valid, key=lambda s: s["mean_inner_auc"])
    print(f"\n  Best config: {best['config']}  "
          f"(mean inner AUC={best['mean_inner_auc']:.4f})")
    return best["config"]


# Nested LOSO
def nested_loso_cv(
    bin_mat:       np.ndarray,
    y:             np.ndarray,
    groups:        np.ndarray,
    param_grid:    Dict[str, List],
    input_dim:     int,
    num_classes:   int   = 2,
    device:        torch.device = torch.device("cpu"),
    epochs:        int   = 20,
    patience:      int   = 5,
    weight_decay:  float = 1e-2,
    warmup_epochs: int   = 2,
    batch_size:    int   = 64,
    sigma_aug:     float = 0.1,
    log_path             = None,
    fold_plot_dir        = None,
    inner_verbose: bool  = False,
) -> Dict:

    unique_studies = np.array([
        s for s in np.unique(groups)
        if len(np.unique(y[groups == s])) >= 2
    ])
    n_outer = len(unique_studies)

    fold_results     = []
    oof_probs_list   = []
    oof_preds_list   = []
    oof_labels_list  = []
    fold_val_indices = []
    all_fold_histories = []
    best_state_global  = None

    print(f"\nNested LOSO CV: {n_outer} outer folds")
    print(f"Grid: {param_grid}")

    for fold, held_out_study in enumerate(unique_studies):
        val_mask   = groups == held_out_study
        train_mask = ~val_mask
        val_idx    = np.where(val_mask)[0]
        train_idx  = np.where(train_mask)[0]

        print(f"\n{'='*65}")
        print(f"  Outer fold {fold+1}/{n_outer}  |  Held-out: '{held_out_study}'")
        print(f"  Train size: {len(train_idx)}  |  Val size: {len(val_idx)}")
        print(f"  Train class dist: {dict(zip(*np.unique(y[train_idx], return_counts=True)))}")
        print(f"  Val   class dist: {dict(zip(*np.unique(y[val_idx],   return_counts=True)))}")

        fold_val_indices.append(val_idx)

        # Inner grid search
        best_cfg = inner_grid_search(
            bin_mat        = bin_mat[train_idx],
            y              = y[train_idx],
            groups         = groups[train_idx],
            param_grid     = param_grid,
            input_dim      = input_dim,
            num_classes    = num_classes,
            device         = device,
            weight_decay   = weight_decay,
            epochs         = epochs,
            patience       = patience,
            warmup_epochs  = warmup_epochs,
            batch_size     = batch_size,
            sigma_aug      = sigma_aug,
            outer_fold_seed= fold,
            verbose        = inner_verbose,
        )

        hidden_dim = best_cfg["hidden_dim"]
        n_layers   = best_cfg["n_layers"]
        dropout    = best_cfg["dropout"]
        lr         = best_cfg["lr"]

        # Retraining with best hyperparameter combination
        
        X_train_aug, y_train_aug = oversampling_gaussian_noise(
            bin_mat[train_idx].astype(np.float32), y[train_idx],
            sigma=sigma_aug, random_state=fold
        )
        rng  = np.random.default_rng(fold)
        perm = rng.permutation(len(y_train_aug))
        X_train_aug = X_train_aug[perm]
        y_train_aug = y_train_aug[perm]

        train_ds = BinnedDataset(X_train_aug,      y_train_aug)
        val_ds   = BinnedDataset(bin_mat[val_idx], y[val_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        model = build_model(input_dim, hidden_dim, num_classes, n_layers, dropout, device)

        class_counts  = np.bincount(y_train_aug, minlength=num_classes).astype(float)
        class_weights = torch.tensor(
            1.0 / np.maximum(class_counts, 1), dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler = get_scheduler(optimizer, warmup_epochs, epochs)

        best_auc   = -1.0
        no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        fold_history = []

        for epoch in range(1, epochs + 1):
            train_loss        = train_one_epoch(model, train_loader, optimizer,
                                                criterion, device)
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
        final_metrics, fold_probs, fold_labels = evaluate(
            model, val_loader, device, num_classes
        )

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
              f"Config={best_cfg} | "
              f"AUC={final_metrics['auc']:.4f} | F1={final_metrics['f1']:.4f} | "
              f"MCC={final_metrics['mcc']:.4f} | Acc={final_metrics['accuracy']:.4f}")
        print(classification_report(fold_labels, fold_probs.argmax(axis=-1),
                                    zero_division=0))

        best_state_global = best_state

        if fold_plot_dir is not None:
            _plot_fold_curves(fold_history, fold + 1, fold_plot_dir)

    # OOF 
    y_true_all = np.concatenate(oof_labels_list)
    y_pred_all = np.concatenate(oof_preds_list)
    y_prob_all = np.concatenate(oof_probs_list, axis=0)

    if num_classes == 2:
        oof_auc = roc_auc_score(y_true_all, y_prob_all[:, 1])
    else:
        oof_auc = roc_auc_score(y_true_all, y_prob_all,
                                multi_class="ovr", average="macro")

    oof_f1  = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    oof_mcc = matthews_corrcoef(y_true_all, y_pred_all)
    oof_acc = accuracy_score(y_true_all, y_pred_all)

    print(f"\n{'='*65}")
    print("Overall OOF Performance (nested LOSO):")
    print(f"  AUC      : {oof_auc:.4f}")
    print(f"  F1       : {oof_f1:.4f}")
    print(f"  MCC      : {oof_mcc:.4f}")
    print(f"  Accuracy : {oof_acc:.4f}")
    print("\nOverall Classification Report:")
    print(classification_report(y_true_all, y_pred_all, zero_division=0))

    print("\nPer-study (fold) variance:")
    for metric in ["auc", "f1", "mcc", "accuracy"]:
        vals       = [r[metric] for r in fold_results]
        valid_vals = [v for v in vals if not np.isnan(v)]
        n_nan      = len(vals) - len(valid_vals)
        nan_note   = f" ({n_nan} NaN folds excluded)" if n_nan else ""
        print(f"  {metric.upper():10s}: {np.mean(valid_vals):.4f} ± "
              f"{np.std(valid_vals, ddof=1):.4f}{nan_note}")

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
        "fold_histories":                all_fold_histories,
        "best_state":                    best_state_global,
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
        

    return results


# Plotting
def _plot_fold_curves(history: List[Dict], fold: int, save_dir: str):
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
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    plt.savefig(p / f"fold_{fold}_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix",
                           cm_path=None):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=class_names).plot(ax=ax1, colorbar=False, cmap="Blues")
    ConfusionMatrixDisplay(confusion_matrix=np.round(cm_norm, 2),
                           display_labels=class_names).plot(ax=ax2, colorbar=False, cmap="Blues")
    ax1.set_title(f"{title} (Counts)")
    ax2.set_title(f"{title} (Normalised)")

    plt.tight_layout()
    if cm_path is not None:
        Path(cm_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, fold_results, oof_probs, y,
                   fold_val_indices, title="ROC Curve", roc_path=None):
    fig, ax = plt.subplots(figsize=(8, 7))

    for fold_idx, val_idx in enumerate(fold_val_indices):
        fold_probs_pos = oof_probs[val_idx, 1]
        fold_labels    = y[val_idx]
        fold_auc       = fold_results[fold_idx]["auc"]

        if np.isnan(fold_auc) or len(np.unique(fold_labels)) < 2:
            continue

        fpr, tpr, _ = roc_curve(fold_labels, fold_probs_pos)
        ax.plot(fpr, tpr, color="steelblue", alpha=0.3, linewidth=1,
                label=f"Fold {fold_idx+1} (AUC={fold_auc:.3f})")

    fpr_oof, tpr_oof, _ = roc_curve(y_true, y_prob)
    oof_auc_val          = auc(fpr_oof, tpr_oof)
    ax.plot(fpr_oof, tpr_oof, color="red", linewidth=2.5,
            label=f"Overall OOF (AUC={oof_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")

    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()

    if roc_path is not None:
        Path(roc_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()


# UMAP on last-before representations, as this is the heavily compressed, high-level feature space just before the final layer
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    reps = []
    for batch in loader:
        batch.pop("label", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        reps.append(model.encode(batch).cpu().numpy())
    return np.concatenate(reps, axis=0)


def run_umap(Z, y_labels, label_col="disease", title="UMAP of MLP Embeddings",
             umap_path=None, random_state=42):
    y_disease = y_labels[label_col].astype(str).to_numpy()
    reducer   = umap.UMAP(n_components=2, random_state=random_state)
    Z_umap    = reducer.fit_transform(Z)

    plt.figure(figsize=(6, 5))
    for lab in np.unique(y_disease):
        idx = y_disease == lab
        plt.scatter(Z_umap[idx, 0], Z_umap[idx, 1], s=8, alpha=0.8, label=lab)
    plt.title(f"{title} — {label_col}")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if umap_path is not None:
        Path(umap_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(umap_path, dpi=300)
    plt.close()
    return Z_umap


# Main function
def main():
    set_seed(42)
    begin = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    data_dir     = project_root / "CMD_data" / "binary_healthy_X"
    output_dir   = project_root / "results"

    parser = argparse.ArgumentParser(description="MLP fine-tuning — Healthy vs T2D")
    parser.add_argument("--train_csv",    type=str,   default=str(data_dir / "X_T2D.csv"))
    parser.add_argument("--labels_csv",   type=str,   default=str(data_dir / "y_T2D.csv"))
    parser.add_argument("--groups_csv",   type=str,   default=str(data_dir / "cohort_T2D.csv"))
    parser.add_argument("--output_dir",   type=str,   default=str(output_dir / "MLP_T2D_nested"))
    parser.add_argument("--bins",         type=int,   default=40)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--wdecay",       type=float, default=0.01)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--inner_verbose", action="store_true")

    # Grid search axes
    parser.add_argument("--grid_hidden_dims", type=str, default="64,128",
                        help="Comma-separated hidden_dim values.")
    parser.add_argument("--grid_n_layers",    type=str, default="3",
                        help="Comma-separated n_layers values.")
    parser.add_argument("--grid_dropouts",    type=str, default="0.3,0.5",
                        help="Comma-separated dropout values.")
    parser.add_argument("--grid_lrs",         type=str, default="1e-4,5e-4",
                        help="Comma-separated learning rate values.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_grid = {
        "hidden_dim": [int(v)   for v in args.grid_hidden_dims.split(",")],
        "n_layers":   [int(v)   for v in args.grid_n_layers.split(",")],
        "dropout":    [float(v) for v in args.grid_dropouts.split(",")],
        "lr":         [float(v) for v in args.grid_lrs.split(",")],
    }
    print(f"\nParam grid: {param_grid}")

    run_name    = f"bins{args.bins}_epoch{args.epochs}_nested_loso"
    result_path = output_dir / run_name
    result_path.mkdir(parents=True, exist_ok=True)

    cm_path   = result_path / "confusion_matrix.png"
    roc_path  = result_path / "roc_plot.png"
    emb_path  = result_path / "mlp_embeddings.npy"
    umap_path = result_path / "umap.png"

    
    X_train, y_labels, groups = load_data_cohort(args.train_csv, args.labels_csv, args.groups_csv)
    X_train_np = X_train.to_numpy()

    label_map = {"healthy": 0, "T2D": 1}
    y_np      = np.array([label_map[v] for v in y_labels["disease"].to_numpy()])
    groups_np = groups["study_name"].to_numpy()

    n_species = X_train_np.shape[1]
    print(f"  Samples  : {len(y_np)}")
    print(f"  Species  : {n_species}")
    print(f"  Cohorts  : {np.unique(groups_np).tolist()}")

    bin_mat = abundance_binning(X_train_np, bins=args.bins)
    print(f"  bin_mat  : {bin_mat.shape}, {bin_mat.dtype}")

    
    results = nested_loso_cv(
        bin_mat        = bin_mat,
        y              = y_np,
        groups         = groups_np,
        param_grid     = param_grid,
        input_dim      = n_species,
        num_classes    = 2,
        device         = device,
        epochs         = args.epochs,
        patience       = 5,
        weight_decay   = args.wdecay,
        warmup_epochs  = 2,
        batch_size     = args.batch_size,
        sigma_aug      = 0.1,
        log_path       = str(result_path / "cv_results.json"),
        fold_plot_dir  = str(result_path / "fold_curves"),
        inner_verbose  = args.inner_verbose,
    )

 
    best_state_path = result_path / "best_mlp_model.pt"
    torch.save(results["best_state"], str(best_state_path))
    print(f"  Best weights saved → {best_state_path}")

    y_true_all  = results["y_true_all"]
    y_pred_all  = results["y_pred_all"]
    y_prob_all  = results["y_prob_all"]
    class_names = ["healthy", "T2D"]

    plot_confusion_matrix(y_true=y_true_all, y_pred=y_pred_all,
                          class_names=class_names, title="Confusion Matrix",
                          cm_path=cm_path)
    plot_roc_curve(
        y_true           = y_true_all,
        y_prob           = y_prob_all[:, 1],
        fold_results     = results["fold_results"],
        oof_probs        = y_prob_all,
        y                = y_np,
        fold_val_indices = results["fold_val_indices"],
        title            = "ROC Curve (nested LOSO — MLP)",
        roc_path         = roc_path,
    )

    
    best_cfg    = results["fold_results"][-1]["best_config"]
    embed_model = build_model(
        input_dim   = n_species,
        hidden_dim  = best_cfg["hidden_dim"],
        num_classes = 2,
        n_layers    = best_cfg["n_layers"],
        dropout     = best_cfg["dropout"],
        device      = device,
    )
    embed_model.load_state_dict(
        {k: v.to(device) for k, v in results["best_state"].items()}
    )
    embed_model.eval()

    full_ds     = BinnedDataset(bin_mat)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)
    Z           = extract_embeddings(embed_model, full_loader, device)
    np.save(emb_path, Z)

    run_umap(Z=Z, y_labels=y_labels, label_col="disease",
             title="UMAP of MLP Embeddings", umap_path=umap_path, random_state=42)

    
    summary = {
        "disease":      "T2D",
        "n_samples":    len(y_np),
        "n_disease":    int(np.sum(y_np == 1)),
        "n_healthy":    int(np.sum(y_np == 0)),
        "n_species":    n_species,
        "n_cohorts":    len(np.unique(groups_np)),
        "oof_auc":      results["oof_auc"],
        "oof_f1":       results["oof_f1"],
        "oof_mcc":      results["oof_mcc"],
        "oof_accuracy": results["oof_accuracy"],
        "mean_auc":     results["mean_auc"],
        "std_auc":      results["std_auc"],
    }
    summary_df   = pd.DataFrame([summary])
    summary_path = result_path / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
   

    plt.figure(figsize=(6, 3))
    plt.barh(["T2D"], [results["oof_auc"]],
             xerr=[[0], [results["std_auc"]]], color="steelblue", capsize=4, alpha=0.85)
    plt.axvline(0.5, color="red", linestyle="--", label="Random (0.5)")
    plt.xlabel("OOF AUC"); plt.title("Nested LOSO AUC (MLP)"); plt.xlim(0, 1)
    plt.legend(); plt.tight_layout()
    plt.savefig(result_path / "T2D_auc.png", dpi=300)
    plt.close()

    end = time.time()
   


if __name__ == "__main__":
    main()
