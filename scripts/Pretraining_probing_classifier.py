# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:17:24 2026
Logistic Regression and KNN - Pretraining probing classifiers
Reporting only LR
@author: shab3
"""

import time
from pathlib import Path
import argparse
import math
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, auc
from sklearn.metrics import classification_report, roc_curve

# Load data
def load_embeddings (X_path, meta_path):
    
    X = np.load (X_path)
    y = pd.read_csv (meta_path)
    #groups = pd.read_csv (groups_path)

    
    return X,y

# Logistic Regression and knn models

def logistic_Regression_knn (X,y,groups,outer_splits: int =5, inner_splits: int=5,random_state: int = 42, scoring: str = "f1_macro",):

    X = np.asarray(X)
    y = np.asarray(y)
  
    groups = np.asarray(groups)
    
    
    # Label encoder
    lab = LabelEncoder()
    y_encode = lab.fit_transform(y)
    n_classes = lab.classes_
    
    
    '''    
    
    If multi-class, then finding rare classes and only taking the eligible classes because if classes are represented in only 1 study then evaluation becomes impossible
    
    def macro_auroc(y_true, y_prob):
      if len(np.unique(y_true)) < 2:
        return np.nan
      return roc_auc_score(y_true, y_prob[:, 1])

     
    
    encode_study_df = pd.DataFrame({"y": y_encode, "group": groups})
    study_counts = (encode_study_df.drop_duplicates().groupby("y")["group"].nunique().sort_values())

    rare_classes_enc = study_counts[study_counts < min_studies_per_class].index.to_numpy()
    rare_classes = lab.inverse_transform(rare_classes_enc) if rare_classes_enc.size else np.array([], dtype=object)
    
    
    rare_mask = np.isin(y_encode, rare_classes_enc) if rare_classes_enc.size else np.zeros_like(y_encode, dtype=bool)
    rare_idx = np.where(rare_mask)[0]
    eligible_idx = np.where(~rare_mask)[0]
    '''
    
    models = {"logistic_reg": {"pipeline": Pipeline([("scaler", StandardScaler()),("model", LogisticRegression(solver="lbfgs",max_iter=10000))]), "param_grid": {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__class_weight": [None, "balanced"],
                "model__penalty":["l2"]}},
            "knn": {"pipeline": Pipeline([("scaler", StandardScaler()),("model", KNeighborsClassifier())]),"param_grid": {"model__n_neighbors":[3, 5, 10, 15, 25],"model__weights": ["uniform", "distance"]}}}

    outer_cv = StratifiedGroupKFold(n_splits=outer_splits,shuffle=True,random_state=random_state)
    
    each_model = {name: {"acc": [], "f1": [], "auc": [], "best_params": [], "fold_reports": [], "fold_confusion_matrices":[],"oof_y_true":[],"oof_y_pred":[],"oof_y_prob":[]} for name in models}
    
    
    
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encode, groups=groups),start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encode[train_idx], y_encode[test_idx]
        g_train = groups[train_idx]
        
        inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        
      
        for name, use in models.items():
            
            grid = GridSearchCV(use["pipeline"],use["param_grid"],cv=inner_cv,scoring=scoring,n_jobs=-1,refit=True)
            
            grid.fit(X_train, y_train, groups=g_train)
            y_pred = grid.predict(X_test)
            y_prob = grid.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")
            auc = roc_auc_score(y_test, y_prob[:,1])
            confusion_mat = confusion_matrix(y_test,y_pred,labels=[0,1])
            
            report = classification_report(
                y_test,
                y_pred,
                target_names=n_classes,
                output_dict=True,
                zero_division=0
            )
            
            
            each_model[name]["acc"].append(acc)
            each_model[name]["f1"].append(f1m)
            each_model[name]["auc"].append(auc)
            each_model[name]["best_params"].append(grid.best_params_)
            each_model[name]["fold_confusion_matrices"].append(confusion_mat)
            each_model[name]["fold_reports"].append(report)
            each_model[name]["oof_y_true"].append(y_test)
            each_model[name]["oof_y_pred"].append(y_pred)
            each_model[name]["oof_y_prob"].append(y_prob)
            
           
    summary = {}       
    overall_reports = {}
    overall_confusion_matrices = {}
    
    for name in models:
      y_true_all = np.concatenate(each_model[name]["oof_y_true"], axis=0)
      y_pred_all = np.concatenate(each_model[name]["oof_y_pred"], axis=0)
      y_prob_all = np.concatenate(each_model[name]["oof_y_prob"])

     

      overall_reports[name] = classification_report(
            y_true_all,
            y_pred_all,
            target_names=n_classes, 
            output_dict=True,
            zero_division=0
        )
      
      overall_confusion_matrices[name] = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

      accs = np.array(each_model[name]["acc"], dtype=float)
      f1s = np.array(each_model[name]["f1"], dtype=float)
      aucs = np.array(each_model[name]["auc"], dtype=float)
      summary[name] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        }
         
     

    return {
        "label_encoder": lab,
        "classes": n_classes,
        "each_model": each_model,
        "summary": summary,
        "confusion_mat": confusion_mat,
        "overall_classification_report": overall_reports,
        "overall_confusion_matrices": overall_confusion_matrices,
    }
    

# Plotting confusion matrix
def plot_confusion_matrix(confusion_mat, n_classes, title="Confusion Matrix", normalize=False, save_cm_path: str | None = None):
    cm = np.asarray(confusion_mat, dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(n_classes)))
    ax.set_yticks(np.arange(len(n_classes)))
    ax.set_xticklabels(n_classes)
    ax.set_yticklabels(n_classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()

    # Save plot
    if save_cm_path is not None:
        p = Path(save_cm_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()

# Plotting ROC curve
def plot_binary_roc(y_true, y_prob, title="ROC Curve", save_roc_path: str | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    # Save plot
    if save_roc_path is not None:
        p = Path(save_roc_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()
    
def main():
    
    begin = time.time()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "results" 
    lab_dir = project_root / "CMD_data"
    output_dir = project_root / "results"
    embed_dir = data_dir / "Phase1_PT_gut" /"selected"
    
    
    parser = argparse.ArgumentParser(description="Logistic regression and KNN model training")
    #parser.add_argument('--cls_dir', type=str, default=str(embed_dir), help='embeddings file')
    parser.add_argument('--labels_csv', type=str, default=str(lab_dir/"gut_meta_adv.csv"), help='Labels file')
    parser.add_argument("--study_col",   type=str, default="study_name", help="Metadata column containing study identifiers")
    parser.add_argument("--disease_col",   type=str, default="body_site", help="Metadata column containing disease identifiers")
    #parser.add_argument('--groups_csv', type=str, default=str(lab_dir/"groups.csv"), help='Study names')
    parser.add_argument('--output_dir', type=str, default=str(output_dir/"Classifiers"), help='Directory to save model outputs')
    
    args = parser.parse_args()  # Parse command-line arguments
    #X_path = Path(args.cls_npy)
    meta_path = Path(args.labels_csv)
    #groups_path = meta_path[args.study_col].values
    #y_path = meta_path[args.disease_col].values
   
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    base_dir = Path(embed_dir)

    for exp_dir in base_dir.iterdir():
      if not exp_dir.is_dir():
        continue

      print(f"\nRunning experiment: {exp_dir.name}")

   
      X_path = exp_dir / "cls_embeddings.npy"
      

      if not X_path.exists():
        print("No embeddings found. Skipping.")
        continue

      
      X, y = load_embeddings(X_path, meta_path)
      y_d=y["body_site"].to_numpy()
      groups=y["study_name"].to_numpy()
      
      
    # Run evaluation function
      out = logistic_Regression_knn(X, y_d, groups)
      
    # Plot confusion matrix
      for model_name, confusion_mat in out["overall_confusion_matrices"].items():
          
        plot_confusion_matrix(confusion_mat,n_classes=out["classes"], title=f"{model_name}: Confusion Matrix", normalize=True, save_cm_path= f"{exp_dir}/plots/confusion_{model_name}.png")
    
      
    # Plot binary roc curve
      for model_name, model_result in out["each_model"].items():
        y_true_all = np.concatenate(model_result["oof_y_true"])
        y_prob_all = np.concatenate(model_result["oof_y_prob"])

        plot_binary_roc(y_true_all, y_prob_all, title=f"{model_name}: ROC Curve", save_roc_path = f"{exp_dir}/plots/roc_{model_name}.png")

    # Save summary inside same folder
      with open(exp_dir / "results.json", "w") as f:
        json.dump(out["summary"], f, indent=2)
      
      with open(exp_dir / "classification_report.json", "w") as f:
        json.dump(out["overall_classification_report"], f, indent=2)
    
    
      
      
    
    
    
    
    
    '''
    
    X, y,groups = load_embeddings(X_path, y_path, groups_path)
    y=y["body_site"].to_numpy()
    groups=groups["study_name"].to_numpy()
    
    out = logistic_Regression_knn(X, y, groups)
    
    with open(output_dir/"logreg_classification_report.json", "w") as f:
      json.dump(out["overall_classification_report"]["logistic_reg"],f,indent=2)

    with open(output_dir/"knn_classification_report.json", "w") as f:
      json.dump(out["overall_classification_report"]["knn"],f,indent=2)
      
    with open(output_dir/"knn_summary.json", "w") as f:
      json.dump(out["summary"]["knn"],f,indent=2)
      
    with open(output_dir/"log_summary.json", "w") as f:
      json.dump(out["summary"]["logistic_reg"],f,indent=2)
      '''
   
   

   
    end = time.time()
    Time = end - begin
    print(f"Time: {Time}")

# Run main if script is executed directly
if __name__ == "__main__":
    main()
    
    
