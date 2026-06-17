# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 13:08:11 2026

@author: shabnam
"""

import time

# Module import
import argparse
from pathlib import Path
import numpy as np
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple


# Setting project root
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "CMD_data"
output_dir = data_dir 


# Load data
def load_data(abundance_path):
    abundance_df = pd.read_csv(abundance_path, index_col=0)
    return abundance_df

# Data split
def train_eval_split(X: pd.DataFrame,y: pd.Series, groups: pd.Series, metadata:pd.Series, gut_mask:pd.Series,healthy_label: str = "healthy",test_size: float = 0.20,random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    # All diseases except healthy
    nonhealthy = y.astype(str) != healthy_label
    y_nonhealthy = y[nonhealthy]
    groups_nonhealthy = groups[nonhealthy]
 
    disease_types = []
    for d in y_nonhealthy.dropna().unique():
        d_mask = y_nonhealthy == d
        disease_types.append({"disease": d,"n_samples": int(d_mask.sum()),"n_study": int(groups_nonhealthy[d_mask].nunique())})

    disease_types_df = pd.DataFrame(disease_types)
    eligible_diseases = disease_types_df[disease_types_df["n_study"] > 2].copy()

    print(f"Eligible diseases for external validation: {len(eligible_diseases)}/{len(disease_types_df)}")
    print(eligible_diseases[["disease", "n_study", "n_samples"]].to_string(index=False))

    eligible_set = set(eligible_diseases["disease"].astype(str))

    # Eligible studies = studies containing >=1 eligible disease
    eligible_studies = groups_nonhealthy[y_nonhealthy.astype(str).isin(eligible_set)].dropna().unique()
    eligible_studies = np.asarray(eligible_studies)

    if len(eligible_studies) < 2:
        raise ValueError(f"Not enough eligible studies to split: {len(eligible_studies)} found.")

    # split studies 80/20 (by studies)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    dummy_X = np.zeros((len(eligible_studies), 1))

    train_study_idx, test_study_idx = next(gss.split(dummy_X, y=None, groups=eligible_studies))
    test_studies = set(eligible_studies[test_study_idx])

    # Apply to ALL samples including healthy
    external_mask = groups.isin(test_studies)
    train_mask = ~external_mask
    
    gut_train_mask = train_mask & gut_mask
    


    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_ext = X.loc[external_mask]
    y_ext = y.loc[external_mask]
    X_gut_train = X.loc[gut_train_mask]
    y_gut_train = y.loc[gut_train_mask]
    metadata_train = metadata.loc[train_mask]
    metadata_external = metadata.loc[external_mask]
    metadata_gut_train = metadata.loc[gut_train_mask]
   
    print(f"Eligible studies total: {len(eligible_studies)}")
    
    print(f"External studies: {len(test_studies)}")
    print(f"External studies: {test_studies}")
    print(f"Train samples: {len(X_train)} | External samples: {len(X_ext)}")
    print(f"Gut Train samples: {len(X_gut_train)} | All train samples: {len(X_train)}")

    return X_train, X_ext, X_gut_train, y_gut_train, metadata_gut_train, y_train, y_ext, metadata_train, metadata_external
    
def main():
    
    begin = time.time()
    
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "CMD_data"
    output_dir = data_dir 
    
    parser = argparse.ArgumentParser(description="Train-ExternalValidation-split")
    
    parser.add_argument('--abundance_csv', type=str, default=str(data_dir/"raw"/"relative_abundance_matrix.csv"), help='Relative abundance file')
    
    parser.add_argument('--output_dir', type=str, default=str("CMD_data"), help='Save files here')
    
    args = parser.parse_args()  # Parse command-line arguments
    abundance_path = Path(args.abundance_csv)
    output_dir = Path(args.output_dir)
    #labels_path = Path(args.labels_csv)
    
    
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    abundance_df = load_data(abundance_path)
    X=abundance_df.drop(["disease","study_name", "body_site","DNA_extraction_kit", "gender", "sequencing_platform", "age_category", "country", "study_condition","subject_id"],axis=1)
    y = abundance_df["disease"]
    metadata = abundance_df[["disease","study_name","DNA_extraction_kit","body_site","age_category","gender","sequencing_platform", "country", "study_condition","subject_id"]]
    gut_mask = abundance_df["body_site"] == "stool"
    
    

   
   
    groups = abundance_df["study_name"]
    print(X.shape)
    print(y.shape)
    
   
    X_train, X_ext, X_gut_train, y_gut_train, metadata_gut_train, y_train, y_ext, metadata_train, metadata_external = train_eval_split(X=X, y=y,groups=groups,metadata=metadata,gut_mask=gut_mask,random_state=42)
    X_train_np=X_train.to_numpy()
    
    
    
    (X_train).to_csv(output_dir / "X_train.csv", index=False)
    (X_gut_train).to_csv(output_dir / "X_gut_train.csv", index=False)
    (y_gut_train).to_csv(output_dir / "y_gut_train.csv", index=False)
    (y_train).to_csv(output_dir / "y_train.csv", index=False)
    (metadata_external).to_csv(output_dir / "metadata_external.csv", index=False)
    
    (X_ext).to_csv(output_dir / "X_external.csv", index=False)
    (y_ext).to_csv(output_dir / "y_external.csv", index=False)
    (metadata_train).to_csv(output_dir / "gut_meta_adv.csv", index=False)
    (gut_mask).to_csv(output_dir / "gut.csv", index=False)
    (metadata_gut_train).to_csv(output_dir / "metadata_gut.csv", index=False)
    
    end = time.time()
    Time = end - begin
    print(f"Time: {Time}")
    
if __name__ == "__main__":
    main()
