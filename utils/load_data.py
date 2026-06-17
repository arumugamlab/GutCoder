# Loading data - Autoencoder, transformer, MLP
import pandas as pd
def load_data(X_path, y_path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    return X, y

# Loading data for classification
def load_data_cohort(X_path, y_path, groups_path):
    X_train  = pd.read_csv(X_path)
    y_labels = pd.read_csv(y_path)
    groups   = pd.read_csv(groups_path)
    return X_train, y_labels, groups
