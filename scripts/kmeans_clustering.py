# -*- coding: utf-8 -*-
"""
Last updated on Wed June 3 2026

@author: shab3

Steps:
  1. Load pretrained embeddings + metadata (body-site labels)
  2. Find optimal K for KMeans (silhouette + elbow)
  3. Fit KMeans with optimal K
  4. t-SNE of embeddings, coloured by KMeans cluster
  5. t-SNE of embeddings, coloured by true body-site label
  6. Side-by-side comparison plot (clusters vs labels)
  7. Cluster composition table CSV (for R stacked barchart)
  8. Pairwise centroid distances between clusters
"""

import argparse

import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist


PALETTE = "PiYG"


def load_data(embeddings_path: str, labels_path: str, label_col: str):
    """
    Embeddings: .npy array of shape (N, D)
    Labels:     .csv with at least one column named `label_col` (N rows, same order)
    """
  
    X = np.load(embeddings_path)
  
    meta = pd.read_csv(labels_path)
    
    labels = meta[label_col].astype(str).values
   
    return X, labels, meta


# Finding optimal k using elbow plot, silhouette score, DB score
# Choosing based on silhouette score
def find_optimal_k(X: np.ndarray, k_min: int, k_max: int, out_dir: str) -> int:
   
    ks = range(k_min, k_max + 1)
    inertias, silhouettes, db_scores = [], [], []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_km = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels_km, sample_size=min(5000, len(X))))
        db_scores.append(davies_bouldin_score(X, labels_km))
        

    best_k = ks[int(np.argmax(silhouettes))]
   
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, y, title, best_fn in zip(
        axes,
        [inertias, silhouettes, db_scores],
        ["Inertia (Elbow)", "Silhouette Score", "Davies-Bouldin Score ?"],
        [None, max, min],
    ):
        ax.plot(list(ks), y, marker="o", color="#2563EB")
        if best_fn:
            best_val = best_fn(y)
            best_x = list(ks)[y.index(best_val)]
            ax.axvline(best_x, color="#DC2626", linestyle="--", alpha=0.7,
                       label=f"Best K={best_x}")
            ax.legend(fontsize=9)
        ax.set_xlabel("Number of Clusters K")
        ax.set_title(title)
        ax.set_xticks(list(ks))

    plt.suptitle("KMeans Cluster Selection Metrics", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "kmeans_selection.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    
    return best_k


#Fit kmeans

def fit_kmeans(X: np.ndarray, k: int):
  
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    cluster_labels = km.fit_predict(X)
    return km, cluster_labels

#t-sne plot

def run_tsne(X: np.ndarray, perplexity: float = 50.0) -> np.ndarray:
  
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,metric = "cosine",
                max_iter=1000, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(X)
    return Z


def plot_tsne_comparison(Z: np.ndarray, cluster_labels: np.ndarray,
                         gender_labels: np.ndarray, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    k = len(np.unique(cluster_labels))
    cmap_k = plt.get_cmap(PALETTE, k)
    for c in range(k):
        mask = cluster_labels == c
        axes[0].scatter(Z[mask, 0], Z[mask, 1],
                        s=10, alpha=0.8, color=cmap_k(c), label=f"Cluster {c}", rasterized=True)
    axes[0].set_title(f"t-SNE coloured by KMeans (K={k})")
    axes[0].legend(markerscale=8, fontsize=14, ncol=2,
                   loc="upper right", framealpha=0.5)
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")

    unique_sites = sorted(np.unique(gender_labels))
    cmap_s = plt.get_cmap("Dark2", len(unique_sites))
    site_to_idx = {s: i for i, s in enumerate(unique_sites)}
    colors = [cmap_s(site_to_idx[s]) for s in gender_labels]
    axes[1].scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.8, c=colors, rasterized=True)
    patches = [mpatches.Patch(color=cmap_s(i), label=s)
               for i, s in enumerate(unique_sites)]
    axes[1].legend(handles=patches, markerscale=6, fontsize=14, ncol=2,
                   loc="upper right", framealpha=0.5)
    axes[1].set_title("t-SNE coloured by Healthy and IBD labels")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

    plt.suptitle("Comparison of k-means clusters with Healthy and IBD labels", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "tsne_comparison_IBD.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

# Saving cluster compositions to csv file for further analysis
def cluster_composition(cluster_labels: np.ndarray, disease_labels: np.ndarray,
                        out_dir: str) -> pd.DataFrame:
  
    df = pd.DataFrame({"cluster": cluster_labels, "disease": disease_labels})
    comp = (df.groupby(["cluster", "disease"])
              .size()
              .reset_index(name="count"))
    comp["total_in_cluster"] = comp.groupby("cluster")["count"].transform("sum")
    comp["proportion"] = comp["count"] / comp["total_in_cluster"]

    # For convinience in R
    wide = comp.pivot_table(index="cluster", columns="disease",
                            values="proportion", fill_value=0).reset_index()
    path_long = os.path.join(out_dir, "cluster_composition_long.csv")
    path_wide = os.path.join(out_dir, "cluster_composition_wide.csv")
    comp.to_csv(path_long, index=False)
    wide.to_csv(path_wide, index=False)
    return comp, wide


# Pairwise distance between cluster centroids for quantification
def pairwise_centroid_distances(km: KMeans, out_dir: str):
   
    centroids = km.cluster_centers_          
    k = centroids.shape[0]

    # Euclidean distances between all pairs
    dist_matrix = cdist(centroids, centroids, metric="euclidean")
    labels = [f"C{i}" for i in range(k)]
    df_dist = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    path = os.path.join(out_dir, "centroid_distances.csv")
    df_dist.to_csv(path)
    

    # Most and least similar pairs
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append((f"C{i}", f"C{j}", dist_matrix[i, j]))
    pairs_df = pd.DataFrame(pairs, columns=["cluster_a", "cluster_b", "distance"])
    pairs_df = pairs_df.sort_values("distance", ascending=False).reset_index(drop=True)

    print("Most different cluster pairs:")
    print(pairs_df.head(5).to_string(index=False))
    print("\n Most similar cluster pairs:")
    print(pairs_df.tail(5).to_string(index=False))

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(6, k * 0.7), max(5, k * 0.6)))
    sns.heatmap(df_dist, annot=k <= 20, fmt=".2f", cmap="viridis_r",
                square=True, linewidths=0.5, ax=ax, cbar_kws={"label": "Euclidean Distance"})
    ax.set_title("Pairwise Centroid Distances Between Clusters",pad=12)
    plt.tight_layout()
    path_hm = os.path.join(out_dir, "centroid_distances_heatmap.png")
    plt.savefig(path_hm, bbox_inches="tight", dpi=150)
    plt.close()
   
    return df_dist, pairs_df


def intra_cluster_distances(km: KMeans, embeddings: np.ndarray, out_dir: str):
   
    labels = km.labels_
    centroids = km.cluster_centers_
    k = centroids.shape[0]

    rows = []
    for i in range(k):
        mask = labels == i
        cluster_points = embeddings[mask]
        
        if len(cluster_points) == 0:
            rows.append((f"C{i}", 0, 0, 0.0))
            continue
        
        # Distance of each point to its centroid
        dists = cdist(cluster_points, centroids[i].reshape(1, -1), metric="euclidean").flatten()
        rows.append((f"C{i}", len(cluster_points), dists.mean(), dists.std()))

    df_intra = pd.DataFrame(rows, columns=["cluster", "n_samples", "mean_intra_dist", "std_intra_dist"])
    
    path = os.path.join(out_dir, "intra_cluster_distances.csv")
    df_intra.to_csv(path, index=False)
    print("\nIntra-cluster distances:")
    print(df_intra.to_string(index=False))

    return df_intra

def save_cluster_labels(cluster_labels: np.ndarray, meta: pd.DataFrame, 
                        id_col: str, out_dir: str, prefix: str = ""):
    """Save cluster labels with sample IDs for later comparison."""
    df = pd.DataFrame({
        'sample_id': meta[id_col].values,
        'cluster': cluster_labels
    })
    path = os.path.join(out_dir, f"{prefix}cluster_labels.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} cluster labels to {path}")
    return df

def cluster_separation_summary(df_inter: pd.DataFrame, df_intra: pd.DataFrame):
   
    intra_lookup = df_intra.set_index("cluster")["mean_intra_dist"].to_dict()

    rows = []
    for _, row in df_inter.iterrows():
        a, b, inter = row["cluster_a"], row["cluster_b"], row["distance"]
        avg_intra = (intra_lookup.get(a, 0) + intra_lookup.get(b, 0)) / 2
        ratio = inter / avg_intra if avg_intra > 0 else np.nan
        rows.append((a, b, round(inter, 3), round(avg_intra, 3), round(ratio, 3)))

    df_summary = pd.DataFrame(rows, columns=["cluster_a", "cluster_b", "inter_dist", "avg_intra_dist", "separation_ratio"])
    df_summary = df_summary.sort_values("separation_ratio", ascending=False).reset_index(drop=True)
    print("\nCluster separation summary (inter / avg_intra):")
    print(df_summary.to_string(index=False))
    return df_summary
# Main function 
def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "CMD_data" /"binary_healthy_X"
    results_dir = project_root / "results"
    embed_dir = results_dir / "Finetuned_IBD_nested/final_L2H2_test/"
    parser = argparse.ArgumentParser(description="Embedding body-site structure analysis")
    parser.add_argument("--embeddings",  default=str(embed_dir/"test_cls_embeddings.npy"),
                        help="Path to .npy embeddings array (N, D)")
    parser.add_argument("--labels", default=str(data_dir/"y_IBD_test.csv"),
                        help="Path to .csv with body-site labels")
    parser.add_argument("--label_col", default="disease",
                        help="Column name for body-site labels (default: disease)")
    parser.add_argument("--k_min", type=int, default=2,
                        help="Minimum K to evaluate (default: 2)")
    parser.add_argument("--k_max", type=int, default=15,
                        help="Maximum K to evaluate (default: 15)")
    parser.add_argument("--k_fixed", type=int, default=None,
                        help="Skip K selection and use this fixed K")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0,
                        help="t-SNE perplexity (default: 30)")
    parser.add_argument("--out_dir", default=str(embed_dir/"kmeans"),
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, disease_labels, meta = load_data(args.embeddings, args.labels, args.label_col)
   

    if args.k_fixed:
        best_k = args.k_fixed
        print(f"\n[2] Using fixed K={best_k} (skipping selection)")
    else:
        best_k = find_optimal_k(X, args.k_min, args.k_max, args.out_dir)

    km, cluster_labels = fit_kmeans(X, best_k)
    save_cluster_labels(cluster_labels, meta, 'sample_id', args.out_dir, prefix="transformer_")
    Z = run_tsne(X, perplexity=args.tsne_perplexity)
    plot_tsne_comparison(Z, cluster_labels, disease_labels, args.out_dir)
    
    cluster_composition(cluster_labels, disease_labels, args.out_dir)
    pairwise_centroid_distances(km, args.out_dir)
    
    df_intra = intra_cluster_distances(km, X, args.out_dir)
    df_summary = cluster_separation_summary(pairs_df, df_intra)
    

if __name__ == "__main__":
    main()
