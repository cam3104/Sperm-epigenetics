#!/usr/bin/env python3
"""
Flexible Δβ clustering with bug fixes, score saving/plots, and parallel k-evaluation.

Usage:
  python3 250621_DMR_clustering_dimention_rd_OPT.py \
    --input <input_matrix.tsv> --output_prefix <prefix> --target_cols DMR# \
    --cluster_method gmm --n_clusters 2,10 --viz_methods pca umap tsne \
    --n_neighbors 20 --min_dist 0.1 --perplexity 30

Notes:
- GMM: selects best k by lowest BIC.
- KMeans: selects best k by highest silhouette score.
- For GMM/KMeans, ALL clustering results in the k-range are saved.
- DBSCAN/HDBSCAN: run once; silhouette reported when valid.
- UMAP uses all CPUs via n_jobs.
- All random_state fixed for reproducibility.
- Parallelization over k via joblib (only when a range is provided).
"""

import os
import argparse
import time
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import umap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import hdbscan
from joblib import Parallel, delayed

RANDOM_STATE = 42 # 재현성을 위해 고정 , UMAP 사용 시 n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism. 문구가 나타남

def adjust_omp_threads(n_jobs: int):
    """Adjusts OMP_NUM_THREADS to prevent oversubscription with joblib."""
    if n_jobs in (-1, 0):
        threads = os.cpu_count()
    else:
        threads = max(1, os.cpu_count() // max(1, n_jobs))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    print(f"OMP_NUM_THREADS set to {threads}")

# -----------------------------
# Argument parsing
# -----------------------------

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Flexible Δβ clustering with parameter range support and parallel k-evaluation")

    # I/O
    parser.add_argument("--input", required=True, help="Input Δβ matrix (TSV) with target cols + feature cols")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files")
    parser.add_argument("--target_cols", nargs='+', required=True, help="Target (non-feature) column names, e.g., DMR#")

    # clustering method
    parser.add_argument("--cluster_method", choices=["gmm", "kmeans", "dbscan", "hdbscan"], default="gmm", help="Clustering algorithm to use.")
    parser.add_argument("--n_clusters", type=str, default=None, help="Number of clusters or a range 'min,max' for GMM/KMeans.")
    parser.add_argument("--eps", type=float, default=1.0, help="DBSCAN eps parameter.")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN/HDBSCAN min_samples parameter.")

    # HDBSCAN-specific
    parser.add_argument("--min_cluster_size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--hdbscan_metric", type=str, default="euclidean", help="HDBSCAN distance metric.")

    # visualization
    parser.add_argument("--viz_methods", nargs='+', choices=["pca", "umap", "tsne"], default=["umap"], help="Visualization methods to run for the best clustering solution.")
    parser.add_argument("--n_neighbors", type=int, default=20, help="UMAP n_neighbors parameter.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist parameter.")
    parser.add_argument("--perplexity", type=float, default=30, help="t-SNE perplexity parameter.")

    # compute
    parser.add_argument("--n_jobs", type=int, default=20, help="Number of parallel jobs for evaluating k (<=0 means all available cores).")

    return parser.parse_args()

# -----------------------------
# Helpers
# -----------------------------

def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Calculate silhouette score safely, returning None if not possible."""
    unique_labels = np.unique(labels)
    n_clusters = len([label for label in unique_labels if label != -1])
    
    if n_clusters >= 2 and len(labels) > n_clusters:
        try:
            return float(silhouette_score(X, labels))
        except ValueError:
            return None
    return None

def plot_scores(xs: List[int], ys: List[float], ylabel: str, out_png: str, title: str):
    """Plots clustering evaluation scores (e.g., BIC, Silhouette) vs. k."""
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_embedding(df: pd.DataFrame, x_col: str, y_col: str, method_name: str, prefix: str, xlabel: str = None, ylabel: str = None):
    """Generates and saves a scatter plot for a given 2D embedding."""
    plt.figure(figsize=(9, 8))
    if "Cluster" in df.columns and (df["Cluster"] == -1).any():
        noise = df[df["Cluster"] == -1]
        core = df[df["Cluster"] != -1]
        
        if not core.empty:
            sns.scatterplot(x=x_col, y=y_col, hue="Cluster", data=core, palette="tab20", s=10, linewidth=0.2, edgecolor="black")
        if not noise.empty:
            plt.scatter(noise[x_col], noise[y_col], s=10, linewidth=0.2, edgecolors="black", facecolors="none", label="noise (-1)")
        
        plt.legend(title="Cluster", bbox_to_anchor=(1.04, 1), loc="upper left")
    else:
        sns.scatterplot(x=x_col, y=y_col, hue="Cluster", data=df, palette="tab20", s=10, linewidth=0.2, edgecolor="black")
        plt.legend(title="Cluster", bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.title(f"{method_name} of Δβ")
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.tight_layout()
    plt.savefig(f"{prefix}_{method_name.lower()}.png", dpi=300)
    plt.close()

def run_viz(df: pd.DataFrame, X_scaled: np.ndarray, methods: List[str], prefix: str, args: argparse.Namespace, feat_cols: List[str]):
    """Runs and plots dimensionality reduction methods."""
    df_vis = df.copy() # Use a copy to avoid modifying the dataframe passed in
    if "pca" in methods:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(X_scaled)
        df_vis[["PC1", "PC2"]] = coords

        loadings = pd.DataFrame(pca.components_.T, index=feat_cols, columns=["PC1_loading", "PC2_loading"]) \
                  .sort_values("PC1_loading", key=lambda s: s.abs(), ascending=False)
        loadings.to_csv(f"{prefix}_pca_loadings.tsv", sep="\t")

        pc1_var = pca.explained_variance_ratio_[0] * 100
        pc2_var = pca.explained_variance_ratio_[1] * 100

        plot_embedding(df_vis, "PC1", "PC2", "PCA", prefix, 
                       xlabel=f"PC1 ({pc1_var:.1f}% explained variance)", 
                       ylabel=f"PC2 ({pc2_var:.1f}% explained variance)")
        
        pca_out = df_vis[args.target_cols + ["PC1", "PC2", "Cluster"]]
        pca_out.to_csv(f"{prefix}_pca_coordinates.tsv", sep="\t", index=False)

    if "umap" in methods:
        n_jobs = os.cpu_count() if args.n_jobs in (-1, 0) else max(1, args.n_jobs)
        reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=RANDOM_STATE, n_jobs=n_jobs)
        coords = reducer.fit_transform(X_scaled)
        df_vis[["UMAP1", "UMAP2"]] = coords
        plot_embedding(df_vis, "UMAP1", "UMAP2", "UMAP", prefix)
        umap_out = df_vis[args.target_cols + ["UMAP1", "UMAP2", "Cluster"]]
        umap_out.to_csv(f"{prefix}_umap_coordinates.tsv", sep="\t", index=False)

    if "tsne" in methods:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=RANDOM_STATE, init='pca', learning_rate='auto')
        coords = tsne.fit_transform(X_scaled)
        df_vis[["TSNE1", "TSNE2"]] = coords
        plot_embedding(df_vis, "TSNE1", "TSNE2", "t-SNE", prefix)
        tsne_out = df_vis[args.target_cols + ["TSNE1", "TSNE2", "Cluster"]]
        tsne_out.to_csv(f"{prefix}_tsne_coordinates.tsv", sep="\t", index=False)

# -----------------------------
# Clustering runners for a given k
# -----------------------------

def run_gmm_for_k(X_scaled: np.ndarray, k: int) -> Dict[str, Any]:
    """Runs Gaussian Mixture Model for a specific k."""
    gm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    labels = gm.fit_predict(X_scaled)
    bic = gm.bic(X_scaled)
    sil = safe_silhouette(X_scaled, labels)
    return {"k": k, "labels": labels, "bic": bic, "silhouette": sil}

def run_kmeans_for_k(X_scaled: np.ndarray, k: int) -> Dict[str, Any]:
    """Runs KMeans for a specific k."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
    labels = km.fit_predict(X_scaled)
    sil = safe_silhouette(X_scaled, labels)
    return {"k": k, "labels": labels, "silhouette": sil}

# -----------------------------
# Main
# -----------------------------

def main():
    """Main execution function."""
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at '{args.input}'")
        return
    
    df = pd.read_csv(args.input, sep="\t")

    missing_cols = [col for col in args.target_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Target columns not found in input file: {', '.join(missing_cols)}")
        return

    adjust_omp_threads(args.n_jobs)
    t0 = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Clustering started.")

    feat_cols = [c for c in df.columns if c not in args.target_cols]
    X = df[feat_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Data shape: {X_scaled.shape}, features: {feat_cols}")

    method = args.cluster_method.lower()
    k_list: List[Optional[int]] = [None]
    if method in ("gmm", "kmeans"):
        if not args.n_clusters:
            raise ValueError(f"{method.upper()} requires --n_clusters (single value or 'min,max' range).")
        if "," in args.n_clusters:
            try:
                a, b = map(int, args.n_clusters.split(","))
                if not (0 < a <= b): raise ValueError
                k_list = list(range(a, b + 1))
            except ValueError:
                raise ValueError("--n_clusters range must be 'a,b' with 1 <= a <= b.")
        else:
            k_list = [int(args.n_clusters)]

    best_k = None
    best_result = None
    n_jobs = os.cpu_count() if args.n_jobs in (-1, 0) else max(1, args.n_jobs)

    if method == "gmm":
        results = Parallel(n_jobs=n_jobs)(delayed(run_gmm_for_k)(X_scaled, k) for k in k_list)
        
        best_result = min(results, key=lambda d: d["bic"])
        best_k = best_result["k"]
        print(f"Best k by BIC: {best_k} (BIC={best_result['bic']:.3f})")
        
        df_scores = pd.DataFrame(results).drop(columns="labels").sort_values("k")
        df_scores.to_csv(f"{args.output_prefix}_gmm_scores.tsv", sep="\t", index=False)
        plot_scores(df_scores["k"].tolist(), df_scores["bic"].tolist(), "BIC (lower is better)", f"{args.output_prefix}_gmm_bic_plot.png", "GMM BIC vs k")

        print("Saving clustering results and visualizations for each k...")
        for result in results:
            k = result["k"]
            prefix_k = f"{args.output_prefix}_k{k}"
            df_k = df.copy()
            df_k["Cluster"] = result["labels"]
            df_k.to_csv(f"{prefix_k}_clustered.tsv", sep="\t", index=False)
            run_viz(df_k, X_scaled, args.viz_methods, prefix_k, args, feat_cols)

    elif method == "kmeans":
        results = Parallel(n_jobs=n_jobs)(delayed(run_kmeans_for_k)(X_scaled, k) for k in k_list)
        
        valid_results = [d for d in results if d["silhouette"] is not None]
        if not valid_results:
            raise RuntimeError("No valid silhouette scores found. Check data or k range.")
        
        best_result = max(valid_results, key=lambda d: d["silhouette"])
        best_k = best_result["k"]
        print(f"Best k by Silhouette: {best_k} (silhouette={best_result['silhouette']:.3f})")
        
        df_scores = pd.DataFrame(results).drop(columns="labels").sort_values("k")
        df_scores.to_csv(f"{args.output_prefix}_kmeans_scores.tsv", sep="\t", index=False)
        plot_scores(df_scores["k"].tolist(), df_scores["silhouette"].tolist(), "Silhouette (higher is better)", f"{args.output_prefix}_kmeans_silhouette_plot.png", "KMeans Silhouette vs k")

        print("Saving clustering results and visualizations for each k...")
        for result in results:
            k = result["k"]
            prefix_k = f"{args.output_prefix}_k{k}"
            df_k = df.copy()
            df_k["Cluster"] = result["labels"]
            df_k.to_csv(f"{prefix_k}_clustered.tsv", sep="\t", index=False)
            run_viz(df_k, X_scaled, args.viz_methods, prefix_k, args, feat_cols)

    elif method == "dbscan":
        db = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=n_jobs)
        labels = db.fit_predict(X_scaled)
        df["Cluster"] = labels
        sil = safe_silhouette(X_scaled, labels)
        print(f"DBSCAN silhouette: {sil:.3f}" if sil is not None else "DBSCAN silhouette: N/A")
        prefix_run = f"{args.output_prefix}_dbscan"
        df.to_csv(f"{prefix_run}_clustered.tsv", sep="\t", index=False)
        run_viz(df, X_scaled, args.viz_methods, prefix_run, args, feat_cols)

    elif method == "hdbscan":
        hdb = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, metric=args.hdbscan_metric)
        labels = hdb.fit_predict(X_scaled)
        df["Cluster"] = labels
        sil = safe_silhouette(X_scaled, labels)
        print(f"HDBSCAN silhouette: {sil:.3f}" if sil is not None else "HDBSCAN silhouette: N/A")
        prefix_run = f"{args.output_prefix}_hdbscan"
        df.to_csv(f"{prefix_run}_clustered.tsv", sep="\t", index=False)
        run_viz(df, X_scaled, args.viz_methods, prefix_run, args, feat_cols)

    with open(f"{args.output_prefix}_runlog.txt", "w") as f:
        f.write("--- Arguments ---\n")
        f.write(json.dumps(vars(args), indent=2))
        f.write("\n\n--- Results ---\n")
        f.write(f"Method: {method}\n")
        if best_k is not None:
            f.write(f"Best_k: {best_k}\n")
        if best_result:
            result_summary = {k: v for k, v in best_result.items() if k != 'labels'}
            f.write(f"Best_result_summary: {json.dumps(result_summary)}\n")
        f.write(f"Elapsed_sec: {time.time() - t0:.2f}\n")

    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
