#usage: python 250624_DMR_dimention_rd_clustering.py --input_file input.tsv --target_cols col1 col2 --prefix output_prefix --output_dir output_dir
# for pca : # --dim_red pca --n_components 2
# for tsne : # --dim_red tsne --perplexity 30
# for umap : # --dim_red umap --n_neighbors 15 --min_dist 0.1
# for clustering : # --cluster_method gmm --n_clusters 3
# for clustering : # --cluster_method kmeans --n_clusters 3
# for clustering : # --cluster_method hdbscan --min_cluster_size 5 --min_samples 5
#################

import os
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['OPENBLAS_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'
import time
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import hdbscan

def parse_args():
    parser = argparse.ArgumentParser(description='Flexible dimension reduction and clustering')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument("--target_cols", nargs='+', required=True, help="Target (non-feature) column names")
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--dim_red', choices=['pca', 'umap', 'tsne'], default='pca')
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--min_dist', type=float, default=0.1)
    parser.add_argument('--perplexity', type=float, default=30.0)

    parser.add_argument('--cluster_method', choices=['gmm', 'kmeans', 'hdbscan'], default='gmm')
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--min_cluster_size', type=int, default=5)
    parser.add_argument('--min_samples', type=int, default=5)

    return parser.parse_args()

def dimension_reduction(X, method, args):
    if method == 'pca':
        reducer = PCA(n_components=args.n_components)
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, n_components=args.n_components, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=args.n_components, perplexity=args.perplexity, random_state=42)
    else:
        raise ValueError('Unsupported dimension reduction method')
    return reducer.fit_transform(X)

def clustering(X, method, args):
    if method == 'gmm':
        model = GaussianMixture(n_components=args.n_clusters, random_state=42)
        return model.fit_predict(X)
    elif method == 'kmeans':
        model = KMeans(n_clusters=args.n_clusters, random_state=42)
        return model.fit_predict(X)
    elif method == 'hdbscan':
        model = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
        return model.fit_predict(X)
    else:
        raise ValueError('Unsupported clustering method')

def visualize(df, x_col, y_col, cluster_col, prefix, output_dir):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_col, y=y_col, hue=cluster_col, palette='tab10', data=df, s=40, alpha=0.8)
    plt.title(f'{x_col} vs {y_col} Clustering')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_{x_col}_{y_col}_cluster.png'), dpi=300)
    plt.close()

def main():
    args = parse_args()
    start = time.time()
    print(f"Starting DMR table dimention reduction -> clustering program :{datetime.now()}")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_file, sep='\t')
    ctx_cols = [c for c in df.columns if c not in args.target_cols]   
    X = df[ctx_cols].values
    print(f"selected dimension reduction method: {args.dim_red}")  
    print(f"Input shape: {X.shape}, Target Columns: {args.target_cols}, Context Columns: {ctx_cols}")
    X_scaled = StandardScaler().fit_transform(X)

    reduced = dimension_reduction(X_scaled, args.dim_red, args)
    df[[f'{args.dim_red.upper()}1', f'{args.dim_red.upper()}2']] = reduced

    print(f"selected cluster method: {args.cluster_method}")
    cluster_labels = clustering(reduced, args.cluster_method, args)
    df['Cluster'] = cluster_labels

    df.to_csv(os.path.join(args.output_dir, f'{args.prefix}_result.tsv'), sep='\t', index=False)
    visualize(df, f'{args.dim_red.upper()}1', f'{args.dim_red.upper()}2', 'Cluster', args.prefix, args.output_dir)
    print(f"Completed in {time.time()-start:.2f}s")
if __name__ == '__main__':
    main()
