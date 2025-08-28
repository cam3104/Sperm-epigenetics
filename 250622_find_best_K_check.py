# usage: python 250622_find_best_K_check.py --input delta_beta_matrix.tsv --output_prefix output --target_cols DMR# context1 context2
################
import os
os.environ["OMP_NUM_THREADS"] = "40"

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def parse_args():
    parser = argparse.ArgumentParser(description="Find best K using silhouette or BIC")
    parser.add_argument("--input", required=True, help="Input Δβ matrix (TSV format)")
    parser.add_argument("--target_cols", nargs='+', required=True, help="Metadata columns to exclude from features")
    parser.add_argument("--output_prefix", required=True, help="Prefix for all output files")
    parser.add_argument("--cluster_method", choices=["gmm", "kmeans"], default="gmm", help="Clustering algorithm")
    parser.add_argument("--min_k", type=int, default=2, help="Minimum K to evaluate")
    parser.add_argument("--max_k", type=int, default=20, help="Maximum K to evaluate")
    return parser.parse_args()


def find_best_k(X, method, min_k, max_k, output_prefix):
    bic_scores = []
    sil_scores = []
    ks = list(range(min_k, max_k + 1))
    for k in ks:
        if method == "gmm":
            model = GaussianMixture(n_components=k, random_state=42)
            model.fit(X)
            bic = model.bic(X)
            labels = model.predict(X)
            sil = silhouette_score(X, labels)
            bic_scores.append(bic)
            sil_scores.append(sil)
            print(f"K={k}, BIC={bic:.3f}, Silhouette={sil:.3f}")
        else:
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)
            sil = silhouette_score(X, labels)
            sil_scores.append(sil)
            print(f"K={k}, Silhouette={sil:.3f}")

    # GMM: best K based on lowest BIC
    if method == "gmm":
        best_k = ks[np.argmin(bic_scores)]
    else:
        best_k = ks[np.argmax(sil_scores)]

    print(f"Best K: {best_k}")

    # Save scores
    score_df = pd.DataFrame({
        "K": ks,
        "BIC": bic_scores if method == "gmm" else [np.nan]*len(ks),
        "Silhouette": sil_scores
    })
    score_df.to_csv(f"{output_prefix}_K_scores.txt", sep="\t", index=False)

    # Plot
    plt.figure(figsize=(6, 4))
    if method == "gmm":
        plt.figure(figsize=(6, 4))
        plt.plot(ks, bic_scores, marker='o')
        plt.gca().invert_yaxis() 
        plt.xlabel("K")
        plt.ylabel("BIC")
        plt.title("BIC Scores for GMM")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_BIC_plot.png", dpi=300)
        plt.close()
        
    plt.plot(ks, sil_scores, marker='s', label="Silhouette")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.title(f"Scores for {method.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_K_scores_plot.png", dpi=300)
    plt.close()

    return best_k



def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")
    ctx_cols = [col for col in df.columns if col not in args.target_cols]
    print(f"Input shape: {df.shape}, Target Columns: {args.target_cols}, Context Columns: {ctx_cols}")
    df[ctx_cols] = df[ctx_cols].fillna(df[ctx_cols].mean())

    X = df[ctx_cols].values

    best_k = find_best_k(X, args.cluster_method, args.min_k, args.max_k, args.output_prefix)
    print(f"Best K found: {best_k}")

 
if __name__ == "__main__":
    main()
