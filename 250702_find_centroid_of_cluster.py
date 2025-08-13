# usage: python 250702_find_centroid_of_cluster.py input.csv --target_col DMR# --number_cols CCG CXG CHH CpG  --prefix output


import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Plot a heatmap from a CSV file.")
    parser.add_argument("input_file", type=str, help="Path to the CSV file containing the cluster & delta_beta data.")
    parser.add_argument("--target_col", type=str, help="Column name for the target variable (e.g., 'cluster').")
    parser.add_argument("--number_cols", nargs='+', required=True, help="columns to calculate centroids for clusters.")
    parser.add_argument("--prefix", type=str, default="output", help="Output filename .")
    return parser.parse_args()

def plot_heatmap(df, ctx_cols,prefix):
    means = df.groupby("Cluster")[ctx_cols].mean()
    means.to_csv(f"{prefix}_cluster_delta_beta.centroid.txt", sep="\t")
    plt.figure(figsize=(6, 0.5 * len(means) + 1))
    sns.heatmap(means, annot=True, fmt=".2f", cmap="RdBu_r", center=0, cbar_kws={"label": "Δβ"})
    plt.title("Cluster-wise Δβ Means")
    plt.tight_layout()
    plt.savefig(f"{prefix}_cluster_heatmap.png", dpi=300)
    print(f"Heatmap saved as {prefix}_cluster_centroid_heatmap.png")
    plt.close()
    
def main():
    args = parse_args()
    df = pd.read_csv(args.input_file, sep="\t")    
    ctx_cols = args.number_cols
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in the input file.")
    if not all(col in df.columns for col in ctx_cols):
        raise ValueError(f"One or more columns in {ctx_cols} not found in the input file.")
    
    plot_heatmap(df, ctx_cols,args.prefix)
    

if __name__ == "__main__":
    main()