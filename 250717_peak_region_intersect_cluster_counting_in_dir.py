#!/usr/bin/env python3
# usage: python peak_region_intersect_with_expansion.py \
#   --peak peaks.bed \
#   --region regions.bed \
#   --cluster_dir cluster_tsv_directory \
#   --chromsize mm10.chrom.sizes \
#   --output_prefix output \
#   --intersect_cols peak_chrom peak_start peak_end peak_id DMR_chrom DMR_start DMR_end DMR# \
#   --merge_on DMR# \
#   --peak_id_col peak_id \
#   --expand_sizes 1000 2000 5000 \
#   [--no_expand]

import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extend regions, intersect peaks, merge clusters from directory, and visualize."
    )
    parser.add_argument("--peak", required=True, help="Path to peak BED file")
    parser.add_argument("--region", required=True, help="Path to region BED file (e.g., DMRs)")
    parser.add_argument("--cluster_dir", required=True, help="Directory containing cluster TSV files")
    parser.add_argument("--chromsize", required=True, help="Path to chrom.sizes file for genome boundaries")
    parser.add_argument("--output_prefix", required=True, help="Prefix for all output files")
    parser.add_argument(
        "--intersect_cols", nargs='+', required=True,
        help="Column names for bedtools intersect output, e.g. peak_chrom peak_start peak_end peak_id DMR_chrom DMR_start DMR_end DMR#"
    )
    parser.add_argument("--merge_on", default="DMR#", help="Column name to merge on")
    parser.add_argument("--peak_id_col", required=True, help="Column name for peak ID in intersect data")
    parser.add_argument(
        "--expand_sizes", nargs='+', type=int, default=[1000, 2000, 5000],
        help="List of extension sizes (one-sided) in bp, e.g. 1000 2000 5000"
    )
    parser.add_argument(
        "--no_expand", action="store_true",
        help="Do not extend regions; use original region file as is."
    )
    return parser.parse_args()


def load_chromsizes(chromsize_file):
    chromsizes = {}
    with open(chromsize_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            chromsizes[parts[0]] = int(parts[1])
    return chromsizes


def extend_regions(region_file, chromsizes, size, merge_on):
    df = pd.read_csv(region_file, sep='\t', header=None)
    ncol = df.shape[1]
    names = ['chrom', 'start', 'end', merge_on] + [f'col{i}' for i in range(4, ncol)]
    df.columns = names

    def _extend(row):
        orig_len = row['end'] - row['start']
        window = size * 2
        if orig_len > window:
            s, e = row['start'], row['end']
        else:
            s = max(0, row['start'] - size)
            e = min(chromsizes.get(row['chrom'], row['end']), row['end'] + size)
        return pd.Series([row['chrom'], s, e, row[merge_on]])

    df_ext = df.apply(_extend, axis=1)
    df_ext.columns = ['chrom', 'start', 'end', merge_on]

    out_file = f"{args.output_prefix}_region_ext_{size}.bed"
    df_ext.to_csv(out_file, sep='\t', header=False, index=False)
    return out_file


def run_bedtools_intersect(peak_file, region_file, output_file):
    cmd = ['bedtools', 'intersect', '-wa', '-wb', '-a', peak_file, '-b', region_file]
    with open(output_file, 'w') as out:
        res = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Bedtools error:\n{res.stderr}")

def load_and_merge(intersect_file, intersect_cols, cluster_file, merge_on):
    df = pd.read_csv(intersect_file, sep='\t', header=None, names=intersect_cols)
    df[merge_on] = df[merge_on].astype(str)
    df_cluster = pd.read_csv(cluster_file, sep='\t')
    df_cluster[merge_on] = df_cluster[merge_on].astype(str)
    df_merged = df.merge(df_cluster, on=merge_on, how='left')

    if df_merged[merge_on].isna().all():
        raise ValueError(f"No merge_on '{merge_on}' values matched.")
    if 'Cluster' not in df_merged.columns:
        raise ValueError("Column 'Cluster' missing after merge.")
    return df_merged


def plot_peak_counts(df, peak_id_col, prefix):
    counts = (
        df.groupby('Cluster')[peak_id_col]
          .nunique()
          .reset_index(name='peak_count')
          .sort_values('Cluster')
    )
    if counts.empty:
        print(f"Warning: no peaks in any cluster for {prefix}. Skipping plot.")
        return
    counts.to_csv(f"{prefix}_peak_counts.tsv", sep='\t', index=False)
    ax = counts.plot.bar(x='Cluster', y='peak_count', legend=False)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of peaks')
    ax.set_title(f'Peaks per cluster ({prefix})')
    plt.tight_layout()
    plt.savefig(f"{prefix}_peaks_per_cluster.png")
    plt.close()

def main():
    global args
    args = parse_args()
    chromsizes = load_chromsizes(args.chromsize)

    cluster_files = [os.path.join(args.cluster_dir, f) for f in os.listdir(args.cluster_dir) if f.endswith(".tsv")]

    for cluster_file in cluster_files:
        cluster_label = os.path.splitext(os.path.basename(cluster_file))[0]
        if args.no_expand:
            print(f"Processing original regions without extension for {cluster_label}...")
            intersect_out = f"{args.output_prefix}_{cluster_label}_intersect_orig.bed"
            run_bedtools_intersect(args.peak, args.region, intersect_out)
            df_m = load_and_merge(intersect_out, args.intersect_cols, cluster_file, args.merge_on)
            merged_out = f"{args.output_prefix}_{cluster_label}_merged_orig.tsv"
            df_m.to_csv(merged_out, sep='\t', index=False)
            print(f"Saved merged output to {merged_out}")
            plot_peak_counts(df_m, args.peak_id_col, f"{args.output_prefix}_{cluster_label}_orig")
        else:
            for size in args.expand_sizes:
                print(f"\nProcessing extension ±{size} bp for {cluster_label}...")
                region_ext = extend_regions(args.region, chromsizes, size, args.merge_on)
                intersect_out = f"{args.output_prefix}_{cluster_label}_intersect_{size}.bed"
                run_bedtools_intersect(args.peak, region_ext, intersect_out)
                df_m = load_and_merge(intersect_out, args.intersect_cols, cluster_file, args.merge_on)
                merged_out = f"{args.output_prefix}_{cluster_label}_merged_{size}.tsv"
                df_m.to_csv(merged_out, sep='\t', index=False)
                print(f"Saved merged output to {merged_out}")
                plot_peak_counts(df_m, args.peak_id_col, f"{args.output_prefix}_{cluster_label}_{size}")

    print(f"\nAll done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" )

if __name__ == '__main__':
    main()
