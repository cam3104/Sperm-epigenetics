# usage: python find_signal_in_DMR.py --dmr_bed <DMR_bed_file> --cyto_count <cytosine_count_file> \
#         --cond1_bwfiles <bw1.bw> <bw2.bw> --cond2_bwfiles <bw3.bw> <bw4.bw> --prefix <prefix> --output_dir <output_file>

import argparse
import pandas as pd
import numpy as np
import pyBigWig
from datetime import datetime
import os
import sys
def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize BigWig signals per DMR and merge with cytosine counts"
    )
    parser.add_argument("--dmr_bed", required=True, help="Input BED file with DMRs")
    parser.add_argument("--cyto_count", required=True, help="TSV file with cytosine counts")
    parser.add_argument("--cond1_bwfiles", nargs="+", required=True, help="List of BigWig files for condition 1")
    parser.add_argument("--cond2_bwfiles", nargs="+", required=True, help="List of BigWig files for condition 2")
    parser.add_argument("--prefix", default="prefix", help="Prefix for DMR names in output")
    parser.add_argument("--output_dir", required=True, help="Output file path for merged results")
    return parser.parse_args()


def summarize_bigwig(bw_path, regions):
    """Compute average signal for each region using pyBigWig."""
    bw = pyBigWig.open(bw_path)
    chrom_lengths = bw.chroms()  # dict: {'chr1': 195471971, ...}

    means = []
    for chrom, start, end in regions:
        # 1. chromosome 존재 여부 확인
        if chrom not in chrom_lengths:
            means.append(np.nan)
            continue

        # 2. 유효 범위로 자르기
        chrom_len = chrom_lengths[chrom]
        start = max(0, start)
        end = min(end, chrom_len)

        # 3. start < end 확인
        if start >= end:
            means.append(np.nan)
            continue

        # 4. 값 가져오기
        try:
            vals = bw.values(chrom, start, end, numpy=True)
            vals = vals[~np.isnan(vals)]
            means.append(vals.mean() if len(vals) > 0 else np.nan)
        except RuntimeError:
            means.append(np.nan)

    bw.close()
    return means

def load_dmr_bed(dmr_bed, cyto_count, bwfiles):
    # 1. Load DMR BED file with peak info (assumes 11 columns)
    bed_cols = ["chrom","start","end","DMR#","significant_CpG","meth_diff",
                "peak_chrom","peak_start","peak_end","peak_name","peak_distance"]
    dmr_df = pd.read_csv(
        dmr_bed, sep="\t", header=None, usecols=range(11),
        names=bed_cols, dtype={"chrom":str,"start":int,"end":int,"DMR#":str,"significant_CpG":str,
                               "meth_diff":float,"peak_chrom":str,"peak_start":int, "peak_end":int,
                               "peak_name":str,"peak_distance":int}
    )
    dmr_df["peak_width"] = dmr_df["peak_end"] - dmr_df["peak_start"]

    # 2. BigWig signal summary for each region
    regions = list(zip(dmr_df["peak_chrom"], dmr_df["peak_start"], dmr_df["peak_end"]))
    
    for bw in bwfiles:
        prefix = bw.rsplit(".",1)[0]
        dmr_df[f"{prefix}_avg"] = summarize_bigwig(bw, regions)
        print(f"[{prefix}] summarized for {len(regions)} regions.")
    

    # 3. Load cytosine count file
    cyto_df = pd.read_csv(
        cyto_count, sep="\t", header=0, usecols=[0,1],
        names=["DMR#","cytosine_count"], dtype={"DMR#":str,"cytosine_count":int}
    )

    # 4. Merge on DMR#
    merged = pd.merge(dmr_df, cyto_df, on="DMR#", how="left")

    return merged


def calculate_signal_fold_change(df, cond1_bwfiles, cond2_bwfiles):
    """Calculate log2 fold change between two condition groups based on average signal."""
    cond1_cols = [bw.rsplit(".",1)[0] + "_avg" for bw in cond1_bwfiles]
    cond2_cols = [bw.rsplit(".",1)[0] + "_avg" for bw in cond2_bwfiles]

    df["cond1_mean"] = df[cond1_cols].mean(axis=1)
    df["cond2_mean"] = df[cond2_cols].mean(axis=1)

    df["log2ratio_diff"] = df["cond2_mean"] - df["cond1_mean"]
    
    return df


def main():
    args = parse_args()
    print(f"\n[START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Merge BigWig file list
    all_bwfiles = args.cond1_bwfiles + args.cond2_bwfiles

    # Load and summarize data
    merged_df = load_dmr_bed(args.dmr_bed, args.cyto_count, all_bwfiles)
    merged_df_path = f"{args.output_dir}/{args.prefix}_signal_merged.txt"

    merged_df.to_csv(merged_df_path, sep="\t", index=False)

    # Calculate log2 fold change
    foldchange_df = calculate_signal_fold_change(
        merged_df, args.cond1_bwfiles, args.cond2_bwfiles
    )
    # filter out rows with NaN or empty values in log2ratio_diff
    foldchange_df = foldchange_df.dropna(subset=["log2ratio_diff"])
    foldchange_df = foldchange_df[foldchange_df["log2ratio_diff"] != 0]
    
    # Select columns for output
    output_cols = ["DMR#", "meth_diff", "cytosine_count", "log2ratio_diff", "peak_width"]
    output_df = foldchange_df[output_cols].copy()
    output_df.to_csv(f"{args.output_dir}/{args.prefix}_signal_foldchange.txt", sep="\t", index=False)

    #apply log1p scaling to peak_width & cytosine_count
    output_df["peak_width"] = np.log1p(output_df["peak_width"])
    output_df["cytosine_count"] = np.log1p(output_df["cytosine_count"])
    output_df.to_csv(f"{args.output_dir}/{args.prefix}_signal_foldchange_scaled.txt", sep="\t", index=False)

    print(f"\n[FINISHED] Output written to: {args.output_dir}")
    print(f"[END] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
