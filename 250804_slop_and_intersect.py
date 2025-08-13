#!/usr/bin/env python3
import argparse
import subprocess
import pandas as pd
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extend regions, sort BEDs, intersect or find closest peaks, and visualize."
    )
    parser.add_argument("--peak", required=True, help="Path to peak BED file")
    parser.add_argument("--region", required=True, help="Path to region BED file (e.g., DMRs)")
    parser.add_argument("--chromsize", required=True, help="Path to chrom.sizes file for genome boundaries")
    parser.add_argument("--output_prefix", required=True, help="Prefix for all output files")
    parser.add_argument(
        "--expand_sizes", nargs='+', type=int, default=[1000, 2000, 5000],
        help="List of extension sizes (one-sided) in bp"
    )
    parser.add_argument(
        "--no_expand", action="store_true",
        help="Do not extend regions; use original region file"
    )
    parser.add_argument(
        "--closest", action="store_true",
        help="For each region, report only the nearest peak within given distances"
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


def extend_regions(region_file, chromsizes, size, output_prefix):
    df = pd.read_csv(region_file, sep='\t', header=None)
    ncol = df.shape[1]
    names = ['chrom', 'start', 'end'] + [f'col{i}' for i in range(3, ncol)]
    df.columns = names

    def _extend(row):
        if row['chrom'] not in chromsizes:
            return pd.Series([row['chrom'], row['start'], row['end']] + list(row[3:]))
        orig_len = row['end'] - row['start']
        window = size * 2
        if orig_len > window:
            s, e = row['start'], row['end']
        else:
            s = max(0, row['start'] - size)
            e = min(chromsizes[row['chrom']], row['end'] + size)
        return pd.Series([row['chrom'], s, e] + list(row[3:]))

    df_ext = df.apply(_extend, axis=1)
    df_ext.columns = ['chrom', 'start', 'end'] + list(df.columns[3:])
    out_file = f"{output_prefix}_region_ext_{size}.bed"
    df_ext.to_csv(out_file, sep='\t', header=False, index=False)
    return out_file


def run_bedtools_sort(input_bed, output_bed):
    cmd = ['bedtools', 'sort', '-i', input_bed]
    with open(output_bed, 'w') as out:
        res = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Bedtools sort error:\nCommand: {' '.join(cmd)}\nError: {res.stderr}")


def run_bedtools_intersect(peak_file, region_file, output_file):
    cmd = ['bedtools', 'intersect', '-wa', '-wb', '-a', peak_file, '-b', region_file]
    with open(output_file, 'w') as out:
        res = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Bedtools intersect error:\nCommand: {' '.join(cmd)}\nError: {res.stderr}")


def run_bedtools_closest(peak_file, region_file, output_file, max_dist):
    cmd = ['bedtools', 'closest', '-a', region_file, '-b', peak_file, '-d', '-t', 'first']
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Bedtools closest error:\nCommand: {' '.join(cmd)}\nError: {res.stderr}")
    with open(output_file, 'w') as out:
        for line in res.stdout.splitlines():
            dist = int(line.split()[-1])
            if dist <= max_dist:
                out.write(line + "\n")


def main():
    args = parse_args()
    chromsizes = load_chromsizes(args.chromsize)

    # 1) sort peak file
    sorted_peak = f"{args.output_prefix}_peak_sorted.bed"
    print(f"\nSorting peak file -> {sorted_peak}")
    run_bedtools_sort(args.peak, sorted_peak)

    # --closest mode: sort region and find nearest peaks within each size
    if args.closest:
        sorted_region = f"{args.output_prefix}_region_sorted.bed"
        print(f"Sorting region file -> {sorted_region}")
        run_bedtools_sort(args.region, sorted_region)
        for size in args.expand_sizes:
            out = f"{args.output_prefix}_closest_{size}.bed"
            print(f"\nFinding nearest peaks within ±{size} bp -> {out}")
            run_bedtools_closest(sorted_peak, sorted_region, out, max_dist=size)
        return

    # existing extend + intersect pipeline
    region_files = []
    if args.no_expand:
        sorted_region = f"{args.output_prefix}_region_sorted.bed"
        print(f"Sorting region file -> {sorted_region}")
        run_bedtools_sort(args.region, sorted_region)
        region_files.append((sorted_region, 'orig'))
    else:
        for size in args.expand_sizes:
            print(f"\nProcessing extension ±{size} bp...")
            fn = extend_regions(args.region, chromsizes, size, args.output_prefix)
            sorted_fn = f"{args.output_prefix}_region_ext_{size}.sorted.bed"
            print(f"Sorting extended regions -> {sorted_fn}")
            run_bedtools_sort(fn, sorted_fn)
            region_files.append((sorted_fn, str(size)))

    for region_file, tag in region_files:
        out = f"{args.output_prefix}_intersect_{tag}.bed"
        print(f"...running bedtools intersect -> {out}")
        run_bedtools_intersect(sorted_peak, region_file, out)

    print(f"\nAll done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
