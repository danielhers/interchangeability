import os
from argparse import ArgumentParser
from glob import glob

import matplotlib
import numpy as np

from evaluation.benchmark import BENCHMARKS

matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUM_WINDOW_SIZES = 15


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for pattern in args.correlations:
        for corr_file in glob(pattern) or [pattern]:
            corrs = np.genfromtxt(corr_file, delimiter="\t", dtype=None)[:, :NUM_WINDOW_SIZES]
            x = range(1, corrs.shape[1] + 1)
            plt.figure(figsize=(5, 4))
            axes = plt.plot(x, corrs.T)
            plt.xlim((min(x), max(x)))
            plt.ylim((0.1, 0.8))
            plt.xticks(x)
            plt.xlabel("Window size")
            if "skipgram" in corr_file:
                plt.legend(axes, [b.name for b in BENCHMARKS[:corrs.shape[0]]], loc="lower right")
            else:
                plt.ylabel("Spearman correlation")
            plt.tight_layout()
            fig_file = os.path.join(args.out_dir, os.path.basename(os.path.splitext(corr_file)[0]) + ".png")
            plt.savefig(fig_file, transparent=True)
            print("Wrote figure to '%s'" % fig_file)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Find POS for all word similarity benchmark pairs")
    argparser.add_argument("correlations", nargs="+", help="tsv file, output of benchmark_pos_hist.py")
    argparser.add_argument("-o", "--out-dir", default=".")
    main(argparser.parse_args())
