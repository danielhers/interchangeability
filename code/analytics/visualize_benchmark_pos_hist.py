import os
from argparse import ArgumentParser
from glob import glob

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLUMNS = ["NOUN", "ADJ", "VERB"]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for pattern in args.hists:
        for hist_file in glob(pattern) or [pattern]:
            title = os.path.basename(hist_file).partition("_")[0]
            df = pd.read_csv(hist_file, delimiter="\t")
            x = range(1, len(df) + 1)
            plt.figure(figsize=(5, 2.5))
            axes = plt.plot(x, df[COLUMNS])
            if title == "VERB":
                plt.xticks(x)
                plt.xlabel("Window size")
            else:
                plt.xticks(x, ())
            plt.title(title + " pivot")
            if "skipgram" in hist_file:
                if title == "NOUN":
                    plt.legend(axes, [c + " neighbors" for c in COLUMNS], loc="best")
            else:
                plt.ylabel("Number of neighbors")
            plt.xlim((min(x), max(x)))
            plt.tight_layout()
            fig_file = os.path.join(args.out_dir, os.path.basename(os.path.splitext(hist_file)[0]) + ".png")
            plt.savefig(fig_file, transparent=True)
            print("Wrote figure to '%s'" % fig_file)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Find POS for all word similarity benchmark pairs")
    argparser.add_argument("hists", nargs="+", help="tsv file, output of benchmark_pos_hist.py")
    argparser.add_argument("-o", "--out-dir", default=".")
    main(argparser.parse_args())
