import csv
import os
from argparse import ArgumentParser

from evaluation.benchmark import BENCHMARKS


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for benchmark in BENCHMARKS:
        with open(os.path.join(args.out_dir, benchmark.name + ".tsv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            pairs = list(benchmark.load(verbose=False).itertuples(index=False))
            for pair in pairs:
                if not any(" " in w for w in pair[:2]):
                    writer.writerow(map(str.lower, pair[:2]))


if __name__ == "__main__":
    argparser = ArgumentParser(description="Print all benchmark pairs to file")
    argparser.add_argument("-o", "--out-dir", default=".")
    main(argparser.parse_args())
