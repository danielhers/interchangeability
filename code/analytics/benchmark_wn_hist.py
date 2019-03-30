import csv
import os
from argparse import ArgumentParser

from analytics.wn_hist import find_relations
from evaluation.benchmark import BENCHMARKS


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for benchmark in BENCHMARKS:
        pairs = list(benchmark.load().iloc[:, :2].itertuples(index=False))
        # pprint(relation_hist(pairs))
        filename = benchmark.name + "_wordnet_relations.csv"
        with open(os.path.join(args.out_dir, filename), "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(tuple(pair) + tuple(find_relations(pair)) for pair in pairs)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Calculate WordNet relations on all word similarity benchmark pairs")
    argparser.add_argument("--out-dir", default="reports")
    main(argparser.parse_args())
