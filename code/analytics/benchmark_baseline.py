import logging
import os
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import spacy

from evaluation.benchmark import WORD_BENCHMARKS

nlp = spacy.load("en", disable=("parser", "ner"))  # requires `python -m spacy download en'


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "baselines.tsv")
    with open(out_file, "w", encoding="utf-8") as f:
        if args.header_column:
            print("benchmark", end="\t", file=f)
        print("majority", "pos", sep="\t", file=f)
        for benchmark in WORD_BENCHMARKS:
            if args.header_column:
                print(benchmark.name, end="\t", file=f)
            pairs = list(benchmark.load(verbose=False).itertuples(index=False))
            terms1, terms2, gold_scores = zip(*pairs)
            low, high = min(gold_scores), max(gold_scores)
            low_threshold = low + .5 * (high - low)
            high_threshold = low + .5 * (high - low)
            bin_scores = np.array([0 if x <= low_threshold else (1 if x > high_threshold else .5) for x in gold_scores],
                                  dtype=float)
            score_counts = Counter(bin_scores)
            majority_baseline = np.array([max(score_counts, key=score_counts.get)] * len(bin_scores), dtype=float)
            pos1, pos2 = [[doc[0].pos for doc in nlp.pipe(t)] for t in (terms1, terms2)]
            pos_baseline = np.array([p1 == p2 for p1, p2 in zip(pos1, pos2)], dtype=float)
            accuracies = [np.mean(baseline == bin_scores) for baseline in (majority_baseline, pos_baseline)]
            print(*accuracies, sep="\t", file=f)
    logging.info("Wrote '%s'" % out_file)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Evaluate majority-based and POS-based baselines on multiple benchmarks")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save report to")
    argparser.add_argument("--header-column", action="store_true", help="Include benchmark as header column")
    logging.basicConfig(filename=os.path.splitext(argparser.prog)[0] + ".log", level=logging.INFO)
    main(argparser.parse_args())
