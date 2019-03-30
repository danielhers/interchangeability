import csv
import os
from argparse import ArgumentParser

import spacy
from scipy.stats import hypergeom
from tqdm import tqdm

from evaluation.benchmark import BENCHMARKS

nlp = spacy.load("en")  # requires `python -m spacy download en'


def pos_set_per_pair(pairs, desc):
    """Get POS for each pair"""
    pos = [set() for _ in range(len(pairs))]
    t = tqdm(((x, i) for i, p in enumerate(pairs) for x in p[:2]), desc=desc, unit=" terms", leave=False)
    for doc, i in nlp.pipe(t, as_tuples=True):
        pos[i].add(doc[0].pos_)
    return pos


def find_related_unrelated_indices(pairs):
    """Find related, unrelated and ignored pairs according to score thresholds of 30% and 70%"""
    scores = [p[-1] for p in pairs]
    low, high = min(scores), max(scores)
    low_threshold = low + .3 * (high - low)
    high_threshold = low + .7 * (high - low)
    related = []
    unrelated = []
    for i, (_, _, s) in enumerate(pairs):
        if s > high_threshold:
            related.append(i)
        elif s < low_threshold:
            unrelated.append(i)
    return related, unrelated


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "benchmark_same_pos.csv"), "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("benchmark", "related", "same_pos_related", "unrelated", "same_pos_unrelated", "pval"))
        for benchmark in BENCHMARKS:
            pairs = list(benchmark.load(verbose=False).itertuples(index=False))
            pos = pos_set_per_pair(pairs, desc=benchmark.name)
            related, unrelated = find_related_unrelated_indices(pairs)
            same_pos_related, same_pos_unrelated = [sum(1 for i in x if len(pos[i]) == 1) for x in (related, unrelated)]
            pval = hypergeom.sf(same_pos_related - 1, len(related) + len(unrelated),
                                same_pos_related + same_pos_unrelated, len(related))
            print("\r%s same-POS pairs: %d/%d (%.1f%%) in related, %d/%d (%.1f%%) in unrelated (p=%.3f)" % (
                benchmark.name,
                same_pos_related, len(related), 100.0 * same_pos_related / len(related),
                same_pos_unrelated, len(unrelated), 100.0 * same_pos_unrelated / len(unrelated),
                pval))
            writer.writerow((benchmark.name, len(related), same_pos_related, len(unrelated), same_pos_unrelated, pval))


if __name__ == "__main__":
    argparser = ArgumentParser(description="Find POS for all word similarity benchmark pairs")
    argparser.add_argument("-o", "--out-dir", default=".")
    main(argparser.parse_args())
