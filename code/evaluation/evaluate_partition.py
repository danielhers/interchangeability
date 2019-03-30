import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy import spatial

from cooccur.io import load_sub_counts
from evaluation.benchmark import load_benchmark, extract_all_concepts, CONCEPT_COLUMNS, CLUSTER_MEASURES, CORR_METHODS
from sib.misc import js
from sib.partition import load_partition
from sib.sequential_information_bottleneck import calc_distances


def main(args):
    T = load_partition(args.partition)
    df = load_benchmark(args.benchmark, train_only=True)
    num_pairs = len(df)
    m, enum = load_sub_counts(args.counts, args.titles, extract_all_concepts(df), len(T.pt_x))
    dist = calc_distances(T, m, args.uniform_prior)
    empty_scores = np.empty((num_pairs, len(CLUSTER_MEASURES)))
    empty_scores.fill(np.nan)
    scores = pd.DataFrame(empty_scores, index=df.index, columns=CLUSTER_MEASURES)
    num_pairs_found = 0
    print("Evaluating on benchmark concepts... ", end="")
    for i, source, target in df[CONCEPT_COLUMNS].itertuples():
        try:
            x1, x2 = enum[source], enum[target]
        except KeyError:
            continue
        num_pairs_found += 1
        t1, t2 = T.pt_x[[x1, x2]]
        scores.iloc[i] = (
            t1 == t2,
            -js(dist.py_x[:, x1], dist.py_x[:, x2]),
            -T.costs[T.pt_x == t1, t2].mean(),
            -(T.costs[x1, t2] + T.costs[x2, t1]) / 2,
            -spatial.distance.cosine(T.costs[x1], T.costs[x2]),
            -spatial.distance.cosine(dist.py_x[:, x1], dist.py_x[:, x2]),
            df["score"][i],
        )
    print("calculated scores for %f%% of pairs" % (num_pairs_found / num_pairs * 100))
    scores = scores.apply(lambda x: x.fillna(x.mean()), axis=0)  # replace NaN values with column mean
    print_correlations(df, scores)
    basename = args.out_dir + os.path.sep + os.path.basename(os.path.splitext(args.partition)[0])
    report_filename = basename + "_benchmark_report.csv"
    pd.concat([df[CONCEPT_COLUMNS], scores], axis=1).to_csv(report_filename)
    print("Wrote report to '%s'" % report_filename)


def print_correlations(benchmark, scores):
    # noinspection PyStringFormat
    print("distance measure              %s correlation    %s correlation" % tuple(CORR_METHODS))
    for measure in scores.columns:
        print("%-30s" % measure, end="")
        for corr in CORR_METHODS.values():
            print("%-22s" % "%.3f" % corr(benchmark["score"], scores[measure]), end="")
        print()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("counts")
    argparser.add_argument("partition")
    argparser.add_argument("titles")
    argparser.add_argument("benchmark")
    argparser.add_argument("--uniform-prior", action="store_true")
    argparser.add_argument("--out-dir", default="reports")
    main(argparser.parse_args())
