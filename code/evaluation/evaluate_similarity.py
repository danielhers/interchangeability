import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from cooccur.io import load_counts, load_enum, load_w2v
from evaluation.benchmark import CORR_METHODS, BENCHMARKS, WORD_COLUMNS
from sib.joint_distribution import JointDistribution
from sib.misc import normalize, js

np.warnings.filterwarnings("ignore")

MEASURES = ["1 - js p(y|x)", "weighted js"]


def main(args):
    # m, enum = load_sub_counts(args.counts, args.titles, set(pd.concat(map(df.get, WORD_COLUMNS))))
    if args.counts.endswith(".bin"):
        vectors, enum = load_w2v(args.counts)
        dist = JointDistribution(py_x=vectors.T)
    else:
        m, enum = load_counts(args.counts), load_enum(args.titles)
        if args.normalize:
            m = normalize(m)
        dist = JointDistribution(py_x=normalize(m).T, pxy=m / np.sum(m))
    measures = MEASURES if args.weighted_js else MEASURES[:-1]
    results = []
    for benchmark in BENCHMARKS:
        basename = args.out_dir + os.path.sep + os.path.basename(os.path.splitext(args.counts)[0])
        results.append(create_report(basename, benchmark, dist, enum, measures, weighted_js=args.weighted_js,
                                     average_missing=args.average_missing))
    print()
    # noinspection PyStringFormat
    print("benchmark            distance measure              %s correlation    %s correlation" % tuple(CORR_METHODS))
    for benchmark, df in zip(BENCHMARKS, results):
        for measure in measures:
            print("%-20s %-30s" % (benchmark.name, measure), end="")
            for corr in CORR_METHODS.values():
                print("%-22s" % "%.3f" % corr(df["score"], df[measure]), end="")
            print()


def create_report(basename, benchmark, dist, enum, measures, weighted_js=False, average_missing=False):
    def _enum_get(word):
        for func in (str, str.lower, str.title, str.upper):
            index = enum.get(func(word))
            if index is not None:
                return index
        return None
    df = benchmark.load(verbose=False)
    num_pairs = len(df)
    empty_scores = np.empty((num_pairs, len(measures)))
    empty_scores.fill(np.nan)
    scores = pd.DataFrame(empty_scores, index=df.index, columns=measures)
    num_pairs_found = 0
    print("Evaluating on %d pairs from the %s benchmark... " % (num_pairs, benchmark.name), end="")
    for i, source, target in df[WORD_COLUMNS].itertuples():
        x1, x2 = [list(filter(None, map(_enum_get, item.split()))) for item in (source, target)]
        if x1 and x2:
            num_pairs_found += 1
            p1, p2 = [dist.py_x[:, x].mean(axis=1) for x in (x1, x2)]
            scores.iloc[i] = (1-js(p1, p2),) + (
                (1-js(p1, p2, dist.px[x1] / (dist.px[x1] + dist.px[x2])),) if weighted_js else ())
    if not num_pairs_found:
        raise RuntimeError("None of the benchmark words was found")
    print("calculated scores for %f%% of pairs" % (num_pairs_found / num_pairs * 100))
    if average_missing:
        scores = scores.apply(lambda x: x.fillna(x.mean()), axis=0)  # replace NaN values with column mean
    report_filename = "%s_%s_report_raw.csv" % (basename, benchmark.name)
    pd.concat([df[WORD_COLUMNS], scores], axis=1).to_csv(report_filename)
    # print("Wrote report to '%s'" % report_filename)
    return pd.concat([df["score"].to_frame(), scores], axis=1).dropna()


if __name__ == "__main__":
    argparser = ArgumentParser(description="Evaluate vector representations on word similarity benchmarks")
    argparser.add_argument("counts")
    argparser.add_argument("titles", nargs="?")
    argparser.add_argument("--out-dir", default="reports")
    argparser.add_argument("--normalize", action="store_true")
    argparser.add_argument("--weighted-js", action="store_true")
    argparser.add_argument("--average-missing", action="store_true")
    main(argparser.parse_args())
