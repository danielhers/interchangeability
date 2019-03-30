import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from cooccur.io import load_sub_counts
from evaluation.benchmark import load_benchmark, extract_all_concepts, CONCEPT_COLUMNS
from evaluation.evaluate_partition import print_correlations
from sib.joint_distribution import JointDistribution
from sib.misc import normalize, js

np.warnings.filterwarnings("ignore")

MEASURES = ["pmi1", "pmi2", "dot p(y|x)", "-js p(y|x)", "-eu p(y|x)", "weighted js"]


def main(args):
    df = load_benchmark(args.benchmark, train_only=True)
    num_pairs = len(df)
    m, enum = load_sub_counts(args.counts, args.titles, extract_all_concepts(df))
    if args.normalize:
        m = normalize(m)
    pmi1 = pmi(m)
    pmi2 = pmi(pmi1)
    dist = JointDistribution(py_x=normalize(m).T, pxy=m / np.sum(m))
    empty_scores = np.empty((num_pairs, len(MEASURES)))
    empty_scores.fill(np.nan)
    scores = pd.DataFrame(empty_scores, index=df.index, columns=MEASURES)
    num_pairs_found = 0
    print("Evaluating on benchmark concepts... ", end="")
    for i, source, target in df[CONCEPT_COLUMNS].itertuples():
        try:
            x1, x2 = enum[source], enum[target]
        except KeyError:
            continue
        num_pairs_found += 1
        scores.iloc[i] = (
            pmi1[x1].dot(pmi1[x2]),
            pmi2[x1].dot(pmi2[x2]),
            dist.py_x[:, x1].dot(dist.py_x[:, x2]),
            -js(dist.py_x[:, x1], dist.py_x[:, x2]),
            -euclidean_distance(dist.py_x[:, x1], (dist.py_x[:, x2])),
            -js(dist.py_x[:, x1], dist.py_x[:, x2], dist.px[x1] / (dist.px[x1] + dist.px[x2])),
        )
    print("calculated scores for %f%% of pairs" % (num_pairs_found / num_pairs * 100))
    scores = scores.apply(lambda x: x.fillna(x.mean()), axis=0)  # replace NaN values with column mean
    print_correlations(df, scores)
    basename = args.out_dir + os.path.sep + os.path.basename(os.path.splitext(args.counts)[0])
    report_filename = basename + "_benchmark_report_raw.csv"
    pd.concat([df[CONCEPT_COLUMNS], scores], axis=1).to_csv(report_filename)
    print("Wrote report to '%s'" % report_filename)


def pmi(m):
    m = m.astype(float)
    marginal_word = m.sum(axis=1)
    marginal_context = m.sum(axis=0)
    m /= marginal_word[:, None]  # #(w, c) / #w
    m /= marginal_context  # #(w, c) / (#w * #c)
    m *= marginal_word.sum()  # #(w, c) * D / (#w * #c)
    np.log(m, out=m)  # PMI = log(#(w, c) * D / (#w * #c))
    m.clip(0.0, out=m)  # SPPMI = max(0, log(#(w, c) * D / (#w * #c)) - log(k))
    return m


def euclidean_distance(x, y):
    return np.sqrt(np.dot(x - y, x - y))


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("counts")
    argparser.add_argument("titles")
    argparser.add_argument("benchmark")
    argparser.add_argument("--out-dir", default="reports")
    argparser.add_argument("--normalize", action="store_true")
    main(argparser.parse_args())
