import csv
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy import spatial
from tqdm import tqdm

from cooccur.io import load_sub_counts
from sib.joint_distribution import JointDistribution
from sib.misc import js, normalize
from sib.partition import load_partition
from sib.sequential_information_bottleneck import calc_distances

np.seterr("raise")

CONCEPT_COLUMNS = ["pivot", "sentence1", "sentence2"]
DIST_MEASURES = [
    "cosine(y|x1,y|x2)",
    "js(y|x1,y|x2)",
]
CLUSTER_MEASURES = [
    "cosine(t|x1,t|x2)",
    "js(t|x1,t|x2)",
]
GOLD_MEASURES = [
    "confidence",
]


def load_benchmark(filename):
    print("Loading sentence relatedness benchmark data from '%s'..." % filename)
    return pd.read_csv(filename)


def load_idf(filename, enum):
    idf = np.ones(max(enum.values()) + 1)
    if filename:
        with open(filename, encoding="utf-8") as f:
            for concept, i in tqdm(csv.reader(f), unit=" concepts", desc="Reading concept IDFs", file=sys.stdout):
                x = enum.get(concept)
                if i != "0.0":
                    if x is None:
                        x = enum[concept] = max(enum.values()) + 1
                        idf.resize(x + 1)
                    idf[x] = float(i)
    return idf


def split(cs):
    try:
        return cs.split("|")
    except AttributeError:  # nan due to empty cell
        return []


def extract_all_concepts(df):
    return set(c for cs in pd.concat(map(df.get, CONCEPT_COLUMNS)) for c in split(cs))


def accuracy(x, y):
    return np.mean(np.sign(x) == np.sign(y))


def print_evaluation(gold, scores):
    # noinspection PyStringFormat
    print("distance measure              accuracy (%)")
    for measure in scores.columns:
        print("%-30s" % measure, end="")
        print("%-22s" % "%.3f" % (100 * accuracy(gold, scores[measure])))


def sim(x1, x2, dist, T=None, idf=None):
    """
    Calculate similarities between two sets of concepts
    :param x1: one set of concept IDs
    :param x2: another set of concept IDs
    :param dist: JointDistribution object to use for co-occurrence-based measures
    :param T: Partition object to use for clustering-based measures (optional)
    :param idf: array of IDF per concept (by ID)
    :return: array of similarity values
    """
    # average p(y|x) over sentence concepts, weighted by IDF
    p1, p2 = [dist.py_x[:, np.clip(x, None, dist.py_x.shape[1] - 1)].dot(
        np.ones(len(x)) if idf is None else 100-idf[x]) / len(x)
        for x in (x1, x2)]
    assert len(p1) and len(p2), "Empty representation"
    ret = (
        -spatial.distance.cosine(p1, p2),
        -js(p1, p2),
    )
    if T:
        ret += (
            -spatial.distance.cosine(p1, p2),
            -js(p1, p2),
        )
    return ret


def main(args):
    df = load_benchmark(args.benchmark)
    num_triplets = len(df)
    all_concepts = extract_all_concepts(df)
    print("Total %d triplets in benchmark, containing %d concepts" % (num_triplets, len(all_concepts)))
    for i, counts in enumerate(args.counts):
        titles = args.enum or counts.replace("counts", "enum").replace("npz", "csv")
        m, enum = load_sub_counts(counts, titles, all_concepts)
        idf = load_idf(args.idf, enum) if args.idf else None
        measures = list(DIST_MEASURES)
        if i == 0 and args.partition:
            T = load_partition(args.partition)
            dist = calc_distances(T, m, args.uniform_prior)
            measures += CLUSTER_MEASURES
        else:
            T = None
            dist = JointDistribution(py_x=normalize(m).T, pxy=m / np.sum(m))
        measures += GOLD_MEASURES
        empty_scores = np.empty((num_triplets, len(measures)))
        empty_scores.fill(np.nan)
        gold = df[GOLD_MEASURES[0]]
        scores = pd.DataFrame(empty_scores, index=df.index, columns=measures)
        num_concepts = num_concepts_found = 0
        dist.py_x = np.append(dist.py_x, np.ones((dist.py_x.shape[0], 1)) / dist.py_x.shape[0], axis=1)
        for j, *sentences in tqdm(df[CONCEPT_COLUMNS].itertuples(), unit=" triplets", total=num_triplets,
                                  desc="Evaluating on benchmark sentence triplets", file=sys.stdout):
            concepts = list(map(split, sentences))  # three lists of strings
            num_concepts += sum(map(len, concepts))
            x = [np.array([enum.get(c, -1) for c in cs] or [-1], dtype=int) for cs in concepts]
            if args.drop_missing:
                x = [np.array([xii for xii in xi if xii != -1] or [-1], dtype=int) for xi in x]
            num_concepts_found += sum(map(len, x))
            measure_scores = [s1 - s2 for s1, s2 in zip(*[sim(x1, x2, dist, T, idf) for x1, x2 in (x[0:2], x[0:3:2])])]
            scores.iloc[j] = measure_scores + [gold[j]]
        print("Calculated scores for %.1f%% of concepts" % (num_concepts_found / num_concepts * 100))
        scores = scores.apply(lambda v: v.fillna(v.mean()), axis=0)  # replace NaN with column mean
        print_evaluation(gold, scores)
        basename = os.path.join(args.out_dir, os.path.basename(os.path.splitext(counts)[0]))
        report_filename = basename + "_sentence_relatedness_benchmark_report.csv"
        pd.concat([df[CONCEPT_COLUMNS], scores], axis=1).to_csv(report_filename)
        print("Wrote report to '%s'" % report_filename)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("counts", nargs="+")
    argparser.add_argument("-e", "--enum", help="Title enumeration input file")
    argparser.add_argument("-p", "--partition", help="Partition to evaluate just for the first 'counts'")
    argparser.add_argument("-b", "--benchmark", default=os.path.join("sentence_similarity", "labeled_triplets_tw.csv"))
    argparser.add_argument("-i", "--idf", help="concept IDFs")
    argparser.add_argument("--no-uniform-prior", action="store_false", dest="uniform_prior")
    argparser.add_argument("--out-dir", default="reports")
    argparser.add_argument("--drop-missing", action="store_true", help="omit concepts without counts from calculation")
    main(argparser.parse_args())
