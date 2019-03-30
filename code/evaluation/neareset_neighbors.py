import os
import sys
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from cooccur.io import load_sub_counts
from evaluation.evaluate_sentence_relatedness import load_benchmark, extract_all_concepts, sim, \
    DIST_MEASURES, CLUSTER_MEASURES
from sib.joint_distribution import JointDistribution
from sib.misc import normalize
from sib.partition import load_partition
from sib.sequential_information_bottleneck import calc_distances

NUM_NEAREST_NEIGHBORS = 10


def main(args):
    df = load_benchmark(args.benchmark)
    all_concepts = extract_all_concepts(df)
    for i, counts in enumerate(args.counts):
        titles = args.enum or counts.replace("counts", "enum").replace("npz", "csv")
        m, enum = load_sub_counts(counts, titles, all_concepts)
        measures = list(DIST_MEASURES)
        if i == 0 and args.partition:
            T = load_partition(args.partition)
            dist = calc_distances(T, m, args.uniform_prior)
            measures += CLUSTER_MEASURES
        else:
            T = None
            dist = JointDistribution(py_x=normalize(m).T, pxy=m / np.sum(m))
        concepts = [c for c in all_concepts if c in enum]
        id2concept = {enum[c]: c for c in concepts}
        neighbors = np.empty((len(measures), len(id2concept), NUM_NEAREST_NEIGHBORS), dtype=object)
        for j, concept_id in enumerate(tqdm(id2concept, unit=" concepts",
                                            desc="Calculating nearest neighbors", file=sys.stdout)):
            scores = np.array([sim([concept_id], [j], dist, T) for j in id2concept if concept_id != j])
            for n, s in zip(neighbors, scores.T):  # loop over measures
                n[j, :] = [concepts[k] for k in s.argsort()[::-1][:NUM_NEAREST_NEIGHBORS]]
        basename = os.path.join(args.out_dir, os.path.basename(os.path.splitext(counts)[0]))
        for name, n in zip(measures, neighbors):
            report_filename = basename + "_" + name + "_nearest_neighbors.csv"
            np.savetxt(report_filename.translate(str.maketrans(dict.fromkeys("|,()"))),
                       np.hstack((np.array(concepts)[:, None], n)), delimiter=",", fmt="%s")
            print("Wrote report to '%s'" % report_filename)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("counts", nargs="+")
    argparser.add_argument("-e", "--enum", help="Title enumeration input file")
    argparser.add_argument("-p", "--partition", help="Partition to evaluate just for the first 'counts'")
    argparser.add_argument("-b", "--benchmark", default=os.path.join("sentence_similarity", "labeled_triplets_tw.csv"))
    argparser.add_argument("--no-uniform-prior", action="store_false", dest="uniform_prior")
    argparser.add_argument("--out-dir", default="reports")
    main(argparser.parse_args())
