import csv
from argparse import ArgumentParser

import numpy as np

from cooccur.io import load_titles
from sib.partition import load_partition


def main(args):
    T = load_partition(args.partition)
    distances = T.cluster_distances()
    titles = load_titles(args.titles)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        headers = ("cluster-id", "centroid", "size", "score") + tuple(map(str, range(T.size)))
        writer.writerow(headers)
        for t, centroid in enumerate(T.centroids()):
            print("\rCluster %d..." % t, end="", flush=True)
            inds_t = list(np.where(T.pt_x == t)[0])
            score = T.costs[inds_t, t].mean()
            writer.writerow((t, titles[centroid], len(inds_t), score) + tuple(distances[t]))
    print("\rWrote report to '%s'" % args.out)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("partition")
    argparser.add_argument("titles")
    argparser.add_argument("out")
    main(argparser.parse_args())
