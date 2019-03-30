import os
from argparse import ArgumentParser

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

from cooccur.io import load_titles
from sib.partition import load_partition


def write_distances(basename, distances):
    filename = basename + "_distances.csv"
    np.savetxt(filename, distances, delimiter=",")
    print("Wrote cluster distance matrix to '%s'..." % filename)


def main(args):
    T = load_partition(args.partition)
    distances = T.cluster_distances()
    basename = args.out_dir + os.path.sep + os.path.basename(os.path.splitext(args.partition)[0])
    if args.distances:
        write_distances(basename, distances)
    print("Fitting multidimensional scaling...")
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(distances).embedding_
    colors = plt.cm.rainbow(np.linspace(0, 1, T.size))
    print("Plotting points...")
    plt.figure(figsize=(16, 10))
    plt.scatter(pos[:, 0], pos[:, 1], c=colors[T.pt_x], s=50, lw=0)
    titles = load_titles(args.titles)
    labels = [titles[x] for x in T.centroids()]
    for label, point in zip(labels, pos):
        plt.annotate(label, point)
    filename = basename + "_plot.png"
    plt.savefig(filename)
    print("Wrote figure to '%s'" % filename)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("partition")
    argparser.add_argument("titles")
    argparser.add_argument("out_dir")
    argparser.add_argument("--distances", action="store_true")
    main(argparser.parse_args())
