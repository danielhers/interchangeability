import csv
import os
from argparse import ArgumentParser

import numpy as np

from cooccur.io import load_titles, load_counts
from sib.joint_distribution import create_distribution
from sib.partition import load_partition


def write_report(filename, dist, T, titles, all_distances):
    sizes = np.zeros(T.size)
    scores = np.zeros(T.size)
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        headers = ("cluster-id", "size", "concept-id", "concept", "score", "concept-score", "concept-prior")
        if all_distances:
            headers += tuple(map(str, range(T.size)))
        writer.writerow(headers)
        for t in range(T.size):
            inds_t = list(np.where(T.pt_x == t)[0])
            sizes[t] = len(inds_t)
            scores[t] = T.costs[inds_t, t].mean()
            for x in inds_t:
                fields = (t, sizes[t], x, titles[x], scores[t], T.costs[x, t], dist.px[x])
                if all_distances:
                    fields += tuple(T.costs[x])
                writer.writerow(fields)
    print("Wrote report to '%s'" % filename)
    return sizes, scores


def write_figure(filename, sizes, scores):
    import matplotlib.pyplot as plt
    width = .5
    x = sizes.argsort()[::-1]
    x_ = np.arange(len(sizes))
    fig, ax1 = plt.subplots(figsize=(len(sizes) / 6, 6))
    ax1.bar(x_ - width / 2, sizes[x], color="b", width=width)
    ax1.set_xlabel("cluster id")
    ax1.set_ylabel("cluster size", color="b")
    ax1.tick_params("y", colors="b")
    plt.xticks(x_, map(str, x), size="x-small", rotation=90)
    ax2 = ax1.twinx()
    ax2.bar(x_ + width / 2, scores[x], color="g", width=width)
    ax2.set_ylabel("average JS to centroid", color="g")
    ax2.tick_params("y", colors="g")
    fig.tight_layout()
    plt.xlim(0, len(sizes) - 1)
    plt.savefig(filename)
    print("Wrote figure to '%s'" % filename)


def main(args):
    basename = args.out_dir + os.path.sep + os.path.basename(os.path.splitext(args.partition)[0])
    if args.all:
        basename += "_all"
    sizes, scores = write_report(basename + "_report.csv", create_distribution(load_counts(args.counts)),
                                 load_partition(args.partition), load_titles(args.titles), args.all)
    write_figure(basename + ".png", sizes, scores)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("counts")
    argparser.add_argument("partition")
    argparser.add_argument("titles")
    argparser.add_argument("out_dir")
    argparser.add_argument("--all", action="store_true")
    main(argparser.parse_args())
