import csv
import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
from scipy.stats import hypergeom

from analytics.compress_relations_file import get_unique_relations, read_word_enums


def gen_files(patterns):
    for pattern in patterns:
        yield from glob(pattern) or [pattern]


def main(args):
    word_lists = list(map(list, read_word_enums(args.words))) if args.words else None

    # Load relations
    print("Loading relations from '%s'" % args.relations)
    relations = np.load(args.relations)["relations"]
    relation_list = list(get_unique_relations(relation_titles=args.relation_titles, out_dir=args.out_dir))

    # Calculate background distribution
    num_pairs = relations.shape[0] * relations.shape[1]
    total_counts = relations.sum(axis=(0, 1))

    # Create report if --out-dir is given
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        pvals_filename = os.path.join(args.out_dir, "pvals.csv")
        pvals_f = open(pvals_filename, "w", encoding="utf-8", newline="")
        pvals_writer = csv.writer(pvals_f)
        counts_filename = os.path.join(args.out_dir, "counts.csv")
        counts_f = open(counts_filename, "w", encoding="utf-8", newline="")
        counts_writer = csv.writer(counts_f)
    else:
        pvals_f = pvals_writer = counts_f = counts_writer = None

    # Get model similarity matrices
    sim_files = sorted(gen_files(args.similarities))
    print("Loading similarity matrices from " + ", ".join(map("'{}'".format, sim_files)))
    sim_names = [os.path.splitext(os.path.basename(f))[0] for f in sim_files]
    sim_names_len = max(map(len, sim_names))
    relation_titles_len = [max(17, len(r)) for r in relation_list]
    num_top = int(args.percent / 100 * relations.shape[0] * relations.shape[1])

    # Print title
    title = ["%-*s" % (sim_names_len, "model")] + ["%-*s" % (l, p) for p, l in zip(relation_list, relation_titles_len)]
    counts_title = ["%-*s" % (sim_names_len, "total")] + \
                   ["%-*s" % (l, p) for p, l in zip(total_counts, relation_titles_len)]
    if args.out_dir:
        pvals_writer.writerow(title)
        counts_writer.writerow(title)
        counts_writer.writerow(counts_title)
    print("Total number of pairs: %d. Getting top %d%% by similarity (%d) for each model" % (
        num_pairs, args.percent, num_top))
    print(*title)
    print(*counts_title)

    # Calculate p-values per model using hypergeometric test TODO also try t-test with actual pairs
    for sim_file, sim_name in zip(sim_files, sim_names):
        print("%-*s" % (sim_names_len, sim_name), end=" ", flush=True)
        sims = np.load(sim_file)["sims"]
        assert sims.shape == relations.shape[:2], "Similarity matrix shape must match relations matrix: %s != %s" % (
            sims.shape, relations.shape)
        # Find indices of top X% pairs by similarity
        top_indices = np.unravel_index(sims.ravel().argsort()[::-1][:num_top], sims.shape)
        # noinspection PyTypeChecker
        top_counts = relations[top_indices].sum(axis=0)
        pvals = [hypergeom.sf(relation_top_count - 1, num_pairs, relation_total_count, num_top)
                 for relation_top_count, relation_total_count in zip(top_counts, total_counts)]
        print(*["%-*s" % (l, "%d (%.3g)" % (c, p)) for p, c, l in zip(pvals, top_counts, relation_titles_len)])
        if args.out_dir:
            pvals_writer.writerow([sim_name] + pvals)
            counts_writer.writerow([sim_name] + list(top_counts))
            if args.words:
                with open(os.path.join(args.out_dir, "%s_pairs.csv" % sim_name), "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows((word_lists[0][i], word_lists[1][j], sims[i, j], relation_list[r])
                                            for i, j in zip(*top_indices) for r in range(len(relation_list))
                                            if relations[i, j, r])
    if args.out_dir:
        pvals_f.close()
        counts_f.close()


if __name__ == "__main__":
    argparser = ArgumentParser(description="Calculate enrichment of various semantic relations from WordNet in"
                                           "nearest neighbor lists of words")
    argparser.add_argument("relations", help=".npz file containing matrix of relations (from compress_relations_file)")
    argparser.add_argument("similarities", nargs="+", help=".npz files containing model sims (from pair_similarities)")
    argparser.add_argument("-r", "--relation-titles", help="Text file with one relation per file to use for the titles")
    argparser.add_argument("-w", "--words", nargs=2, help="Text files containing words to use for ordering")
    argparser.add_argument("-p", "--percent", type=float, default=10, help="Percentage ratio of NNs for foreground dist"
                                                                           " (default: 10)")
    argparser.add_argument("-m", "--model", default="all", help="Word representation model to focus on")
    argparser.add_argument("-o", "--out-dir", help="Directory to save report to")
    main(argparser.parse_args())
