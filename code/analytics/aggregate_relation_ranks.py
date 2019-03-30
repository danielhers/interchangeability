import csv
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


def read_relation_ranks(filename):
    """
    Find all ranks for each model for each relation
    :param filename: .tsv file containing: term1, relation, term2, [rank, score], ...
    :return: dict of relation -> dict of model name -> list of ranks
    """
    relations = {}
    with open(filename, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader)
        for term1, relation, term2, *rank_scores in tqdm(reader, desc="Reading " + filename, unit=" lines"):
            for header, rank in zip(headers[3::2], rank_scores[0::2]):
                if rank:
                    relations.setdefault(relation, {}).setdefault(header.rpartition("_rank")[0], []).append(float(rank))
    return relations


def update_mean_ranks_and_counts(model_ranks, relation, relation_mean_ranks, relation_counts):
    relation_mean_ranks[relation] = {model_name: np.mean(ranks) for model_name, ranks in model_ranks.items()}
    relation_counts[relation] = next(iter(len(ranks) for ranks in model_ranks.values()))


def average_ranks(relation_ranks, group=None):
    """
    Average and count ranks per relation per model
    :param relation_ranks: dict of relation -> dict of model name -> list of ranks
    :param group: list of relations to treat as one
    :return: pair of (dict of relation -> dict of model name -> mean rank,
                      dict of relation -> num pairs)
    """
    relation_mean_ranks = {}
    relation_counts = {}
    for relation, model_ranks in relation_ranks.items():
        if relation not in (group or ()):
            update_mean_ranks_and_counts(model_ranks, relation, relation_mean_ranks, relation_counts)
    if group:
        model_ranks = {}
        for relation in group:
            for model_name, ranks in relation_ranks[relation].items():
                model_ranks.setdefault(model_name, []).extend(ranks)
        update_mean_ranks_and_counts(model_ranks, "+".join(group), relation_mean_ranks, relation_counts)
    return relation_mean_ranks, relation_counts


def write_relation_ranks(relation_ranks, relation_counts, filename):
    """
    Write mean relation ranks and counts to file
    :param relation_ranks: dict of relation -> dict of model name -> mean rank
    :param relation_counts: dict of relation -> num pairs
    :param filename: output .tsv file, will contain: relation, count, [mean rank] ...
    """
    header = ("relation", "count") + tuple(next(iter(relation_ranks.values())).keys())
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(tqdm(([relation, relation_counts[relation]] + list(model_ranks.values())
                               for relation, model_ranks in sorted(relation_ranks.items())),
                              desc="Writing " + filename, unit=" lines"))


def main(args):
    relation_ranks = read_relation_ranks(args.relation_ranks)
    mean_ranks, counts = average_ranks(relation_ranks, args.group)
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.relation_ranks))[0] + "_mean.tsv")
    write_relation_ranks(mean_ranks, counts, out_file)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Calculate mean rank in nearest neighbor list for each lexical relation")
    argparser.add_argument("relation_ranks", help=".tsv file containing: term1, relation, term2, [rank, score], ...")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save report to")
    argparser.add_argument("-g", "--group", nargs="+", help="Relations to group together")
    main(argparser.parse_args())
