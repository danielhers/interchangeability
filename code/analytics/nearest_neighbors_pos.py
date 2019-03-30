import csv
import logging
import os
from argparse import ArgumentParser
from collections import Counter
from glob import glob
from itertools import groupby
from operator import itemgetter

import spacy
from tqdm import tqdm

nlp = spacy.load("en", disable=("parser", "ner"))  # requires `python -m spacy download en'


def read_nearest_neighbors(filename, neighbors_per_term=None):
    """
    Calculate histogram of part of speech per model
    :param filename: .tsv file containing: term1, [term2, score by each model], ...
    :param neighbors_per_term: limit pairs to this many neighbors per term
    :return: dict of model name -> dict of POS -> count
    """
    with open(filename, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader)
        rows = list(reader)
        model_neighbors = {}
        model_num_neighbors = {}
        for term1, *model_term2_score in tqdm(rows, desc="Reading " + filename, unit=" lines"):
            for header, term2, score in zip(headers[1::2], model_term2_score[0::2], model_term2_score[1::2]):
                if term2:
                    model_name = header.rpartition("_term2")[0]
                    num_neighbors = model_num_neighbors.setdefault(term1, {}).setdefault(model_name, 0)
                    if term2 in nlp.vocab:
                        if neighbors_per_term is None or num_neighbors < neighbors_per_term:
                            model_num_neighbors[term1][model_name] = num_neighbors + 1
                            model_neighbors.setdefault(model_name, []).append((float(score), term1, term2))
                    else:
                        logging.debug("'%s' not in vocab" % term2)
        for term1, counts in sorted(model_num_neighbors.items(), key=lambda x: sum(x[1].values())):
            logging.debug("'%s' num neighbors: %s" % (term1, ",".join(map(str, counts.values()))))
        return model_neighbors


def write_hist(hists, filename):
    """
    Write histogram of parts of speech per model
    :param hists: dict of model name -> dict of POS -> count
    :param filename: .tsv file, will contain models as rows and parts of speech as columns
    """
    pos = tuple(sorted(set.union(*(list(map(set, hists.values()))))))
    header = ("model",) + pos
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(tqdm(((model_name,) + tuple(counts.get(p, 0) for p in pos)
                               for model_name, counts in hists.items()), desc="Writing " + filename, unit=" lines"))


def hist_per_term1(neighbors, num_terms):
    hist = Counter()
    for term1, l in groupby(sorted(neighbors, key=itemgetter(1)), key=itemgetter(1)):
        hist[sum(1 for _ in l)] += 1
    hist[0] = num_terms - sum(hist.values())
    return hist


def main(args):
    for pattern in args.nearest_neighbors:
        for filename in glob(pattern) or [pattern]:
            logging.info(filename)
            pos_hists = {}
            neighbors_per_term = args.neighbors if args.exact else None
            model_neighbors = read_nearest_neighbors(filename, neighbors_per_term)
            num_terms = len(set(term1 for neighbors in model_neighbors.values() for _, term1, _ in neighbors))
            hists = []
            for model_name, neighbors in tqdm(sorted(model_neighbors.items()), desc="Analyzing", unit="model"):
                num_neighbors = num_terms * args.neighbors
                sorted_neighbors = sorted(neighbors, key=itemgetter(0), reverse=True)[:num_neighbors]
                logging.info("Using total of %d neighbors for %d terms (average %.1f neighbors per term) for %s" % (
                    len(sorted_neighbors), num_terms, len(sorted_neighbors) / num_terms, model_name))
                hists.append(hist_per_term1(neighbors, num_terms))
                pos_hists[model_name] = Counter(doc[0].pos_ for doc in nlp.pipe(
                    term2 for score, term1, term2 in sorted_neighbors))
            logging.info("\n".join(["Histograms for number of neighbors per term1:"] +
                                   ["\t".join(map(str, [i] + [hist[i] for i in range(max(hist) + 1)]))
                                    for i, hist in enumerate(hists, start=1)]))
            os.makedirs(args.out_dir, exist_ok=True)
            out_file = os.path.join(args.out_dir, os.path.splitext(os.path.basename(filename))[0] + "_pos.tsv")
            write_hist(pos_hists, out_file)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Find part of speech for each nearest neighbor")
    argparser.add_argument("nearest_neighbors", nargs="+", help="tsv file containing term1, [term2 by each model], ...")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save report to")
    argparser.add_argument("-n", "--neighbors", type=int, default=10, help="Number of neighbors per term (or average)")
    argparser.add_argument("-e", "--exact", action="store_true", help="Require exactly this many neighbors per term")
    logging.basicConfig(filename=os.path.splitext(argparser.prog)[0] + ".log", level=logging.INFO)
    main(argparser.parse_args())
