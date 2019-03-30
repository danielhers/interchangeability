import csv
from argparse import ArgumentParser
from collections import Counter
from itertools import chain
from pprint import pprint

from analytics.wn_relations import find_relations, all_relations


def relation_hist(pairs):
    return dict(Counter(chain(*map(find_relations, pairs))))


def main(args):
    with open(args.pairs, encoding="utf-8") as f:
        pairs = list(csv.reader(f))
    pprint(all_relations(pairs) if args.show_all else relation_hist(pairs))


if __name__ == "__main__":
    argparser = ArgumentParser(description="Calculate enrichment of various semantic relations from WordNet in pairs")
    argparser.add_argument("pairs", help="CSV file containing word pairs to look up")
    argparser.add_argument("-a", "--show-all", action="store_true", help="Show all pairs in each relation")
    main(argparser.parse_args())
