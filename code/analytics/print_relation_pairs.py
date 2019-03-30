import os
from argparse import ArgumentParser

import numpy as np

from analytics.compress_relations_file import read_word_enums, get_unique_relations


def main(args):
    word_lists = list(map(list, read_word_enums(args.words)))
    print("Loading relations from '%s'" % args.relations)
    relations = np.load(args.relations)["relations"]
    relation_enum = get_unique_relations(relation_titles=args.relation_titles, out_dir=args.out_dir)
    expected_dim = tuple(map(len, word_lists + [relation_enum]))
    assert relations.shape == expected_dim, "Relation matrix dim does not match word lists and relations: %s != %s" % (
        relations.shape, expected_dim)
    
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        report_filename = os.path.join(args.out_dir, "relation_pairs.txt")
        report_f = open(report_filename, "w", encoding="utf-8")
    else:
        report_f = report_filename = None

    def _print(*a, **kw):
        print(*a, **kw)
        if args.out_dir:
            print(*a, **kw, file=report_f)

    for relation, relation_index in relation_enum.items():
        _print(relation)
        for i, j in zip(*(np.where(np.any(relations, axis=2 + relation_index))[:2])):
            _print(word_lists[0][i], word_lists[1][j])
        _print()

    if args.out_dir:
        report_f.close()
        print("Wrote report to '%s'" % report_filename)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Print all word pairs for each WordNet relation")
    argparser.add_argument("words", nargs=2, help="Text files containing words to use for ordering")
    argparser.add_argument("relations", help=".npz file containing matrix of relations (from compress_relations_file)")
    argparser.add_argument("-r", "--relation-titles", help="Text file with one relation per file to use for the titles")
    argparser.add_argument("-o", "--out-dir", help="Directory to save pairs to")
    main(argparser.parse_args())
