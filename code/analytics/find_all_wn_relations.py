import csv
import os
from argparse import ArgumentParser

from tqdm import tqdm

from analytics.wn_relations import gen_related, DIRECT_RELATIONS


def main(args):
    with open(args.words, encoding="utf-8") as f:
        words = list(map(str.strip, f.readlines()))
    relations_file = args.out_file or os.path.splitext(args.words)[0] + "_relations.csv"
    t = tqdm(((w, r.rstrip("s"), rw) for w in words for r, ls in gen_related(w, args.relations)
              for rw in sorted(set([l.name() for l in ls
                                    if l.synset().pos() in args.pos and "_" not in l.name()]) - {w})),
             desc="Writing '%s'" % relations_file, unit=" pairs")
    with open(relations_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for word, relation, related_word in t:
            t.set_postfix(word=word, relation=relation, related=related_word)
            writer.writerow((word, relation, related_word))


if __name__ == "__main__":
    argparser = ArgumentParser(description="Find relations for all pairs in cartesian product of two lists")
    argparser.add_argument("words", help="Text file containing words to find relations for")
    argparser.add_argument("-o", "--out-file", help="File to save relations to")
    argparser.add_argument("-r", "--relations", nargs="*", choices=DIRECT_RELATIONS, default=DIRECT_RELATIONS,
                           help="Relations to extract")
    argparser.add_argument("-p", "--pos", nargs="+", choices="navr", default=["n"],
                           help="Parts of speech of related words to include")
    main(argparser.parse_args())
