import csv
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


def read_relations(filename):
    with open(filename, encoding="utf-8") as f:
        yield from tqdm(((word1, word2, [relation for relation in relations if relation != "none"])
                         for word1, word2, *relations in csv.reader(f)), desc="Reading '%s'" % filename, unit=" rows")


def read_word_enums(filenames):
    word_enums = []
    for words_file in filenames:
        with open(words_file, encoding="utf-8") as f:
            word_enums.append(dict(map(reversed, enumerate(map(str.strip, f.readlines())))))
    return word_enums


def main(args):
    # Read relations CSV
    relation_enum = get_unique_relations(args.relations, args.relation_titles, args.out_dir)

    # Read word lists
    word_enums = read_word_enums(args.words)
    array = np.zeros((len(word_enums[0]), len(word_enums[1]), len(relation_enum)), dtype=bool)
    for word1, word2, relations in read_relations(args.relations):
        if word2 in word_enums[0]:  # Always word1 is from the first list and word2 from the second, so swap
            word1, word2 = word2, word1
        i = word_enums[0].get(word1)
        j = word_enums[1].get(word2)
        if i is not None and j is not None:
            for relation in relations:
                array[i, j, relation_enum[relation]] = True
    array_file = os.path.join(args.out_dir, "relations.npz")
    np.savez_compressed(array_file, relations=array)
    print("Saved array to '%s'" % array_file)


def get_unique_relations(relations=None, relation_titles=None, out_dir=None):
    if relation_titles:
        print("Loading relation titles from '%s'" % relation_titles)
        with open(relation_titles, encoding="utf-8") as f:
            return dict(map(reversed, enumerate(map(str.strip, f.readlines()))))

    # Not given, so calculate and return
    if not relations:
        return []
    relation_enum = dict(map(reversed, enumerate(sorted(set(relation for _, _, relations in
                                                            read_relations(relations)
                                                            for relation in relations)))))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        relations_file = os.path.join(out_dir, "relations.txt")
        with open(relations_file, "w", encoding="utf-8") as f:
            f.writelines(map("{}\n".format, sorted(relation_enum, key=relation_enum.get)))
        print("Saved unique relations to '%s'" % relations_file)
    return relation_enum


if __name__ == "__main__":
    argparser = ArgumentParser(description="Read CSV file with semantic relations between pairs, and save as matrix")
    argparser.add_argument("words", nargs=2, help="Text files containing words to use for ordering")
    argparser.add_argument("relations", help="CSV file containing word pairs semantic relations (from wn_relations)")
    argparser.add_argument("-r", "--relation-titles", help="Text file with one relation per file to use for the titles")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save compressed file to")
    main(argparser.parse_args())
