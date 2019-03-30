import csv
import os
from argparse import ArgumentParser
from bisect import bisect
from functools import reduce
from itertools import product, chain, permutations
from operator import mul

from nltk.corpus import wordnet as wn
from tqdm import tqdm

from util.lazylist import List

IDENTITY = "identity"
SYNONYMS = "synonyms"
SYNSET_RELATIONS = ("hypernyms", "hyponyms") + \
                   tuple(map("_".join, product(("member", "part", "substance"), ("holonyms", "meronyms"))))
OTHER_SYNSET_RELATIONS = ("entailments", "instance_hypernyms", "instance_hyponyms",
                          "topic_domains", "region_domains", "usage_domains", "attributes", "causes", "also_sees",
                          "verb_groups", "similar_tos")
LEMMA_RELATIONS = "antonyms", "pertainyms", "derivationally_related_forms"
INDIRECT_PREFIX = "indirect_"
CO_PREFIX = "co_"
DIRECT_RELATIONS = (SYNONYMS,) + SYNSET_RELATIONS + OTHER_SYNSET_RELATIONS + LEMMA_RELATIONS
UNIQUE_RELATIONS = ("identity", "synonyms")

DEPTH = 20


def gen_related(word, relations=None):
    lemmas = wn.lemmas(word)
    if not relations or IDENTITY in relations:
        yield IDENTITY, lemmas
    synsets = wn.synsets(word)
    synonyms = {lemma for synset in synsets for lemma in synset.lemmas()}
    if not relations or SYNONYMS in relations:
        yield SYNONYMS, synonyms
    related_synsets = {}
    related_lemmas = {}

    def _r(s):
        return getattr(s, relation)()
    for relation in SYNSET_RELATIONS + OTHER_SYNSET_RELATIONS:
        if not relations or relation in relations:
            related_synsets[relation] = {synset for syn in synsets for synset in _r(syn)}
            related_lemmas[relation] = {lemma for synset in related_synsets[relation] for lemma in synset.lemmas()}
            yield relation, related_lemmas[relation]
    for relation in SYNSET_RELATIONS:
        if not relations or INDIRECT_PREFIX + relation in relations:
            yield INDIRECT_PREFIX + relation, (lemma for syn in related_synsets[relation]
                                               for synset in syn.closure(_r, DEPTH)
                                               if synset not in related_synsets[relation]
                                               for lemma in synset.lemmas() if lemma not in related_lemmas[relation])
    for relation in LEMMA_RELATIONS:
        if not relations or relation in relations:
            yield relation, (lemma for lemma1 in lemmas for lemma in _r(lemma1))
    for relation, inverse in zip(SYNSET_RELATIONS[::2], SYNSET_RELATIONS[1::2]):
        if not relations or CO_PREFIX + relation in relations:
            related_synsets_co = {synset2 for synset1 in related_synsets[inverse] for synset2 in _r(synset1)}
            yield CO_PREFIX + relation, (lemma for synset2 in related_synsets_co for lemma in synset2.lemmas())


def find_relations(words, by_pair=False):
    pivot, *candidates = words
    candidate_lemmas = [set(wn.lemmas(w)) for w in candidates]
    relations = {}
    for relation, related in gen_related(pivot):
        related_lemmas = List(related)
        for candidate, lemmas in zip(candidates, candidate_lemmas):
            if (pivot, candidate) not in chain(*(relations.get(r, ()) for r in UNIQUE_RELATIONS)) and \
                    any(lemma in lemmas for lemma in related_lemmas):
                relations.setdefault(relation, []).append((pivot, candidate))
    if by_pair:
        pair_relations = {(pivot, c): [] for c in candidates} if by_pair == "all" else {}
        for relation, pairs in relations.items():
            for pair in pairs:
                pair_relations.setdefault(pair, []).append(relation)
        return pair_relations
    else:
        return relations


def all_relations(lists):
    relation_pairs = {}
    for words in lists:
        for relation, pairs in find_relations(words).items():
            relation_pairs.setdefault(relation, []).extend(pairs)
    return relation_pairs


def read_word_lists(filenames):
    word_lists = []
    for words_file in filenames:
        with open(words_file, encoding="utf-8") as f:
            word_lists.append(list(map(str.strip, f.readlines())))
    return word_lists


def main(args):
    word_lists = list(map(sorted, read_word_lists(args.words)))
    os.makedirs(args.out_dir, exist_ok=True)
    relations_file = os.path.join(args.out_dir, "relations.csv")
    t = tqdm((item for l1, l2 in permutations(word_lists, 2) for w in l1
              for item in find_relations([w] + l2[bisect(l2, w):], by_pair="all").items()),
             desc="Finding relations", unit=" pairs", total=reduce(mul, map(len, word_lists), 1))
    found = 0
    with open(relations_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for pair, relations in t:
            t.set_postfix(found=found, pair=pair)
            if relations:
                found += 1
                writer.writerow(pair + tuple(relations))


if __name__ == "__main__":
    argparser = ArgumentParser(description="Find relations for all pairs in cartesian product of two lists")
    argparser.add_argument("words", nargs=2, help="Text files containing words to make pairs out of")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save counts to")
    main(argparser.parse_args())
