import os
from argparse import ArgumentParser

import spacy
from nltk.corpus import wordnet as wn
from tqdm import tqdm

POS = ("NOUN", "ADJ", "VERB")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    # noinspection PyUnresolvedReferences
    wn_lemmas = {pos: set(tqdm((l.name() for s in wn.all_synsets(getattr(wn, pos)) for l in s.lemmas()),
                               desc="Getting WordNet %ss" % pos.lower(), unit=" lemmas")) for pos in POS}
    with open(args.vocab, encoding="utf-8") as f:
        vocab = list(tqdm(map(str.strip, f), desc="Reading '%s'" % args.vocab, unit=" words"))
    nlp = spacy.load("en", disable=("parser", "ner"))  # requires `python -m spacy download en'
    vocab = [x for x in vocab if x in nlp.vocab]
    lemmas = {}
    for doc in tqdm(nlp.pipe(vocab, batch_size=1024), total=len(vocab), desc="Tagging", unit=" words"):
        for tok in doc:
            lemma = tok.lemma_
            if "-" in lemma:
                lemma = tok.orth_
            pos = tok.pos_
            if lemma in wn_lemmas.get(pos, ()):
                pos_lemmas = lemmas.setdefault(pos, [])
                if lemma not in pos_lemmas:
                    pos_lemmas.append(lemma)
    lemma_sets = {pos: wn_lemmas[pos].union(pos_lemmas) for pos, pos_lemmas in lemmas.items()}
    for pos, pos_lemmas in lemmas.items():
        filename = os.path.join(args.out_dir, pos + ".txt")
        unique_pos_lemmas = [lemma for lemma in pos_lemmas if
                             not any(lemma in pos_lemma_set for other_pos, pos_lemma_set in lemma_sets.items()
                                     if pos != other_pos)]
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(map("{}\n".format, tqdm(unique_pos_lemmas, desc="Writing '%s'" % filename, unit=" words")))


if __name__ == "__main__":
    argparser = ArgumentParser(description="Print all words uniquely of each part of speech using spaCy, by frequency")
    argparser.add_argument("vocab", help="Input filename of vocabulary sorted by frequency")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save file to")
    main(argparser.parse_args())
