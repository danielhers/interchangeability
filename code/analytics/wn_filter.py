import os
from argparse import ArgumentParser

import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def main(args):
    with open(args.vocab, encoding="utf-8") as f:
        vocab = list(map(str.strip, f.readlines()))
    filtered = list(filter(wn.lemmas, tqdm(vocab, desc="Querying WordNet", unit=" words")))
    assert filtered, "No word was found"
    print("Found %d words in WordNet" % len(filtered))
    filtered_file = os.path.join(args.out_dir, "filtered.txt")
    with open(filtered_file, "w", encoding="utf-8") as f:
        f.writelines(map("{}\n".format, filtered))
    print("Wrote '%s'" % filtered_file)
    sample = sorted(np.random.choice(filtered, args.sample, replace=False), key=filtered.index)
    print("Sampled %d words as pivots" % len(sample))
    sample_file = os.path.join(args.out_dir, "sample.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.writelines(map("{}\n".format, sample))
    print("Wrote '%s'" % sample_file)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Read vocabulary file and keep only words in WordNet; also sample subset")
    argparser.add_argument("vocab", help="Vocabulary file to load")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save files to")
    argparser.add_argument("-s", "--sample", type=int, default=1000, help="Size of sample")
    main(argparser.parse_args())
