import os
from argparse import ArgumentParser

from tqdm import tqdm

from analytics.semantic_nearest_neighbors import gen_files


def main(args):
    vocabs = []
    t = tqdm(list(gen_files(args.vocabs)), desc="Reading vocabs", unit=" vocabs")
    for vocab_file in t:
        t.set_postfix(file=os.path.basename(vocab_file))
        with open(vocab_file, encoding="utf-8") as f:
            vocabs.append(list(map(str.strip, f.readlines())))
    os.makedirs(args.out_dir, exist_ok=True)
    filtered_file = os.path.join(args.out_dir, "filtered.txt")
    with open(filtered_file, "w", encoding="utf-8") as f:
        f.writelines(tqdm(map("{}\n".format, sorted(set.intersection(*map(set, vocabs)), key=vocabs[0].index)),
                          desc="Writing '%s'" % filtered_file, unit=" rows"))


if __name__ == "__main__":
    argparser = ArgumentParser(description="Find intersection of all vocabulary files and save to file")
    argparser.add_argument("vocabs", nargs="+", help="Vocabulary files to filter")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save file to")
    main(argparser.parse_args())
