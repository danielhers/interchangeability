import os
from argparse import ArgumentParser
from glob import glob

from tqdm import tqdm


def main(args):
    out_files = []
    word2f = {}
    for pattern in args.lists:
        for list_file in glob(pattern) or [pattern]:
            out_file = open(os.path.join(os.path.dirname(args.filename),
                                         "_".join((os.path.splitext(os.path.basename(list_file))[0],
                                                   os.path.splitext(os.path.basename(args.filename))[0] + ".tsv"))),
                            "w", encoding="utf-8")
            with open(list_file, encoding="utf-8") as f:
                for line in f:
                    if line:
                        word2f.setdefault(line.strip(), []).append(out_file)
            out_files.append(out_file)
    with open(args.filename, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    with open(args.filename, encoding="utf-8") as f:
        line = f.readline().strip()
        for fh in out_files:
            print(line, file=fh)
        for line in tqdm(f, desc="Splitting '%s'" % args.filename, unit=" lines", total=total - 1):
            for f in word2f["\t".join(line.split("\t")[0:(2 if args.pairs else 1)])]:
                print(line.strip(), file=f)
    for f in out_files:
        f.close()


if __name__ == "__main__":
    argparser = ArgumentParser(description="Split file to separate file based on its first column")
    argparser.add_argument("filename", help="original tsv file")
    argparser.add_argument("lists", nargs="+", help="Lists of files to use as indices")
    argparser.add_argument("--pairs", action="store_true", help="Indices are pairs rather than words")
    main(argparser.parse_args())
