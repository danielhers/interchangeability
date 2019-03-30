from argparse import ArgumentParser
from collections import namedtuple
from glob import glob
from itertools import product

from evaluation import evaluate_counts

CONFIG_FILE = "evaluate_counts.config"


def main(args):
    print("Loading config from '%s'..." % CONFIG_FILE)
    with open(CONFIG_FILE) as f:
        configs = map(str.split, filter(None, f.readlines()))
    R = "reports"
    E = "exported"
    B = "evaluation/benchmarks/WCRD_with_title.csv"
    c_hist = set()
    for line in configs:
        for c, t in product(*map(glob, line)):
            if c in c_hist:
                continue
            c_hist.add(c)
            print("-- %s %s" % (c, t))
            try:
                evaluate_counts.main(namedtuple("args", "counts titles benchmark out_dir, normalize")(c, t, B, R, False))
            except IOError:
                print("Not found.")


if __name__ == "__main__":
    argparser = ArgumentParser(description="Evaluate all count matrices on concept relatedness benchmark")
    main(argparser.parse_args())
