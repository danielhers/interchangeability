from argparse import ArgumentParser
from collections import namedtuple
from glob import glob
from itertools import product

from analytics import report
from analytics import visualize
from evaluation import evaluate_partition
from util import export

CONFIG_FILE = "evaluate_partition.config"


def main(args):
    print("Loading config from '%s'..." % CONFIG_FILE)
    with open(CONFIG_FILE) as f:
        configs = map(str.split, filter(None, f.readlines()))
    R = "reports"
    E = "exported"
    B = "evaluation/benchmarks/WCRD_with_title.csv"
    p_hist = set()
    for line in configs:
        for c, p, t in product(*map(glob, line)):
            if p in p_hist:
                continue
            p_hist.add(p)
            print("-- %s %s %s" % (c, p, t))
            try:
                u = "uniform" in p
                if args.benchmark:
                    evaluate_partition.main(namedtuple("args", "counts partition titles benchmark uniform_prior "
                                                               "out_dir")(c, p, t, B, u, R))
                if args.report:
                    report.main(namedtuple("args", "counts partition titles out_dir all")(c, p, t, R, False))
                if args.visualize:
                    visualize.main(namedtuple("args", "partition titles out_dir distances")(p, t, R, True))
                if args.export:
                    export.main(namedtuple("args", "counts partition titles benchmark uniform_prior "
                                                   "out_dir")(c, p, t, B, u, E))
            except IOError:
                print("Not found.")


if __name__ == "__main__":
    argparser = ArgumentParser(description="Evaluate all count matrices and clustering partitions"
                                           " on concept relatedness benchmark")
    argparser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    argparser.add_argument("--no-report", dest="report", action="store_false")
    argparser.add_argument("--no-visualize", dest="visualize", action="store_false")
    argparser.add_argument("--no-export", dest="export", action="store_false")
    main(argparser.parse_args())
