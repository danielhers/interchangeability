import re
from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.benchmark import CLUSTER_MEASURES, CORR_METHODS


def int_field(f):
    try:
        return int(f)
    except ValueError:
        return f


def main(args):
    reports = [r for pattern in args.benchmark_reports for r in glob(pattern)]
    fields = list(zip(*[map(int_field, re.findall(r"([a-zA-Z0-9]+)_.*_(\d+)x(\d+)", r)[0]) for r in reports]))
    corpora, rows, columns = map(sorted, map(set, fields))
    uniform = byfreq = [False, True]
    data = np.zeros((len(corpora), len(rows), len(columns), len(uniform), len(byfreq), len(CLUSTER_MEASURES), len(CORR_METHODS)))
    for report, corpus, row, column in zip(reports, *fields):
        df = pd.read_csv(report)
        for measure in CLUSTER_MEASURES:
            for m, corr in enumerate(CORR_METHODS.values()):
                data[corpora.index(corpus), rows.index(int(row)), columns.index(int(column)),
                     int("uniform" in report), int("byfreq" in report),
                     CLUSTER_MEASURES.index(measure), m] = corr(df["score"], df[measure])
    corpora, rows, columns = map(np.array, (corpora, rows, columns))
    for c, corpus in enumerate(corpora):
        for u, uniform in enumerate(("", "_uniform")):
            for b, byfreq in enumerate(("", "_byfreq")):
                filename = "%s%s%s_correlations.png" % (corpus, uniform, byfreq)
                _, plts = plt.subplots(len(CORR_METHODS), len(columns))
                for m, method in enumerate(CORR_METHODS):
                    for cl, column in enumerate(columns):
                        y = data[c, :, cl, u, b, :, m]
                        x = rows[y.any(axis=1)]
                        y = y[y.any(axis=1)]
                        plts[m][cl].set_title("%s columns, %s correlation" % (column, method))
                        for ms, measure in enumerate(CLUSTER_MEASURES[:-1]):
                            plts[m][cl].plot(x, y[:, ms], label=measure)
                        plts[m][cl].set_xlabel("# rows")
                        plts[m][cl].set_ylabel(method + " correlation")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.gcf().set_size_inches(18.5, 10.5)
                plt.savefig(filename)
                print("Wrote figure to '%s'" % filename)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("benchmark_reports", nargs="+")
    main(argparser.parse_args())
