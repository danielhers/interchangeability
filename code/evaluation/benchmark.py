import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CONCEPT_COLUMNS = ["concept 1", "concept 2"]
WORD_COLUMNS = ["word 1", "word 2"]

CLUSTER_MEASURES = [
    "t1=t2",
    "js(y|x1,y|x2)",
    "js(y|t1,y|t2)",
    "js(y|x1,y|t2)+js(y|x2,y|t1)",
    "cosine(js(y|x,y|t))",
    "cosine(y|x1,y|x2)",
    "score",
]


def pearson(x, y):
    return np.corrcoef(x, y)[0, 1]


def spearman(x, y):
    return spearmanr(x, y)[0]


CORR_METHODS = OrderedDict((("pearson", pearson), ("spearman", spearman)))


def load_benchmark(filename, train_only=False, verbose=True):
    if verbose:
        print("Loading benchmark data from '%s'..." % filename)
    df = pd.read_csv(filename)
    return df[df.iloc[:, -1] == "Train"] if train_only else df


def extract_all_concepts(df):
    # noinspection PyTypeChecker
    return set(pd.concat(map(df.get, CONCEPT_COLUMNS)))


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Benchmark:
    def __init__(self, name, path, sep=",", score_column=-1, header=False):
        self.name = name
        self.path = path
        self.sep = sep
        self.score_column = score_column
        self.header = header

    def load(self, verbose=True):
        filename = self.get_filename()
        if verbose:
            print("Loading benchmark data from '%s'..." % filename)
        return self.extract_columns(pd.read_csv(filename, sep=self.sep, skiprows=1 if self.header else 0, header=None))

    def extract_columns(self, df):
        df = df.iloc[:, [0, 1, self.score_column]]
        df.columns = WORD_COLUMNS + ["score"]
        return df

    def get_filename(self):
        return os.path.join(__location__, "benchmarks", self.path)


class ConceptBenchmark(Benchmark):
    def __init__(self, name, path, term_titles):
        super().__init__(name, path)
        self.term_titles = term_titles

    def load(self, verbose=True):
        return self.extract_columns(load_benchmark(self.get_filename(), verbose=verbose)[self.term_titles + ["score"]])


WORD_BENCHMARKS = (
    Benchmark("WordSim353", os.path.join("wordsim353", "results.csv"), header=True),
    Benchmark("WordSim353-S", os.path.join("wordsim353_sim_rel", "wordsim_similarity_goldstandard.txt"), sep='\t'),
    Benchmark("WordSim353-R",  os.path.join("wordsim353_sim_rel", "wordsim_relatedness_goldstandard.txt"), sep='\t'),
    Benchmark("SimLex999", os.path.join("SimLex-999", "SimLex-999.txt"), sep='\t', score_column=3, header=True),
    Benchmark("RW", os.path.join("rw", "rw.txt"), sep='\t', score_column=2),
    Benchmark("MEN", os.path.join("MEN", "MEN_dataset_natural_form_full"), sep=' '),
    Benchmark("MTurk287", "Mtruk.csv"),
    Benchmark("MTurk771", "MTURK-771.csv"),
    Benchmark("SimVerb3500", os.path.join("simverb-3500", "SimVerb-3500.txt"), sep='\t', score_column=3),
)

CONCEPT_BENCHMARKS = (
    ConceptBenchmark("TR9856", "benchmark20170406.csv", ["sourceTerm", "targetTerm"]),  # ACL 2015
    ConceptBenchmark("WORT", "summary.detailed.csv", ["mterm1", "mterm2"]),  # LREC 2018
)

BENCHMARKS = WORD_BENCHMARKS + CONCEPT_BENCHMARKS
