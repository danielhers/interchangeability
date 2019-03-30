import csv
import logging
import os
from argparse import ArgumentParser
from glob import glob

from tqdm import tqdm

from evaluation.benchmark import WORD_BENCHMARKS, spearman


def read_similarities(filename):
    """
    Calculate vector of similarities per model
    :param filename: .tsv file containing: term1, term2, [score by each model], ...
    :return: dict of model name -> array of similarities
    """
    with open(filename, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader)
        rows = list(reader)
        model_scores = {}
        for term1, term2, *scores in tqdm(rows, desc="Reading " + filename, unit=" lines"):
            for header, score in zip(headers[2:], scores):
                if score:
                    model_name = header.rpartition("_score")[0]
                    try:
                        model_scores.setdefault(model_name, {})[(term1, term2)] = float(score)
                    except ValueError as e:
                        raise IOError("Invalid line in %s: %s" % (filename, "\t".join((term1, term2, score)))) from e
        return model_scores


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for pattern in args.filenames:
        out_file = os.path.join(args.out_dir, os.path.splitext(os.path.basename(pattern))[0] + "_eval.tsv")
        with open(out_file, "w", encoding="utf-8") as f:
            for benchmark in WORD_BENCHMARKS:
                if args.header_column:
                    print(benchmark.name, end="\t", file=f)
                pairs = list(benchmark.load(verbose=False).itertuples(index=False))
                _, _, gold_scores = zip(*pairs)
                benchmark_pattern = os.path.join(os.path.dirname(pattern),
                                                 "_".join((benchmark.name, os.path.basename(pattern))))
                for filename in glob(benchmark_pattern) or [benchmark_pattern]:
                    logging.info(filename)
                    model_scores = read_similarities(filename)
                    for model_name, scores in tqdm(sorted(model_scores.items()), desc="Analyzing", unit="model"):
                        pred_scores = [scores.get(p[:2], 0) for p in pairs]
                        print(spearman(pred_scores, gold_scores), end="\t", file=f)
                print(file=f)
        logging.info("Wrote '%s'" % out_file)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Evaluate similarity scores for multiple models on multiple benchmarks")
    argparser.add_argument("filenames", nargs="+", help="tsv file containing term1, term2, [score by each model], ...")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save report to")
    argparser.add_argument("--header-column", action="store_true", help="Include benchmark as header column")
    logging.basicConfig(filename=os.path.splitext(argparser.prog)[0] + ".log", level=logging.INFO)
    main(argparser.parse_args())
