import math

from evaluation.document_clustering.accuracy import accuracy
from sib.sequential_document_clustering import get_datasets, load_dataset
from sib.sequential_information_bottleneck import sIB

RESTARTS = 5
SEEDS = 5


def main():
    accs = []
    for dataset in get_datasets("all"):
        print("Using dataset '%s'" % dataset)
        m, labels = load_dataset(dataset)
        dataset_accs = []
        for _ in range(SEEDS):
            logs, T, prm = sIB(m, t_size=labels.shape[1], beta=math.inf, centroid_figures=False, restarts=RESTARTS,
                               uniform_prior=False, calc_i=False, loop_limit=100, seed=None)
            acc = accuracy(T, labels)
            print("accuracy = %.3f" % acc)
            dataset_accs.append(acc)
        accs.append((dataset, sum(dataset_accs) / SEEDS))
    if len(accs) > 1:
        print("\nsummary:\n" + "\n".join("%s: %.3f" % x for x in accs))


if __name__ == "__main__":
    main()
