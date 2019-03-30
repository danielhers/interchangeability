import numpy as np


def accuracy(T, labels):
    """
    Classify according to the majority label on each cluster according to T.
    Evaluate accuracy of classification with respect to labels.
    :param T: Partition object with pt_x attribute representing the clustering
    :param labels: one-hot matrix of correct labels for each x
    :return accuracy
    """
    counts = [labels[T.pt_x == t].sum(axis=0) for t in range(T.size)]
    majority = np.argmax(counts, axis=-1).squeeze()
    return np.mean(majority[T.pt_x] == labels.nonzero()[1])
