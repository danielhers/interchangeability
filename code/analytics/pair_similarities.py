import json
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from analytics.compress_relations_file import read_word_enums
from word_reps_tools.nearest_neighbors_models import NearestNeighborsModels


def main(args):
    word_enums = read_word_enums(args.words)
    with open(args.configuration, encoding="utf-8") as f:
        models = NearestNeighborsModels(json.load(f)['configuration'])
    array = {model: np.zeros((len(word_enums[0]), len(word_enums[1])), dtype=float) for model in models.models}
    for word1, i in tqdm(list(word_enums[0].items()), desc="Calculating similarities", unit=" words"):
        for result in models.read_nn_all(word1):
            model_array = array[result["model"]]
            for word2, sim in result["nearestNeighbors"]:
                j = word_enums[1].get(word2)
                if j is not None:
                    model_array[i, j] = sim
    for model in models.models:
        os.makedirs(args.out_dir, exist_ok=True)
        array_file = os.path.join(args.out_dir, os.path.basename(model) + ".npz")
        np.savez_compressed(array_file, sims=array[model])
        print("Saved array to '%s'" % array_file)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Calculate matrix of similarities for all pairs in cartesian product of two "
                                           "lists")
    argparser.add_argument("words", nargs=2, help="Text files containing words to use for ordering")
    argparser.add_argument("configuration", help="JSON configuration for word representation models")
    argparser.add_argument("-o", "--out-dir", default=".", help="Directory to save compressed matrix file to")
    main(argparser.parse_args())
