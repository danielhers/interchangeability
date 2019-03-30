import json
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from analytics.compress_relations_file import read_word_enums
from analytics.semantic_nearest_neighbors import gen_files
from word_reps_tools.nearest_neighbors_models import NearestNeighborsModels


def main(args):
    word_lists = list(map(list, read_word_enums(args.words)))
    for sim_file in sorted(gen_files(args.similarities)):
        print(os.path.splitext(os.path.basename(sim_file))[0])
        sims = np.load(sim_file)["sims"]
        num_pairs = sims.shape[0] * sims.shape[1]
        indices = np.unravel_index(np.random.choice(num_pairs, args.num_pairs), sims.shape)
        for i, j in zip(*indices):
            print(word_lists[0][i], word_lists[1][j], sims[i, j])
        print()


if __name__ == "__main__":
    argparser = ArgumentParser(description="Print a random set of rows for each model to make sure it makes sense")
    argparser.add_argument("words", nargs=2, help="Text files containing words to use for ordering")
    argparser.add_argument("similarities", nargs="+", help=".npz files containing model sims (from pair_similarities)")
    argparser.add_argument("-n", "--num-pairs", type=int, default=10, help="Number of pairs to print for each model")
    main(argparser.parse_args())
