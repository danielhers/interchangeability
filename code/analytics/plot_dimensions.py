import csv
import fnmatch
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

AVERAGE = "Average"
STDEV = "Stdev"
WIDTH = 2


def gen_values(directory):
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, "*.csv"):
            csvfile = os.path.join(root, filename)
            values = {}
            with open(csvfile, encoding="utf-8") as f:
                for line in csv.reader(f):
                    try:
                        i = line[4:].index("") + 4
                        if line[0] in (AVERAGE, STDEV):
                            values[line[0]] = dict(values=np.array(line[4:i], dtype=float),
                                                   weights=np.array(line[i + 1:], dtype=float))
                    except (ValueError, IndexError):
                        continue
            if values:
                yield os.path.splitext(csvfile)[0], values


def main(args):
    t = tqdm(sorted(gen_values(args.directory)), desc="Generating charts", unit="file")
    for path, values in t:
        name = " ".join(path.split(os.sep)[-3::2])
        t.set_postfix(name=name)
        for (suffix, average), (suffix1, stdev) in zip(sorted(values[AVERAGE].items()), sorted(values[STDEV].items())):
            assert suffix == suffix1
            plt.figure()
            plt.title(name + " " + suffix)
            plt.bar(np.arange(len(average)), average, WIDTH, color="blue", label="average")
            plt.scatter(np.arange(len(stdev)), stdev, color="gray", s=1, marker="_", label="standard deviation")
            # plt.scatter(np.arange(len(average)), average, s=stdev/stdev.max())
            # y = np.abs(average).max() * 1.1
            # plt.yticks(np.linspace(-y, y, 10))
            # plt.ylim(-y, y)
            plt.legend()
            plt.savefig(path + "_" + suffix + ".png")
            # plt.show()
            plt.close()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("directory")
    main(argparser.parse_args())
