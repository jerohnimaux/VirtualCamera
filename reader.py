import sys
import os
import numpy as np


def reader(filepath):
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()

    coordinates = []
    with open(filepath) as fp:
        coordinates = fp.read().split()

    coordinates = [float(i) for i in coordinates]
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)
    return coordinates
