import math

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import util


def load():
    data = util.load_regular_sample_file('Data', 4, sparse_matrix=True)
    print(data.shape)
    np.save('04hpf', data)


def main():
    load()

if __name__ == "__main__":
    main()


