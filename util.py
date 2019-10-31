import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DEBUG = True

def debug():
    if DEBUG:
        data = load_regular_sample_file("GSE112294_RAW",6)
        print(data[:10])

def baseline():
    filename = './data/GSM3067191_08hpf.csv'  # 8hpf Zebrafish embryos
    df = pd.read_csv(filename)

    # Each row now corresponds to a cell (example) and each column to a gene (feature)
    data = df.values[:, 1:].astype(np.float).T
    # Make data sparse
    sata = sparse.dok_matrix(data)

    k = 50
    lsvec, svals, rsvect = la.svd(data)
    dnorm = sla.norm(sata)
    approx = lsvec.dot(np.diag(svals)).dot(rsvect)
    print("SVD reconstruction error:", la.norm(sata - approx) / dnorm)

    avgs = np.sum(data[:], axis=0) / data.shape[0]


    plt.plot(avgs)
    plt.show()

    expression_counts = np.sum(sata, axis=0)
    best = np.array(np.argsort(-expression_counts))[0]
    common = best[:k]
    uncommon = best[k:]
    common_norm = la.norm(data[:, common])
    print("Baseline reconstruction error:", np.sqrt(dnorm**2 - common_norm** 2) / dnorm)

def load_regular_sample_file(data_folder, time, sparse=False):
    #GSM3067189_04hpf
    prefix = "GSM"
    sample = int ((time - 4)/2)
    sample_id = 3067189 + sample
    time = "{:02d}".format(time)
    postfix = "hpf.csv"
    filename = prefix + str(sample_id) +  "_" + time + postfix

    df = pd.read_csv(data_folder + "/" + filename)

    # Each row now corresponds to a cell (example) and each column to a gene (feature)
    data = df.values[:, 1:].astype(np.float).T
    if not sparse:
        return data
    # Make data sparse
    sata = sparse.dok_matrix(data)
    return sata


debug()
        
        
        
    
