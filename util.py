import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv

DEBUG = False
SAMPLE_IDS = {4: 89, 6: 90, 8: 91, 10: 92, 14:93, 18: 94, 24: 95}


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

def load_regular_sample_file(data_folder, time, sparse_matrix=False, csc_matrix=False):
    """
    Load data from a single time step file.

    :param data_folder: path to folder containing data files
    :type data_folder: str

    :param time: hour number of time step to load (e.g. 4, 8, 10)
    :type time: int

    :param sparse_matrix: whether to load data into a sparse matrix or not
    :type sparse_matrix: bool

    :return: matrix of data from a time step where rows are cells and columns
        are genes
    :rtype: np.array or sparse.dok_matrix

    """
    filename = _make_filename(time)

    df = pd.read_csv(data_folder + "/" + filename)

    # Each row now corresponds to a cell (example) and each column to a gene (feature)
    data = df.values[:, 1:].astype(np.float).T
    if not sparse_matrix:
        return data
    if csc_matrix:
        sata = sparse.csc_matrix(data)
        return sata
    # Make data sparse
    sata = sparse.dok_matrix(data)
    return sata

def load_gene_names(data_folder, time):
    # return the gene names for a given time step data file
    filename = _make_filename(time)    
    df = pd.read_csv(f'{data_folder}/{filename}')
    return list(df.values[:,0])

def load_names_and_data(data_folder, time):
    filename = _make_filename(time) 
    df = pd.read_csv(f'{data_folder}/{filename}')

    data = df.values[:, 1:].astype(np.float).T
    names = list(df.values[:,0])
    return names, data

def _make_filename(time):
    # generate the filename for a given time step
    prefix = "GSM30671"
    sample = SAMPLE_IDS[time]
    time = "{:02d}".format(time)
    postfix = "hpf.csv"
    filename = f'{prefix}{sample}_{time}{postfix}'

    return filename

def clean_data(data):
    raise NotImplementedError

def get_data_name(at_index, data_folder, time):
    # GSM3067189_04hpf
    prefix = "GSM30671"
    sample = SAMPLE_IDS[time]
    time = "{:02d}".format(time)
    postfix = "hpf.csv"
    filename = f'{prefix}{sample}_{time}{postfix}'
    csv_file_path = data_folder + "/" + filename
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            return row[at_index]
            break

def create_feature_name_file(data_folder, time):
    prefix = "GSM30671"
    sample = SAMPLE_IDS[time]
    time = "{:02d}".format(time)
    postfix = "hpf.csv"
    filename = f'{prefix}{sample}_{time}{postfix}'
    csv_file_path = data_folder + "/" + filename
    feature_names = {}
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)
        index = -1
        for row in reader:
            feature_names[index] = row[0]
            index += 1
    np.save("hour"+str(time)+"featurenames", feature_names)

def get_feature_name_list(indexes, data_folder, time):
    prefix = "GSM30671"
    sample = SAMPLE_IDS[time]
    time = "{:02d}".format(time)
    postfix = "hpf.csv"
    filename = f'{prefix}{sample}_{time}{postfix}'
    csv_file_path = data_folder + "/" + filename
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            return [str(index) + ":" + str(row[index]) for index in indexes]
            break

def visualize_sample(data):
    """
    Plot the first 2 PCA components of the data
    """
    pca = PCA(n_components=2)
    pca.fit(data)

    expl_var_1, expl_var_2 = pca.explained_variance_ratio_

    plt.scatter(pca.components_[0,:], pca.components_[1,:])
    for i in range(len(data)):
        print(pca.components_[0,i])

    plt.xlabel(f'1st PCA component, {100 * expl_var_1:.1f}% variance')
    plt.ylabel(f'2nd PCA component, {100 * expl_var_2:.1f}% variance')
    plt.title('First 2 PCA components')
    plt.show()

debug()
