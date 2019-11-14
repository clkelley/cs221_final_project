import math
import pickle

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_validate

import util


def save_data():
    # for time in util.SAMPLE_IDS.keys():
    #     data = util.load_regular_sample_file('Data', time, sparse_matrix=False)
    #     np.save(f'{time}hpf_dense', data, allow_pickle=False)
    names = util.load_gene_names('Data', 4)
    with open('gene_names.dat', mode='wb') as f:
        pickle.dump(names, f)


def perform_lda(data, gene_names, k):
    """
    Run LDA with a given number of topics k on the time step t
    """
    # perform lda analysis with a given number of topics k
    print(f'Fitting {k} topics...')
    print(f'=================================================================')
    # run with max - 1 cores
    lda = LatentDirichletAllocation(n_components=k, n_jobs=-1)
    # lda.fit(data)

    # perform cross-validation
    # use both log-likelihood as scores
    # use default of 3-fold cv
    scores = cross_validate(lda, data, cv=3, return_estimator=True, n_jobs=-1)
    ll = scores['test_score']
    avg_time = int(scores['fit_time'].mean())
    print(f"Average fit time: {avg_time}")
    # print(f'Cross-validation perplexity: {int(perplexities.mean())} (+/- \
    #         {int(perplexities.std() * 2)})')
    print(f'Accuracy: {int(ll.mean())} (+/- {int(ll.std() * 2)})\n')
    
    estimators = scores['estimator']
    best_lda = estimators[np.argmin(ll)]
    # print(lda.components_.shape)
    top_genes = np.argsort(best_lda.components_, axis=1)
    # print out the top explanatory genes for each topic
    for i in range(k):
        print(f'Topic #{i + 1}:')
        names = [gene_names[j] for j in top_genes[i, -10:]]
        print(', '.join(names))
    # calculate perplexity
    perplexity = best_lda.perplexity(best_lda.components_)
    print(f'Perplexity: {int(perplexity)}\n')

    return perplexity

def main():
    # load()
    # data = np.load('4hpf.npy', allow_pickle=False)
   
    with open('gene_names.dat', mode='rb') as f:
        gene_names = pickle.load(f)
    for time in util.SAMPLE_IDS.keys():
        print(f'TIME STEP: {time}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'Loading file for time {time}...')
        data = np.load(f'{time}hpf_dense.npy', allow_pickle=False)
        print(f'Data size: {data.shape}')

        # k is hyperparameter (number of topics)
        print('Number of topics\tPerplexity')
        for k in range(3, 8):
            perplexity = perform_lda(data, gene_names, k)
            print(f'{k}\t{perplexity}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('')
    # save_data()

if __name__ == "__main__":
    main()


