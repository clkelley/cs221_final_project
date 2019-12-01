import math
import pickle

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import VarianceThreshold

import util


PP_4 = np.array([5864.0550, 5827.4911, 5775.4534, 5743.1625, \
        5722.3561, 5707.3573, 5695.9119, 5686.7926, 5679.4263, 5673.4688, \
        5668.6231, 5664.6005, 5661.1532, 5658.1497, 5655.4736])
PP_10 = np.array([2178.2337, 2162.2220, 2132.5067, 2106.8206, 2093.3225, \
        2086.3238, 2082.2129, 2079.5606, 2077.7511, 2076.4562, 2075.4865, \
        2074.7354, 2074.1359, 2073.6464, 2073.2410])
PP_24 = np.array([2065.4682, 1832.0351, 1779.6822, 1760.4176, 1754.7068, \
        1751.7157, 1749.4180, 1747.7644, 1746.8025, 1746.1088, 1745.5673, \
        1745.1219, 1744.7408, 1744.4071, 1744.1049])

def save_data():
    for time in util.SAMPLE_IDS.keys():
        data = util.load_regular_sample_file('Data', time, sparse_matrix=True, csc_matrix=True)
        sparse.save_npz(f'{time}hpf_sparse', data)
    # names = util.load_gene_names('Data', 4)
    # with open('gene_names.dat', mode='wb') as f:
    #     pickle.dump(names, f)


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

def cross_validate_lda(gene_names, time_steps, feature_selection=True, tf_idf=False):
    for time in time_steps:
        print(f'TIME STEP: {time}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'Loading file for time {time}...')
        data = sparse.load_npz(f'{time}hpf_sparse.npz')
        if feature_selection:
            print("Performing feature selection with variance threshold")
            sel = VarianceThreshold(threshold=(.99 * (1 - .99)))
            data = sel.fit_transform(data)
        if tf_idf:
            print(data.shape)
            print("Performing tf-idf transformation...")
            tfidf = TfidfTransformer(smooth_idf=True)
            data = tfidf.fit_transform(data)
        print(f'Data size: {data.shape}')

        # k is hyperparameter (number of topics)
        print('Number of topics\tPerplexity')
        for k in range(2, 9):
            perplexity = perform_lda(data, gene_names, k)
            print(f'{k}\t{perplexity}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('')


def evaluate_convergence(gene_names, time, k):
    """
    Evaluate convergence for a given time step and number of topics 
    """
    print(f'Evalutating convergence for t={time} and k={k}...')
    print('====================================================')
    data = sparse.load_npz(f'{time}hpf_sparse.npz')
    lda = LatentDirichletAllocation(n_components=k, max_iter=15, \
            evaluate_every=1, n_jobs=-1, verbose=1)
    lda.fit(data)

def plot_convergence(perplexities, time):
    plt.plot(np.arange(1, 16), perplexities)
    plt.xlabel('Iteration number')
    plt.ylabel('Perplexity')
    plt.xlim([1, 15])
    plt.ylim([perplexities[-1] - 10, perplexities[0] + 50])
    plt.title(f'Convergence of LDA, {time}hpf')
    plt.show()


def tf_idf(data):
    tfidf = TfidfTransfomer(smooth_idf=False)
    new_data = tfidf.fit_transform(data)
    return new_data 
 
def main():
    # load()
    # data = np.load('4hpf.npy', allow_pickle=False)
   
    print('loading gene names ...')
    with open('gene_names.dat', mode='rb') as f:
        gene_names = pickle.load(f)
    
    # evaluate_convergence(gene_names, 10, 7)
    # evaluate_convergence(gene_names, 24, 7)
    # plot_convergence(PP_4, 4)
    # plot_convergence(PP_10, 10)
    # plot_convergence(PP_24, 24)
    # save_data()

    # time_steps = util.SAMPLE_IDS.keys()
    # cross_validate_lda(gene_names, time_steps, tf_idf=True)
    cross_validate_lda(gene_names, [4], feature_selection=True, tf_idf=True)


if __name__ == "__main__":
    main()

