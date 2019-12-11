import numpy as np
import numpy.linalg as la
from sklearn.decomposition import NMF, TruncatedSVD, MiniBatchSparsePCA


""" Different datasets to analyze. Each number corresponds to a different time point (in
hours) after fertilization.
"""
TIME_POINTS = [4, 6, 8, 10, 14, 18, 24]


def get_models(k, lambda1):
    """ Load the different matrix decompositions, all with `k` components.

        svd
            the truncated svd, which gives the optimal value among all linear models
        nmf
            a standard nmf, which adds an additional nonnegativity constraint
        sparsepca
            sparse pca, which enforces sparsity through an l-1 regularizer
    """
    return {
        "svd": TruncatedSVD(n_components=k),
        "nmf": NMF(n_components=k, init="nndsvd", solver="cd"),
        # "sparsepca": MiniBatchSparsePCA(n_components=k, alpha=lambda1)
    }



def fit_model(data, model):
    """ Fit the data with a given matrix decomposition, then return the factors and the
    loss.
    """
    W = model.fit_transform(data)
    H = model.components_
    return W, H, np.linalg.norm(np.dot(W, H) - data) / np.linalg.norm(data)


def score_factor_pairing(H1, H2):
    """ Score the similarity between to sets of factors."""
    n_components, n_features = H1.shape
    order = pair_factors(H1, H2)
    H1_perm = H1[order, :]  # permute H2 according to pairing

    losses = la.norm(H1_perm - H2, axis=1) / np.sqrt(2)
    return np.mean(losses)


def pair_factors(H1, H2):
    """ Pair two sets of factors, H1 and H2, via greedy matching."""
    assert H1.shape == H2.shape  # verify shapes match up
    n_components, n_features = H1.shape

    assignments = []
    factors = [(k, H2[k, :]) for k in range(n_components)]
    for k in range(n_components):
        scores = [la.norm(H1[k, :] - h) for (k, h) in factors]
        best = np.argmin(scores)  # find closest factor

        assignments.append(factors[best][0])  # pair factors
        factors.pop(best) # remove from candidate factors

    return assignments



def clean_factors(W, H, debug=False):
    """ Organize factors such that (1) the norm of each row of H is 1 and (2) the columns of
    W / rows of H are ordered by descending norm.

    Return the new factors
    """
    # normalize rows of H
    scaling_matrix = np.diag(la.norm(H, axis=1))
    W_scaled = np.dot(W, scaling_matrix)
    H_scaled = np.dot(la.inv(scaling_matrix), H)

    if (debug):  # check correctness
        assert np.allclose(np.dot(W_scaled, H_scaled), np.dot(W, H))

    # permute columns of W to be in order of descending norm
    order = np.argsort(-la.norm(W_scaled, axis=0))
    W_perm = W_scaled[:, order]
    H_perm = H_scaled[order, :]

    return W_perm, H_perm
