import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD, MiniBatchSparsePCA


"""
Different datasets to analyze. Each number corresponds to a different time point (in
hours) after fertilization.
"""
TIME_POINTS = [4, 6, 8, 10, 14, 18, 24]

"""
Load the different matrix decompositions, all with `k` components.
    svd
        the truncated svd, which gives the optimal value among all linear models
    nmf
        a standard nmf, which adds an additional nonnegativity constraint
    sparsepca
        sparse pca, which enforces sparsity through an l-1 regularizer
"""
def get_models(k, lambda1):
    return {
        "svd": TruncatedSVD(n_components=k),
        "nmf": NMF(n_components=k, init="nndsvd", solver="cd"),
        "sparsepca": MiniBatchSparsePCA(n_components=k, alpha=lambda1)
    }

"""
Fit the data with a given matrix decomposition, then return the factors and the loss.
"""
def fit_model(data, model):
    W = model.fit_transform(data)
    H = model.components_
    return W, H, np.linalg.norm(np.dot(W, H) - data) / np.linalg.norm(data)
