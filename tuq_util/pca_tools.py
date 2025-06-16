import numpy as np



"""
Tools for computing principal components, solving for coefficient scores of 
individual samples, and dimension reduction




"""

def estimateCovarianceEig(data, compute_scores = True):
    # estimate eigenvalues and eigenvectors of the covariance matrix data using SVD
    # data.shape = (N_vars, N_samples)

    # center the data
    mean = np.mean(data, axis=1)

    data_c = (data.T - mean).T

    # compute SVD
    U, s, Vt = np.linalg.svd(data_c, full_matrices=False)

    # eigvals
    eigvals = (s**2)/(data.shape[1] - 1)

    # eigvecs
    eigvecs = U

    scores = None
    if compute_scores:
        scores = computeScores(data_c, mean, eigvecs, centered=True)

    return mean, eigvals, eigvecs, scores


def computeScores(data, mean, eigvecs, centered=False):
    # compute PCA scores
    
    # flag if data is already centered
    if not centered:
        data_c = (data.T - mean).T
    else:
        data_c = data

    scores = np.dot(eigvecs.T, data_c)

    return scores




def computeVarianceFractions(eigvals):

    frac = np.zeros_like(eigvals)

    sum = 0
    for i in range(eigvals.shape[0]):
        sum += eigvals[i]
        frac[i] = sum

    frac /= sum

    return frac