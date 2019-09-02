import numpy as np
from numpy.linalg import svd
from untwist.factorizations import NMF


"""
NMF related functions:
- Initialize with SVD
- Compute decomposition
- Extract a component via soft mask
"""

def init_nmf(mat):
    u,s,vh = svd(mat, full_matrices = False)
    s_sum = np.sum(s)
    d_sum = 0
    i = 0
    curve = []
    while d_sum / s_sum < 0.9:
        d_sum += s[i]
        i += 1
        curve.append(d_sum / s_sum)
    S = np.diag(s)
    W0 = np.abs(u[:,:i])
    H0 = np.abs(np.dot(S[:i,:],vh))
    return i, W0, H0

## one source
def get_mask(i, W, H):
    eps = np.spacing(1)
    V = np.dot(W,H)
    V1 = np.dot(W[:,i].reshape(-1,1),H[i,:].reshape(1,-1))
    return V1/(V+eps)

#multiple sources
def get_mask1(i, W, H):
    eps = np.spacing(1)
    V = np.dot(W,H)
    V1 = np.dot(W[:,i],H[i,:])
    return V1/(V+eps)

# vanilla nmf
def compute_nmf(mat, rank, W0 = None, H0 = None):
    nmf = NMF(rank, iterations = 100, return_divergence=True, update_func='kl')
    W, H, err = nmf.process(mat, W0, H0)
    return W, H

def extract_component(X, W, H, num):
    V = np.dot(W, H)
    V1 = np.dot(W[:,num, np.newaxis], H[np.newaxis,num,:])
    mask = V1/V
    est = mask * X
    return est
