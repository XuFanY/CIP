import numpy.matlib
import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import rbf_kernel
from numpy import linalg as LA

sglDst = 'D:/data/fs/output/single/'
mulDst = 'D:/data/fs/output/multi/'


def header(cls, k, filename):
    hd = '@relation %s\n' % filename
    for i in range(0, k):
        hd += '\n@attribute feature%s numeric' % i
    hd += '\n@attribute class %s' % cls
    return hd + '\n\n@data\n'


# 将字符矩阵M转换成字符串(元素以,间隔，行以\n间隔)
def mStr(M):
    ms = ''
    n, m = np.shape(M)
    for i in range(n):
        s = M[i][0]
        for j in range(1, m):
            s += ','+M[i][j]
        ms += '\n'+s
    return ms


def spec(F, Lbls, k, filename='xxx', path=sglDst, style=0):
    """
    This function implements the SPEC feature selection

    Input
    -----
    F: {numpy array}, shape (n, m)
        input data
    style: {int}
        style == -1, the first feature ranking function, use all eigenvalues
        style == 0, the second feature ranking function, use all except the 1st eigenvalue
        style >= 2, the third feature ranking function, use the first k except 1st eigenvalue

    Output
    ------
    w_fea: {numpy array}, shape (m,)
        SPEC feature score for each feature

    Reference
    ---------
    Zhao, Zheng and Liu, Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
    """

    n, m = F.shape
    W = rbf_kernel(F, gamma=1)
    if type(W) is numpy.ndarray:
        W = csc_matrix(W)

    # build the degree matrix
    X_sum = np.array(W.sum(axis=1))
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = X_sum[i]

    # build the laplacian matrix
    L = D - W
    d1 = np.power(np.array(W.sum(axis=1)), -0.5)
    d1[np.isinf(d1)] = 0
    d2 = np.power(np.array(W.sum(axis=1)), 0.5)
    v = np.dot(np.diag(d2[:, 0]), np.ones(n))
    v = v/LA.norm(v)

    # build the normalized laplacian matrix
    L_hat = (np.matlib.repmat(d1, 1, n)) * np.array(L) * np.matlib.repmat(np.transpose(d1), n, 1)

    # calculate and construct spectral information
    s, U = np.linalg.eigh(L_hat)
    s = np.flipud(s)
    U = np.fliplr(U)

    # begin to select features
    w_fea = np.ones(m)*1000

    for i in range(m):
        f = F[:, i]
        F_hat = np.dot(np.diag(d2[:, 0]), f)
        l = LA.norm(F_hat)
        if l < 100*np.spacing(1):
            w_fea[i] = 1000
            continue
        else:
            F_hat = F_hat/l
        a = np.array(np.dot(np.transpose(F_hat), U))
        a = np.multiply(a, a)
        a = np.transpose(a)

        # use f'Lf formulation
        if style == -1:
            w_fea[i] = np.sum(a * s)
        # using all eigenvalues except the 1st
        elif style == 0:
            a1 = a[0:n-1]
            w_fea[i] = np.sum(a1 * s[0:n-1])/(1-np.power(np.dot(np.transpose(F_hat), v), 2))
        # use first k except the 1st
        else:
            a1 = a[n-style:n-1]
            w_fea[i] = np.sum(a1 * (2-s[n-style: n-1]))

    if style > 0:
        w_fea[w_fea == 1000] = -1000
    index = np.argsort(w_fea)
    # if style = -1 or 0, ranking features in descending order,
    # the higher the w_fea, the more important the feature is
    # if style != -1 and 0, ranking features in ascending order,
    # the lower the w_fea, the more important the feature is
    if style == -1 or style == 0:
        index = index[::-1]
    index = index[:k]
    X = np.zeros((n, k))
    for i in range(k):
        X.T[i] = F.T[index[i]]
    with open(path + 'spec-%s-%s.csv' % (filename, k), 'w') as wf:
        wf.write(header(k))
        wf.write(mStr(np.c_[X.astype(np.str), Lbls]))
        wf.close()
    return index
