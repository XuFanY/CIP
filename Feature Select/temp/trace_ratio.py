import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances

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


# construct and get matrix W
# tr->trace ratio
def getW(F, k=5, Lbls=None, mode='knn', t=1, tr=False):
    n = np.shape(F)[0]
    if mode == 'knn':
        G = np.zeros((n * (k + 1), 3))
        G[:, 0] = np.tile(np.arange(n), (k + 1, 1)).reshape(-1)
        # trace ratio(laplacian)
        if tr:
            # compute matrix D
            D = pairwise_distances(F)
            D **= 2
            # compute G[:, 2]
            idx = np.sort(D, axis=1)[:, 0:k + 1]
            dump_heat_kernel = np.exp(-idx / (2 * t * t))
            G[:, 2] = np.ravel(dump_heat_kernel, order='F')
        # lap score
        else:
            # compute matrix D
            F_normalized = np.power(np.sum(F * F, axis=1), 0.5)
            for i in range(n):
                F[i, :] = F[i, :] / max(1e-12, F_normalized[i])
            D = -np.dot(F, np.transpose(F))
            # compute G[:, 2]
            G[:, 2] = 1
        # compute G[:, 1]
        argidx = np.argsort(D, axis=1)[:, 0:k + 1]
        G[:, 1] = np.ravel(argidx, order='F')
        # build the sparse affinity matrix W
        W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n, n))
        bigger = np.transpose(W) > W
        W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
        return W
    # mode->sup(supervised)
    else:
        # fisher score & trace ratio(fisher score)
        # get true labels and the number of classes
        label = np.unique(Lbls)
        n_classes = np.unique(Lbls).size
        # construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
        W = lil_matrix((n, n))
        for i in range(n_classes):
            class_idx = (Lbls == label[i])
            class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
            W[class_idx_all] = 1.0 / np.sum(np.sum(class_idx))
        return W


# style->{fisher, laplacian}
def trace_ratio(F, Lbls, k, filename='xxx', path=sglDst, style='fisher'):
    """
    This function implements the trace ratio criterion for feature selection

    Input
    -----
    F: {numpy array}, shape (n, m)
        input data
    numLbls: {numpy array}, shape (n,)
        input class labels
    k: {int}
        number of features to select
    style: {string}
        style == 'fisher', build between-class matrix and within-class affinity matrix in a fisher score way
        style == 'laplacian', build between-class matrix and within-class affinity matrix in a laplacian score way

    Output
    ------
    feature_idx: {numpy array}, shape (m,)
        the ranked (descending order) feature index based on subset-level score
    feature_score: {numpy array}, shape (m,)
        the feature-level score
    subset_score: {float}
        the subset-level score

    Reference
    ---------
    Feiping Nie et al. "Trace Ratio Criterion for Feature Selection." AAAI 2008.
    """

    n, m = F.shape
    if style == 'fisher':
        # build within class and between class laplacian matrix L_w and L_b
        W_within = getW(F, Lbls=Lbls, mode='sup')
        L_within = np.eye(n) - W_within
        L_tmp = np.eye(n) - np.ones([n, n]) / n
        L_between = L_within - L_tmp
    else:
        # build within class and between class laplacian matrix L_w and L_b
        W_within = getW(F, tr=True)
        D_within = np.diag(np.array(W_within.sum(1))[:, 0])
        L_within = D_within - W_within
        W_between = np.dot(np.dot(D_within, np.ones([n, n])), D_within) / np.sum(D_within)
        D_between = np.diag(np.array(W_between.sum(1)))
        L_between = D_between - W_between

    # build F'*L_within*F and F'*L_between*F
    L_within = (np.transpose(L_within) + L_within)/2
    L_between = (np.transpose(L_between) + L_between)/2
    S_within = np.array(np.dot(np.dot(np.transpose(F), L_within), F))
    S_between = np.array(np.dot(np.dot(np.transpose(F), L_between), F))

    # reflect the within-class or local affinity relationship encoded on graph, Sw = F*Lw*F'
    S_within = (np.transpose(S_within) + S_within)/2
    # reflect the between-class or global affinity relationship encoded on graph, Sb = F*Lb*F'
    S_between = (np.transpose(S_between) + S_between)/2

    # take the absolute values of diagonal
    s_within = np.absolute(S_within.diagonal())
    s_between = np.absolute(S_between.diagonal())
    s_between[s_between == 0] = 1e-14  # this number if from authors' code

    # preprocessing
    fs_idx = np.argsort(np.divide(s_between, s_within), 0)[::-1]
    temK = np.sum(s_between[0:k])/np.sum(s_within[0:k])
    s_within = s_within[fs_idx[0:k]]
    s_between = s_between[fs_idx[0:k]]

    # iterate util converge
    count = 0
    while True:
        score = np.sort(s_between-temK*s_within)[::-1]
        I = np.argsort(s_between-temK*s_within)[::-1]
        idx = I[0:k]
        old_k = temK
        temK = np.sum(s_between[idx])/np.sum(s_within[idx])
        count += 1
        if abs(temK - old_k) < 1e-3:
            break

    # get feature index, feature-level score and subset-level score
    feature_idx = fs_idx[I]
    feature_score = score
    subset_score = temK

    index = fs_idx
    X = np.zeros((n, k))
    for i in range(k):
        X.T[i] = F.T[index[i]]
    with open(path + 'traceRatio-%s-%s.csv' % (filename, k), 'w') as wf:
        wf.write(header(k))
        wf.write(mStr(np.c_[X.astype(np.str), Lbls]))
        wf.close()
    return feature_idx, feature_score, subset_score
