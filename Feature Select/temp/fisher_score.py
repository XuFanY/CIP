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


def fisher_score(F, Lbls, k, filename='xxx', path=sglDst):
    """
    This function implements the fisher score feature selection, steps are as follows:
    1. Construct the affinity matrix W in fisher score way
    2. For the r-th feature, we define fr = F(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Fisher score for the r-th feature is score = (fr_hat'*D*fr_hat)/(fr_hat'*L*fr_hat)-1

    Input
    -----
    F: {numpy array}, shape (n, m)
        input data
    Lbls: {numpy array}, shape (n,)
        input class labels

    Output
    ------
    score: {numpy array}, shape (m,)
        fisher score for each feature

    Reference
    ---------
    He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
    Duda, Richard et al. "Pattern classification." John Wiley & Sons, 2012.
    """

    # Construct weight matrix W in a fisherScore way
    W = getW(F, Lbls=Lbls, mode='sup')

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), F)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(F)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, F), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, F), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    lap_score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]

    # compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
    score = 1.0/lap_score - 1
    print(score)

    # the larger the fisher score, the more important the feature is
    index = np.argsort(score)[-1:-k - 1:-1]
    print(index)
    n = np.shape(F)[0]
    X = np.zeros((n, k))
    for i in range(k):
        X.T[i] = F.T[index[i]]
    with open(path + 'fisher-%s-%s.arff' % (filename, k), 'w') as wf:
        wf.write(header(k))
        wf.write(mStr(np.c_[X.astype(np.str), Lbls]))
        wf.close()
    print(np.transpose(score))
    return np.transpose(score)
