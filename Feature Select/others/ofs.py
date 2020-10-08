import numpy as np
import numpy.matlib
from numpy import linalg as LA
from scipy.sparse import *
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import pairwise_distances


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
            # D = euclidean_distances(F)
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
        cls = np.unique(Lbls)
        c = cls.size
        # construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
        W = lil_matrix((n, n))
        for i in range(c):
            class_idx = (Lbls == cls[i])
            class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
            W[class_idx_all] = 1.0 / np.sum(np.sum(class_idx))
        return W


# score(str){fisher,laplace}
# 测试多标签数据集时参数score不能为fisher!!!
def fisher_laplace(F, ks, score='fisher', Lbls=None):
    # construct the affinity matrix W
    if score == 'fisher':
        W = getW(F, Lbls=Lbls, mode='sup')
    else:
        W = getW(F)
    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), F)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(F)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, F), 0) - np.multiply(tmp, tmp) / D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, F), 0) - np.multiply(tmp, tmp) / D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    # compute laplacian score for all features
    flscore = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]
    indexs = []
    if score == 'fisher':
        flscore = 1.0 / flscore - 1
        # the larger the fisher score, the more important the feature is
        idx = np.argsort(-flscore)
        for k in ks:
            indexs.append(idx[:k])
    else:
        # the smaller the laplacian score is, the more important the feature is
        idx = np.argsort(flscore)
        for k in ks:
            indexs.append(idx[:k])
    return indexs


# 无法测试多标签数据集
def reliefF(F, ks, Lbls, nbr=5):
    """
    This function implements the reliefF feature selection

    Input
    -----
    F: {numpy array}, shape (n, m)
        input data
    Lbls: {numpy array}, shape (n,)
        input class labels
    nbr: {int}
        choices for the number of neighbors (default nbr = 5)

    Output
    ------
    score: {numpy array}, shape (m,)
        reliefF score for each feature

    Reference
    ---------
    Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
    """
    n, m = F.shape
    # calculate pairwise distances between instances
    # distance = manhattan_distances(F)
    distance = pairwise_distances(F, metric='manhattan')
    score = np.zeros(m)
    # the number of sampled instances is equal to the number of total instances
    for idx in range(n):
        near_hit = []
        near_miss = dict()
        self_fea = F[idx, :]
        c = np.unique(Lbls).tolist()
        stop_dict = dict()
        for label in c:
            stop_dict[label] = 0
        del c[c.index(Lbls[idx])]
        p_dict = dict()
        p_label_idx = float(len(Lbls[Lbls == Lbls[idx]]))/float(n)
        for label in c:
            p_label_c = float(len(Lbls[Lbls == label]))/float(n)
            p_dict[label] = p_label_c/(1-p_label_idx)
            near_miss[label] = []
        distance_sort = []
        distance[idx, idx] = np.max(distance[idx, :])
        for i in range(n):
            distance_sort.append([distance[idx, i], int(i), Lbls[i]])
        distance_sort.sort(key=lambda x: x[0])
        for i in range(n):
            # find k nearest hit points
            if distance_sort[i][2] == Lbls[idx]:
                if len(near_hit) < nbr:
                    near_hit.append(distance_sort[i][1])
                elif len(near_hit) == nbr:
                    stop_dict[Lbls[idx]] = 1
            else:
                # find k nearest miss points for each label
                if len(near_miss[distance_sort[i][2]]) < nbr:
                    near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                else:
                    if len(near_miss[distance_sort[i][2]]) == nbr:
                        stop_dict[distance_sort[i][2]] = 1
            stop = True
            for (key, value) in stop_dict.items():
                    if value != 1:
                        stop = False
            if stop:
                break
        # update reliefF score
        near_hit_term = np.zeros(m)
        for ele in near_hit:
            near_hit_term = np.array(abs(self_fea-F[ele, :]))+np.array(near_hit_term)
        near_miss_term = dict()
        for (label, miss_list) in near_miss.items():
            near_miss_term[label] = np.zeros(m)
            for ele in miss_list:
                near_miss_term[label] = np.array(abs(self_fea-F[ele, :]))+np.array(near_miss_term[label])
            score += near_miss_term[label]/(nbr*p_dict[label])
        score -= near_hit_term/nbr
    indexs = []
    # the higher the reliefF score, the more important the feature is
    idx = np.argsort(-score)
    for k in ks:
        indexs.append(idx[:k])
    return indexs


def spec(F, ks, style=0):
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
    score: {numpy array}, shape (m,)
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
    score = np.ones(m)*1000
    for i in range(m):
        f = F[:, i]
        F_hat = np.dot(np.diag(d2[:, 0]), f)
        l = LA.norm(F_hat)
        if l < 100*np.spacing(1):
            score[i] = 1000
            continue
        else:
            F_hat = F_hat/l
        a = np.array(np.dot(np.transpose(F_hat), U))
        a = np.multiply(a, a)
        a = np.transpose(a)
        # use f'Lf formulation
        if style == -1:
            score[i] = np.sum(a * s)
        # using all eigenvalues except the 1st
        elif style == 0:
            a1 = a[0:n-1]
            score[i] = np.sum(a1 * s[0:n-1])/(1-np.power(np.dot(np.transpose(F_hat), v), 2))
        # use first k except the 1st
        else:
            a1 = a[n-style:n-1]
            score[i] = np.sum(a1 * (2-s[n-style: n-1]))
    if style > 0:
        score[score == 1000] = -1000
    indexs = []
    # if style = -1 or 0, ranking features in descending order,
    # the higher the score, the more important the feature is
    if style == -1 or style == 0:
        idx = np.argsort(-score)
        for k in ks:
            indexs.append(idx[:k])
    # if style != -1 and 0, ranking features in ascending order,
    # the lower the score, the more important the feature is
    else:
        idx = np.argsort(score)
        for k in ks:
            indexs.append(idx[:k])
    return indexs


def startTr(F, style='fisher', Lbls=None):
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
    L_within = (np.transpose(L_within) + L_within) / 2
    L_between = (np.transpose(L_between) + L_between) / 2
    S_within = np.array(np.dot(np.dot(np.transpose(F), L_within), F))
    S_between = np.array(np.dot(np.dot(np.transpose(F), L_between), F))
    # reflect the within-class or local affinity relationship encoded on graph, Sw = F*Lw*F'
    S_within = (np.transpose(S_within) + S_within) / 2
    # reflect the between-class or global affinity relationship encoded on graph, Sb = F*Lb*F'
    S_between = (np.transpose(S_between) + S_between) / 2
    # take the absolute values of diagonal
    s_within = np.absolute(S_within.diagonal())
    s_between = np.absolute(S_between.diagonal())
    s_between[s_between == 0] = 1e-14  # this number if from authors' code
    return s_between, s_within


# style(str){fisher, laplacian}
# 测试多标签数据集时参数style不能为fisher!!!
def trace_ratio(k, s_between, s_within):
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
    # preprocessing
    fs_idx = np.argsort(np.divide(s_between, s_within), 0)[::-1]
    temK = np.sum(s_between[0:k])/np.sum(s_within[0:k])
    s_within = s_within[fs_idx[0:k]]
    s_between = s_between[fs_idx[0:k]]
    # iterate util converge
    count = 0
    while count < 1e3:
        I = np.argsort(s_between-temK*s_within)[::-1]
        idx = I[0:k]
        old_k = temK
        temK = np.sum(s_between[idx])/np.sum(s_within[idx])
        count += 1
        if abs(temK - old_k) < 1e-3:
            break
    # get feature index and save arff file
    index = fs_idx[I]
    return index
