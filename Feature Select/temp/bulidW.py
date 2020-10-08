import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances

# kwargs_fs_all = {
#     'y': Lbls, "metric": "cosine", "neighbor_mode": "supervised",
#     "weight_mode": "binary", "k": 5, "t": 1,
#     "fisher_score": True, "reliefF": False
# }
# kwargs_fs = {
#     'y': Lbls, "neighbor_mode": "supervised",
#     "fisher_score": True
# }

# kwargs_ls_all = {
#     "metric": "cosine", "neighbor_mode": "knn",
#     "weight_mode": "binary", "k": 5, "t": 1,
#     "fisher_score": False, "reliefF": False
# }
# kwargs_ls = {
#     "metric": "cosine", "neighbor_mode": "knn",
#     "weight_mode": "binary", "k": 5
# }

# kwargs_tr_laplacian_all = {"metric": "euclidean", "neighbor_mode": "knn",
# "weight_mode": "heat_kernel", "k": 5, "t": 1,
# "fisher_score": False, "reliefF": False}
# kwargs_tr_laplacian = {"neighbor_mode": "knn",
# "weight_mode": "heat_kernel", "k": 5, "t": 1}

# kwargs_tr_fisher_all = {'y': numLbls, "metric": "cosine", "neighbor_mode": "supervised",
# "weight_mode": "binary", "k": 5, "t": 1,
# "fisher_score": True, "reliefF": False}
# kwargs_tr_fisher = {'y': numLbls, "neighbor_mode": "supervised",
# "fisher_score": True}


# tr->trace ratio
def getW(F, k=5, Lbls=None, mode='knn', t=1, tr=False):
    n = np.shape(F)[0]
    if mode == 'knn':
        G = np.zeros((n * (k + 1), 3))
        G[:, 0] = np.tile(np.arange(n), (k + 1, 1)).reshape(-1)
        # trace ratio -> laplacian
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
        # fisher score & trace ratio -> fisher score
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
