import numpy as np
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


def reliefF(F, Lbls, k, filename='xxx', path=sglDst, nbr=5):
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
    # the higher the reliefF score, the more important the feature is
    index = np.argsort(score)[-1:-k - 1:-1]
    X = np.zeros((n, k))
    for i in range(k):
        X.T[i] = F.T[index[i]]
    with open(path + 'reliefF-%s-%s.csv' % (filename, k), 'w') as wf:
        wf.write(header(k))
        wf.write(mStr(np.c_[X.astype(np.str), Lbls]))
        wf.close()
    return index
