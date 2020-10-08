import math
import numpy as np
from sklearn import metrics as mtr

# 各项固定参数
# Beta->惩罚参数β,Epsilon->矩阵W的收敛精度
# Lambda->拉格朗日乘子λ,LdB->Lambda/Beta
# Tmax->最大迭代次数
Beta = 1e8
Epsilon = 1e-4
Lambda = 1e-3
LdB = Lambda/Beta
Tmax = 1e3


# 计算矩阵mat的F范数的a次方(a默认为1)
def norm(mat, a=1):
    return np.linalg.norm(mat)**a


# 对矩阵mat进行谱分解并返回排好序的特征值和特征向量(降序)
def eig(mat):
    d, q = np.linalg.eig(mat)
    index = np.argsort(-d)
    D = np.diag(d[index])
    Q = q[:, index]
    return D, Q


def start(C, F, mutual=False, multi=False):
    n, c = np.shape(C)
    m = np.shape(F)[1]
    CT = C.transpose()
    # 计算类间相似矩阵S
    S = np.zeros((c, c), dtype=np.float32)
    if mutual:
        for i in range(c):
            S[i][i] = mtr.mutual_info_score(C.T[i], C.T[i])
            for j in range(i + 1, c):
                S[i][j] = S[j][i] = mtr.mutual_info_score(C.T[i], C.T[j])
    else:
        for i in range(c):
            for j in range(i + 1, c):
                S[i][j] = S[j][i] = norm(C.T[i] - C.T[j], 2)
        c0 = -c * (c - 1) / (2 * S.sum())
        for i in range(c):
            for j in range(i, c):
                if np.dot(C.T[i], C.T[j]) != 0:
                    S[i][j] = S[j][i] = math.exp(c0 * S[i][j])
                else:
                    S[i][j] = S[j][i] = 0
    D, Q = eig(S)
    Gamma = n * Q @ np.sqrt(D)
    A = CT @ F
    A0 = A.T @ A
    score = []
    Cdn = C / n
    for i in range(m):
        X = F.T[i:i + 1] @ Cdn
        score.append(norm(X.T @ X - S))
    idx0 = np.argsort(score)
    if multi:
        return idx0, A, A0, CT, Gamma, S
    return idx0, A, A0, CT, Gamma


# CIP算法实现
def cip(idx0, A, A0, CT, F, Gamma, k):
    c, n = np.shape(CT)
    m = np.shape(F)[1]
    # 初始化
    p = np.zeros(m, dtype=np.int8)
    p[idx0[:k]] = 1
    t = 0
    Tau = 1.0 / abs(eig(A0)[0].max())
    W = np.zeros((m, c), dtype=np.float32)
    V = (1.0 / m) * np.ones((m, c), dtype=np.float32)
    # 迭代
    while t < Tmax:
        # 更新U
        Wv = W + V / Beta
        Wv_norm = norm(Wv)
        U = max(Wv_norm - LdB, 0) * Wv / Wv_norm
        # 更新W
        W_old = W
        Ap = A @ np.diag(p)
        Omega = Ap.T @ (Ap @ W - Gamma)
        W = (Tau / (Beta * Tau + 1)) * (Beta * U + V + (W - Tau * Omega) / Tau)
        if norm(W - W_old) <= Epsilon:
            break
        # 更新p
        score = []
        for i in range(m):
            score.append(norm(CT @ F[:, i:i + 1] @ W[i:i + 1] - Gamma))
        index = np.argsort(score)
        p[index[k:]] = 0
        p[index[:k]] = 1
        # 更新V
        V = V - Beta * (W - U)
        # 更新Tau
        Av = A0 @ V
        Av_norms = np.linalg.norm(Av, axis=1)
        Tau = 1 / Av_norms.max()
        t += 1
    index = [i for i in range(m) if p[i] == 1]
    return t, p, index


# 评价指标
# C=(c1, c2, ..., cl, ... ,cc), Fs=(f1', f2', ..., fk')
# cc->类间相关性,cr->分类冗余性,red->分类冗余率
def evaluate(C, F, indexses, multi=False, S=None):
    n, c = np.shape(C)
    rsts = []
    if multi:
        for indexs in indexses:
            rst = []
            for index in indexs:
                k = len(index)
                Fs = F[:, index]
                mat = C.T @ Fs @ Fs.T @ C / (n ** 2) - S
                kk = k * k - k
                if kk == 0:
                    kk = 1
                cc = norm(mat, 2) / kk
                rst.append(list(index) + list([cc]))
            rsts.append(rst)
        return rsts
    for indexs in indexses:
        rst = []
        for index in indexs:
            k = len(index)
            Fs = F[:, index]
            cr = red = 0
            for i in range(k):
                fi = Fs[:, i]
                for j in range(i + 1, k):
                    fj = Fs[:, j]
                    red += 2 * (fi @ fj)
                    for l in range(c):
                        cl = C[:, l]
                        cr += 2 * (((fi @ cl) * (fj @ cl) / np.var(cl)) ** 2)
            kk = k * k - k
            if kk == 0:
                kk = 1
            rst.append(list(index) + list([cr / (c * kk * n ** 4), red / (kk * n)]))
        rsts.append(rst)
    return rsts
