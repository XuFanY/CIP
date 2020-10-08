import math
import numpy as np
from itertools import combinations

# 各项固定参数
# Beta->惩罚参数β，Tmax->最大迭代次数，Lambda->拉格朗日乘子λ
# Epsilon->矩阵W的收敛精度，LdB->Lambda/Beta
Beta = 1e8
Tmax = 1e3
Lambda = 1e-3
Epsilon = 1e-4
LdB = 1e-11


# 将矩阵M中心标准化
def standard(M):
    u = np.mean(M, axis=0)
    s = np.std(M, axis=0)
    n = len(s)
    for i in range(0, n):
        if s[i] == 0:
            s[i] = 1
    return (M-u)/s


# 计算矩阵M的F范数的a次方(a默认为1)
def norm(M, a=1):
    return np.linalg.norm(M)**a


# 计算M*diag(v)，其中v为向量
def linear(M, v):
    return M@np.diag(v)


# 求向量a & b的内积
def dot(a, b):
    return (a * b).sum()


# 从原始数据文件arrhythmia.csv中获取基本输入信息C&F
def arrhythmia():
    with open(r'C:\Users\AdPw\Desktop\arrhythmia.csv') as file:
        lines = file.readlines()
        datas = [line.split(',') for line in lines]
        # 获取样本数n和特征数m
        m = len(datas[0]) - 1
        n = len(datas)
        # 获取类的总数c
        labels = [data[-1].strip() for data in datas]
        cl = [int(labels.pop(0))]
        for label in labels:
            if label not in cl:
                cl.append(int(label))
        c = max(cl)
        # 获取矩阵C & F的数据
        F = np.zeros((n, m))
        C = np.zeros((n, c))
        for i, data in zip(range(0, n), datas):
            # 获取矩阵C的数据
            label = data.pop().strip()
            C[i][int(label)-1] = 1
            # 获取矩阵F的数据
            f = []
            for j in range(0, m):
                if data[j] != '?':
                    f.append(float(data[j]))
                else:
                    f.append(0.0)
            F[i] = f
        file.close()
        return C, standard(F)


# 从原始数据文件ionosphere.csv中获取基本输入信息C&F
def ionosphere():
    with open(r'C:\Users\AdPw\Desktop\ionosphere.csv') as file:
        # 略去第一行
        lines = file.readlines()[1:]
        # 获取样本数n和特征数m
        m = len(lines[0].split(',')) - 1
        n = len(lines)
        # 获取类的总数c
        labels = [line[-2] for line in lines]
        cl = [labels.pop(0)]
        for label in labels:
            if label not in cl:
                cl.append(label)
        c = len(cl)
        # 获取矩阵C & F的数据
        F = np.zeros((n, m))
        C = np.zeros((n, c))
        for i, line in zip(range(0, n), lines):
            datas = line.split(',')
            # 获取矩阵C的数据
            label = datas.pop().strip()
            if label == 'b':
                C[i][0] = 1
            else:
                C[i][1] = 1
            # 获取矩阵F的数据
            f = []
            for data in datas:
                f.append(float(data))
            F[i] = f
        file.close()
        return C, standard(F)


# 对矩阵M进行谱分解并返回其前k大的特征值(存储在对角矩阵中，降序)及其对应的特征向量
def eigk(M, k):
    d, q = np.linalg.eig(M)
    ds = np.argsort(d)
    D = np.diag(d[ds[-1:-k-1:-1]])
    Q = q[:, ds[-1:-k-1:-1]]
    return D, Q


# 对矩阵M进行谱分解并返回排好序的特征值和特征向量(降序)
def eig(M):
    k = np.shape(M)[0]
    return eigk(M, k)


# 计算矩阵M的类间相似度矩阵(按照cip中的定义式计算)
def similarity(M):
    c = np.shape(M)[1]
    S = np.zeros((c, c))
    for i in range(0, c):
        for j in range(i+1, c):
            S[i][j] = S[j][i] = norm(M.T[i]-M.T[j], 2)
    c0 = -c*(c-1)/(2*S.sum())
    for i in range(0, c):
        for j in range(i, c):
            if dot(M.T[i], M.T[j]) != 0:
                S[i][j] = S[j][i] = math.exp(c0 * S[i][j])
            else:
                S[i][j] = S[j][i] = 0
    return S


# 求组合数C(k,m)对应的序列
def combination(k, m):
    if k < 1 or m < k:
        return
    indexs = []
    if k > 2:
        for begin in range(0, m - k + 1):
            l = [begin]
            for j in range(begin + 1, m - k + 2):
                l = l + list(range(j, j + k - 2))
                end = list(range(j + k - 2, m))
                for x in end:
                    l.append(x)
                    indexs.append(l[:])
                    l.pop()
                l = [begin]
        return indexs
    elif k == 2:
        for begin in range(0, m - 1):
            l = [begin]
            for j in range(begin + 1, m):
                l = l + [j]
                indexs.append(l[:])
                l = [begin]
        return indexs
    for num in range(0, m):
        l = [num]
        indexs.append(l)
    return indexs


def testP(k, A, B):
    m = len(A)
    indexs = list(combinations(range(0, m), k))
    print(len(indexs))
    value = []
    for index in indexs:
        M = A[index[0]]
        for i in range(1, k):
            M = M + A[index[i]]
        value.append(norm(M-B))
    vs = np.argsort(value)
    return indexs[vs[0]]


# CIP算法实现
def cip(C, F):
    c = np.shape(C)[1]
    (n, m) = np.shape(F)
    CT = C.transpose()
    FT = F.transpose()
    A = CT@F
    A0 = A.T@A
    Tau = 1.0/eig(A0)[0].max()
    # 输入挑选的特征数k
    k = 0
    while k not in range(1,m):
        k = int(input('Please input the number of selected features k (0<k<' + str(m) + ')'))
    # 初始化
    t = 0
    S = similarity(C)
    D, Q = eig(S)
    Gamma = n * Q @ np.sqrt(D)
    W = np.zeros((m,c))
    fTc = []
    M = C/n
    for i in range(0, m):
        fTi = FT[i].reshape(1, n)
        X = fTi@M
        fTc.append(norm(X.T@X-S))
    fTcs = np.argsort(fTc)
    p = np.zeros(m)
    for index in fTcs[:k]:
        p[index] = 1
    print('p0: \n', p)
    V = 1 / m * np.ones((m, c))

    index0 = []
    index1 = []
    Ma = []
    # 迭代
    while t < Tmax:
        # 更新U
        Wv = W+V/Beta
        Wv_norm = norm(Wv)
        U = max(Wv_norm-LdB,0)*Wv/Wv_norm

        # 更新W
        W_old = W
        Ap = linear(A, p)
        Omega = Ap.T@(Ap@W-Gamma)
        W = (1/(Beta+Tau))*(Beta*U+V+Tau*W-Omega)
        if norm(W - W_old) <= Epsilon or np.isnan(W).any():
            break

        # 更新p
        cTfw = []
        X = []
        for i in range(0, m):
            fTi = F.T[i].reshape(n, 1)
            wi = W[i].reshape(1, c)
            x = CT@fTi@wi
            X.append(x)
            cTfw.append(norm(x-Gamma))
        cTfws = np.argsort(cTfw)
        p = np.zeros(m)
        index0.append(cTfws[:k])
        index1.append(testP(k, X, Gamma))
        Ma.append(X)
        for index in cTfws[:k]:
            p[index] = 1

        # 更新V
        V = V-Beta*(W-U)

        # 更新Tau
        Av = A0@V
        Tau = norm(Av[0])
        for i in range(1, m):
            Tau = max(Tau, norm(Av[i]))
        t += 1
    return t, p, index0, index1, Ma, Gamma

C, F = ionosphere()
# C, F = arrhythmia()

t, p, index0, index1, Ma, Gamma = cip(C, F)
k = len(index0[0])

print('迭代次数: \n', t)
print('p: \n', p)
for i in index0:
    i.sort()
print('old: \n',sorted(index0))

print('combination: \n',index1)

for i0, i1, ma in zip(index0, index1, Ma):
    M0 = ma[i0[0]]
    M1 = ma[i1[0]]
    for j in range(1, k):
        M0 = M0 + ma[i0[j]]
        M1 = M1 + ma[i1[j]]

print('M0: \n', M0)
print('M1: \n', M1)
# print('Gamma: \n', Gamma)
print('norm(M0-Gamma): \n', norm(M0-Gamma))
print('norm(M1-Gamma): \n', norm(M1-Gamma))


# print(t)
# print('u')
# print(U)
# print('v')
# print(V)
# print('w_old')
# print(W_old)
# print('w')
# print(W)
# print('omega')
# print(Omega)
