# 稀疏矩阵运算示例程序(csr)
import numpy as np
from scipy import sparse

# csr_matrix -> 行压缩稀疏矩阵
# 自写构造程序,了解csr稀疏矩阵构造方法
# 内置方法 -> sparse.csr_matrix(F)
def csr(F):
    count = 0
    datas = []
    columns = []
    row_offsets = [0]
    m, n = np.shape(F)
    for i in range(m):
        for j in range(n):
            if F[i][j] != 0:
                count += 1
                columns.append(j)
                datas.append(F[i][j])
        row_offsets.append(count)
    M_csr = sparse.csr_matrix((datas, columns, row_offsets), shape=(m, n))
    return M_csr


# csc_matrix -> 列压缩稀疏矩阵
# 自写构造程序,了解csc稀疏矩阵构造方法
# 内置方法 -> sparse.csc_matrix(F)
def csc(F):
    count = 0
    datas = []
    rows = []
    column_offsets = [0]
    m, n = np.shape(F)
    for j in range(n):
        for i in range(m):
            if F[i][j] != 0:
                count += 1
                rows.append(i)
                datas.append(F[i][j])
        column_offsets.append(count)
        M_csc = sparse.csc_matrix((datas, rows, column_offsets), shape=(m, n))
    return M_csc


def stdard(F_csr):
    u = F_csr.mean(axis=0)
    s = F_csr.std(axis=0)
    n = len(s)
    for i in range(n):
        if s[i] == 0:
            s[i] = 1
    return (F_csr - u) / s
    pass


# A = np.array([[1, 0, 0, 0],
#              [3, 4, 2, 0],
#              [4, 5, 3, 6],
#              [9, 3, 6, 45]])
endl = '\n'
# 手动构造两个稀疏矩阵
A = np.array([[1, 0, 0, 0],
             [3, 0, 2, 0],
             [0, 0, 0, 6]])

B = np.array([[0, 0, 3],
             [3, 0, 0],
             [0, 0, 0],
             [0, 3, 6]])
print('A:\n', A, '\nB:\n', B)
# csra = csr(A)
csra = sparse.csr_matrix(A)
print(sparse.issparse(csra))
print(sparse.issparse(A))
# csrb = csr(B)
csrb = sparse.csr_matrix(B)
print('A_csr:\n', csra, '\nB_csr:\n', csrb)

# 稀疏矩阵转化为一般矩阵
A = csra.toarray()
print('A(toarray):\n', A)

# 查看非0元素
dataB = csrb.data
print('B_data:\n', dataB)

# 最值,平均值 & 求和(0列1行)
# max/min
maxa = csra.max()
maxac = csra.max(axis=0)
maxar = csra.max(axis=1)
print('A_max: ', maxa, '\nA_max_col:\n', maxac, '\nA_max_row:\n', maxar)
mina = csra.min()
minac = csra.min(axis=0)
minar = csra.min(axis=1)
print('A_min: ', mina, '\nA_min_col:\n', minac, '\nA_min_row:\n', minar)
meana = csra.mean()
meanac = csra.mean(axis=0)
meanar = csra.mean(axis=1)
print('A_mean: ', meana, '\nA_mean_col:\n', meanac, '\nA_mean_row:\n', meanar)
suma = csra.sum()
sumac = csra.sum(axis=0)
sumar = csra.sum(axis=1)
print('A_sum: ', suma, '\nA_sum_col:\n', sumac, '\nA_sum_row:\n', sumar)

# 稀疏矩阵转置
csrbt = csrb.transpose()
print('B.T:\n', csrbt.toarray())

# 获取行列
ar1 = csra.getrow(1)
bc2 = csrb.getcol(2)
print('Ar1:\n', ar1)
print('Bc2:\n', bc2)
print('ar1Dbc2:\n', ar1.dot(bc2))

# 稀疏矩阵乘法和内积
AMulb = A*csrb
aAtB = csra@B
aMb = csra*csrb
print('A*B_csr:\n', AMulb)
print('A_csr*B:\n', aAtB)
print('A_csr*B_csr:\n', aMb)
print('A@B:\n', A@B)
