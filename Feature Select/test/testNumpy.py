import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction import DictVectorizer


chList = ['a', 'b', 'c', 'd', 'e', 'f',  'g',
          'h', 'i', 'j', 'k', 'l', 'm', 'n',
          'o', 'p', 'q', 'r', 's', 't',
          'u', 'v', 'w', 'x', 'y', 'z']

# print(len(chList))
# print("chList = ['a'", end='')
# for i in range(1, 26):
#     print(", '%s'" % chList[i], end='')
#     if i % 8 == 0:
#         print()
# j = 26
# k = 0
# while j < 102:
#     curCh = chList[k]
#     for i in range(26):
#         print(", '%s%s'" % (curCh, chList[i]), end='')
#         if j % 8 == 0:
#             print()
#         j += 1
#     k += 1
# print(']')

# p = np.zeros(12, dtype=np.int8)
# print(p)
# index = [1, 3, 6, 8, 10]
# p[index] = 1
# print(p)

# s = '"T671":1.4648209,"T3772":1.2451016,"T974":1.2110636,"T1074":1.2045072'
# sdic = eval('{%s}' % s)
# print(sdic)
# d = {}
# if len(d) == 0:
#     print('kkkk')
# print(set(sdic.keys()))
# ss = 'T671:1.4648209'
# key, value = ss.split(':')
# print(str(key), value)

# es = set()
# if es == set():
#     print('k')

# argsort(list)->升序,argsort(-list)->降序

# src = 'D:\data\python\input\src\\'
# with open(src+'test.txt', 'r+') as file:
#     content = file.readlines()
#     print(content)
#     print(len(content))
#     file.write('1983oesflskf')
#     file.close()
#
# s = 't1,t2'
# print(s)
# print(set(s.split(',')))


# csv = 'D:\data\python\input\src\\'
# user = pd.read_csv(csv+'user_info.txt', encoding='gbk', delimiter='\t')
# # pandas 删除行列
# # 删除行index = [begin,...,end]
# user = user.drop(index=[0, 1])
# user = user.drop([0, 1])
# # 删除列columns=[label1,...,labeln]
# label1 = ''
# label2 = ''
# user = user.drop(columns=[label1, label2])

# for col in user.columns:
#     col_type = user[col].dtype
#     print(col_type)
#
#
# def changeM(M):
#     M[0][0] = 111111


M = np.array([[0, 0, 3, 4],[1, 2,3,4], [1,3,0,4]])
print(M == 0)
M[M == 0] = 1
print(M)
# v0 = M[:, 0]
# v1 = M[:, 1]
# print(M)
# print(v0)
# print(v1)
# print(v0@v1)
# print(v0*v1)
# print(np.dot(v0, v1))
# a = np.array(range(5))
# ar = np.arange(5)
# print(ar)
# print(a)
# a2 = np.power(a, 3)
# print(a2)
# print(a2.sum())
# 打乱顺序,无返回值
# np.random.shuffle(ar)
# print(ar)
# tt = np.random.rand(10)
# print(tt)
# print(np.random.rand(1)[0])

# print(M[1, :])
# print(M[:, 2])
# print(np.shape(M[:, 2:3]))
# print(M.T[2])
# print(np.shape(M.T[2]))
# print(M.sum())
# print(type(M[0][0]))
# M32 = M.astype(np.float32)
# print(type(M[0][0]))
# print(type(M32[0][0]))
# Mc = M
# if Mc is M:
#     print(type(M[1][2]))
# if type(M[1][2]) == np.complex128:
#     print('Yes')
# M = M.astype(np.float32)
# if np.isinf(M).any():
#     print('Yes!!!')
# print(np.isinf(M))
# typeM = type(M[0][0])
# print(typeM)
# if typeM == np.float32:
#     print('yes')
# else:
#     print('no')
# print(M[1:2])
# print(M[:, 1:2])
# # changeM(M)
# print(M[1].dtype)
# M[1] = np.array(M[1]).astype(np.int8)
# # M = M.astype(np.int8)
# print(M[1].dtype)


# print(M[1].dtype)
# print(M.nbytes)
# print(sys.getsizeof(M[1]))
# print(sys.getsizeof(M[1][1]))
#
# M = M.astype(np.int16)
# print(M[1].dtype)
# print(M.nbytes)
# print(sys.getsizeof(M[1]))
# print(sys.getsizeof(M[1][1]))
#
# M = M.astype(np.int8)
# print(M[1].dtype)
# print(M.nbytes)
# print(sys.getsizeof(M[0]))
# print(sys.getsizeof(M[1]))
# print(sys.getsizeof(M[2]))
# print(sys.getsizeof(M[1][1]))

# print(sys.getsizeof(np.zeros((1,1))))
# print(sys.getsizeof(np.zeros((2,2))))
# print(sys.getsizeof(np.zeros((3,3))))
# print(sys.getsizeof(np.zeros((4,4))))

# l = ['a', 'b', 'a', 'a', 'unknown']
# print(l)
# l.remove('unknown')
# l.insert(0, 'unknown')
# print(l)

# 随机数
# import random
#
# m = 12
#
# # [start, end]范围内的随机浮点数
# start = 0.0
# end = 9999.9999
# x = random.uniform(start, end)
# print(x)
#
# # [start, end]范围内的随机浮点数
# start = -2
# end = 10000
# x = random.randint(start, end)
# print(x)
# xlist = [random.randint(start, end) for _ in range(m)]
# print(xlist)
#
# for i in range(1000):
#     xlist = [random.randint(0, 99999) for _ in range(8)]
#     print(xlist)
#
# a = np.array([[1,2,3,4],
#              [3,4,2,1],
#              [4,5,3,6],
#              [9,3,6,45]])
# print(a)
#
# # 矩阵合并
# # 参数为(m1, m2)/(m, l)
# # # 合并行,行数相同 np.c_[m1, m2]
# # # 合并列,列数相同 np.r_[m1, m2]
#
# # 参数为1个矩阵 & 1个列表(m, l)
# # 合并行,行数相同 np.append(m, l, axis=0)
# # 合并列,行数相同 np.append(m, l, axis=1)
#
# # 矩阵删除
# # 删除行
# print(np.delete(a, 0, axis=0))
# print(np.delete(a, [1, 3], axis=0))
# print(a)
# # 删除列
# print(np.delete(a, 0, axis=1))
# print(np.delete(a, [1, 3], axis=1))
# print(a)

# M = np.random.randn(3,4)
# print(M)
# print(M[:, 1])
# print(np.shape(M[:, 1]))
# print(M[:, 0:1])
# print(np.shape(M[:, 0:1]))
#
# # 求标准化矩阵Z
# # 求M的列均值向量和列标准差矩阵
# u = np.mean(M, axis=0)
# s = np.std(M, axis=0)
# Z = (M-u)/s

# 矩阵乘法@
# a = np.zeros((2, 2))
# a[0][0] = 1
# a[0][1] = 2
# a[1][0] = 3
# a[1][1] = 4
# b = np.zeros((2, 2))
# b[0][0] = 1
# b[0][1] = -1
# b[1][0] = -2
# b[1][1] = 2
# print(a)
# print(b)
# a.T[0]=b.T[0]
# a.T[1]=b.T[0]
# print(a)
# print(a@b)
# print(np.dot(a, b))
# # np.savetxt('a.csv', a, '%.3f')
# strMa = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])
# print(type(strMa[0][0]))
# if type(strMa[0][0]) == np.str_:
#     print(strMa)
