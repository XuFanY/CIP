import numpy as np

# sgl->single(单标签),mul(多标签)
sglSrc = 'D:/data/fs/input/single/src/'
sglPrepDst = 'D:/data/fs/input/single/prep/'
sglArffDst = 'D:/data/fs/input/single/arff/'
mulSrc = 'D:/data/fs/input/multi/src/'
# 单标签输出路径 & 多标签输出路径
sglDst = 'D:/data/fs/output/single/'
mulDst = 'D:/data/fs/output/multi/'
chList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
          'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
          'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
          'y', 'z', 'aa', 'ab', 'ac', 'ad', 'ae',
          'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al',
          'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as',
          'at', 'au', 'av', 'aw', 'ax', 'ay', 'az',
          'ba', 'bb', 'bc', 'bd', 'be' , 'bf', 'bg',
          'bh', 'bi', 'bj', 'bk', 'bl', 'bm' , 'bn',
          'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu',
          'bv', 'bw', 'bx', 'by', 'bz', 'ca', 'cb',
          'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci',
          'cj', 'ck', 'cl', 'cm', 'cn', 'co', 'cp',
          'cq', 'cr', 'cs', 'ct', 'cu', 'cv', 'cw',
          'cx', 'cy', 'cz']


# 将数值标签转换为字符标签
def numToCh(y_num):
    y_ch = [chList[int(label)] for label in y_num]
    return y_ch


# 将字符标签转换为数值标签
def chToNum(y_ch, labels, c):
    clsDict = {}
    for i in range(c):
        clsDict[labels[i]] = i
    y_num = [clsDict[label] for label in y_ch]
    return y_num


# 将矩阵xStr所有在列表column中的列依照字典序排序映射成自然数
def reflect(xStr, column):
    m = np.shape(xStr)[0]
    for j in column:
        csj = sorted(set(xStr[:, j]))
        n, csjd = len(csj), {}
        for i in range(n):
            csjd[csj[i]] = i
        for i in range(m):
            xStr[i][j] = csjd[xStr[i][j]]


# 去除X的无用特征并将矩阵X中心标准化
def standard(X, mm=False):
    # 去除X的无用特征(所有值均相同的列)
    columns = []
    m, n = np.shape(X)
    for j in range(n):
        add, ele0 = True, X[0][j]
        for i in range(1, m):
            if X[i][j] != ele0:
                add = False
                break
        if add:
            columns.append(j)
    X = np.delete(X, columns, axis=1)
    # 标准化X
    # max-min方式
    if mm:
        minX = np.min(X, 0)
        return (X - minX) / (np.max(X, 0) - minX)
    # zscore方式
    s = np.std(X, axis=0)
    s[s == 0] = 1
    return (X - np.mean(X, axis=0)) / s


def savearff(F, y_ch, filename, labels):
    with open(sglArffDst + '%s.arff' % filename, 'w') as wf:
        wf.write('@relation %s\n' % filename)
        wf.write('\n')
        n, m = np.shape(F)
        for i in range(m):
            wf.write('@attribute feature%d numeric\n' % i)
        wf.write('@attribute class {%s}\n' % str(labels)[1:-1])
        wf.write('\n@data\n')
        for i in range(n):
            buffs = '%s' % F[i][0]
            for j in range(1, m):
                buffs += ',%s' % F[i][j]
            buffs += ',%s\n' % y_ch[i]
            wf.write(buffs)
        wf.close()


# fs(str){cip,cipm,fisher,laplace,reliefF,spec,trcRto}
def savearffs(F, indexs, filename, fs, multi=False, y_num=None, C=None):
    n = np.shape(F)[0]
    if multi:
        c = np.shape(C)[1]
        dst = mulDst + '%s/%s-' % (fs, filename)
        for index in indexs:
            k = len(index)
            filePath = dst + '%d.arff' % k
            with open(filePath, 'w') as wf:
                wf.write("@relation '%s: -C %d'\n" % (filename, c))
                wf.write('\n')
                for i in range(c):
                    wf.write('@attribute class%d {0,1}\n' % i)
                for i in range(k):
                    wf.write('@attribute feature%d numeric\n' % i)
                wf.write('\n@data\n')
                for i in range(n):
                    buffs = '%s' % C[i][0]
                    for j in range(1, c):
                        buffs += ',%s' % C[i][j]
                    for j in range(k):
                        buffs += ',%s' % F[i][index[j]]
                    wf.write(buffs)
                    wf.write('\n')
                wf.close()
    else:
        y_ch = numToCh(y_num)
        labels = sorted(set(y_ch))
        dst = sglDst + '%s/%s-' % (fs, filename)
        for index in indexs:
            k = len(index)
            filePath = dst + '%d.arff' % k
            with open(filePath, 'w') as wf:
                wf.write('@relation %s\n' % filename)
                wf.write('\n')
                for i in range(k):
                    wf.write('@attribute feature%d numeric\n' % i)
                wf.write('@attribute class {%s}\n' % str(labels)[1:-1])
                wf.write('\n@data\n')
                for i in range(n):
                    buffs = '%s' % F[i][index[0]]
                    for j in range(1, k):
                        buffs += ',%s' % F[i][index[j]]
                    buffs += ',%s\n' % y_ch[i]
                    wf.write(buffs)
                wf.close()


# 预处理单标签数据(需要时可生成arff文件)
# Al->All(源数据集,即产生Te、Tr等数据集的源文件),Te->Test(测试集),Tr->Train(训练集),Va->Valid(验证集),N->Noise(噪声)
# 预处理原始数据文件aya.csv
def arrhythmia(arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + 'aya.csv', dtype=np.str, delimiter=',')
    # 获取矩阵C
    y_num = xY[:, -1].astype(np.int8)
    y_num = [label if label < 11 else label - 3 for label in y_num]
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i] - 1] = 1
    # 获取矩阵F
    xStr = xY[:, :-1]
    xStr[xStr == '?'] = '0'
    F = standard(xStr.astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'aya', labels)
    np.savetxt(sglPrepDst + 'ayaCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'ayaF.csv', F, delimiter=',', fmt='%s')
    return 'aya'


# 预处理原始数据文件bch.data(#->z)
def bachCrlHrmy(arff=False):
    # 读取文件并存放在矩阵中
    ukn2XY = np.loadtxt(sglSrc + 'bch.data', dtype=np.str, delimiter=',')
    # 获取矩阵C
    y_ch = ukn2XY[:, -1]
    n, labels = len(y_ch), sorted(set(y_ch))
    c = len(labels)
    y_num = chToNum(y_ch, labels, c)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    xStr = ukn2XY[:, 2:-1]
    reflect(xStr, [12])
    xStr[xStr == 'NO'] = '0'
    xStr[xStr == 'YES'] = '1'
    F = standard(xStr.astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'bch', labels)
    np.savetxt(sglPrepDst + 'bchCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'bchF.csv', F, delimiter=',', fmt='%s')
    return 'bch'


# 预处理原始数据文件chs.data
def chess(arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + 'chs.data', dtype=np.str, delimiter=',')
    # 获取矩阵C
    y_ch = xY[:, -1]
    n, labels = len(y_ch), sorted(set(y_ch))
    c = len(labels)
    y_num = chToNum(y_ch, labels, c)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    xStr = xY[:, :-1]
    reflect(xStr, [0, 2, 4])
    F = standard(xStr.astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'chs', labels)
    np.savetxt(sglPrepDst + 'chsCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'chsF.csv', F, delimiter=',', fmt='%s')
    return 'chs'


# 预处理原始数据文件fotTe&fotVa&fotTr.csv
def frtOdrTrm(arff=False):
    # 读取文件并存放在矩阵中
    xCTe = np.loadtxt(sglSrc + 'fotTe.csv', dtype=np.float32, delimiter=',')
    xCVa = np.loadtxt(sglSrc + 'fotVa.csv', dtype=np.float32, delimiter=',')
    xCTr = np.loadtxt(sglSrc + 'fotTr.csv', dtype=np.float32, delimiter=',')
    # 获取矩阵C
    C = np.r_[xCTe[:, -6:], xCVa[:, -6:], xCTr[:, -6:]].astype(np.int8)
    n, c = np.shape(C)
    y_num = []
    for i in range(n):
        for j in range(c):
            if C[i][j] == -1:
                C[i][j] = 0
            else:
                y_num.append(j)
    y_ch = numToCh(y_num)
    labels = sorted(set(y_ch))
    # 获取矩阵F
    F = standard(np.r_[xCTe[:, :-6], xCVa[:, :-6], xCTr[:, :-6]])
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'fot', labels)
    np.savetxt(sglPrepDst + 'fotCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'fotF.csv', F, delimiter=',', fmt='%s')
    return 'fot'


# 预处理原始数据文件hvyNTe&hvyNTr/hvyTe&hvyTr.csv
# filename->hvy/hvyN
def hillValley(filename='hvy', arff=False):
    # 读取文件并存放在矩阵中
    xYTe = np.loadtxt(sglSrc + '%sTe.csv' % filename, dtype=np.float32, delimiter=',', skiprows=1)
    xYTr = np.loadtxt(sglSrc + '%sTr.csv' % filename, dtype=np.float32, delimiter=',', skiprows=1)
    # 获取矩阵C
    y_num = np.r_[xYTe[:, -1], xYTr[:, -1]].astype(np.int8)
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    F = standard(np.r_[xYTe[:, :-1], xYTr[:, :-1]])
    # 生成arff文件
    if arff:
        savearff(F, y_ch, filename, labels)
    np.savetxt(sglPrepDst + '%sCY.csv' % filename, np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + '%sF.csv' % filename, F, delimiter=',', fmt='%s')
    return filename


# 预处理原始数据文件ise.csv
def ionosphere(arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + 'ise.csv', dtype=np.str, delimiter=',', skiprows=1)
    # 获取矩阵C
    y_ch = xY[:, -1]
    n, labels = len(y_ch), sorted(set(y_ch))
    c = len(labels)
    y_num = chToNum(y_ch, labels, c)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    F = standard(xY[:, :-1].astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'ise', labels)
    np.savetxt(sglPrepDst + 'iseCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'iseF.csv', F, delimiter=',', fmt='%s')
    return 'ise'


# 预处理原始数据文件lbm.data
def libMove(arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + 'lbm.data', dtype=np.float32, delimiter=',')
    # 获取矩阵C
    y_num = xY[:, -1].astype(np.int8)
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i] - 1] = 1
    # 获取矩阵F
    F = standard(xY[:, :-1])
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'lbm', labels)
    np.savetxt(sglPrepDst + 'lbmCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'lbmF.csv', F, delimiter=',', fmt='%s')
    return 'lbm'


# 预处理原始数据文件mdlVa&mdlTr.data & mdlVa&mdlTr.labels
def madelon(arff=False):
    # 获取矩阵C
    yVa = np.loadtxt(sglSrc + 'mdlVa.labels', np.int8)
    yTr = np.loadtxt(sglSrc + 'mdlTr.labels', np.int8)
    y_num = np.r_[yVa, yTr]
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    xVa = np.loadtxt(sglSrc + 'mdlVa.data', np.float32)
    xTr = np.loadtxt(sglSrc + 'mdlTr.data', np.float32)
    F = standard(np.r_[xVa, xTr])
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'mdl', labels)
    np.savetxt(sglPrepDst + 'mdlCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'mdlF.csv', F, delimiter=',', fmt='%s')
    return 'mdl'


# 预处理原始数据文件mtf*.txt
def multFea(arff=False):
    # 获取矩阵F
    xFac = np.loadtxt(sglSrc + 'mtfFac.txt', dtype=np.float32)
    xFou = np.loadtxt(sglSrc + 'mtfFou.txt', dtype=np.float32)
    xKar = np.loadtxt(sglSrc + 'mtfKar.txt', dtype=np.float32)
    xMor = np.loadtxt(sglSrc + 'mtfMor.txt', dtype=np.float32)
    xPix = np.loadtxt(sglSrc + 'mtfPix.txt', dtype=np.float32)
    xZer = np.loadtxt(sglSrc + 'mtfZer.txt', dtype=np.float32)
    F = standard(np.c_[xFac, xFou, xKar, xMor, xPix, xZer])
    n = np.shape(F)[0]
    # 获取矩阵C
    i, j = 0, 0
    y_num = []
    C = np.zeros((n, int(n/200)), dtype=np.int8)
    while i < n:
        for k in range(200):
            y_num.append(j)
            C[i][j] = 1
            i += 1
        j += 1
    y_ch = numToCh(y_num)
    labels = sorted(set(y_ch))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'mtf', labels)
    np.savetxt(sglPrepDst + 'mtfCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'mtfF.csv', F, delimiter=',', fmt='%s')
    return 'mtf'


# 预处理原始数据文件musk1&musk2.data
def musk(arff=False):
    # 读取文件并存放在矩阵中
    ukn2XY1 = np.loadtxt(sglSrc + 'musk1.data', dtype=np.str, delimiter=',')
    ukn2XY2 = np.loadtxt(sglSrc + 'musk2.data', dtype=np.str, delimiter=',')
    # 获取矩阵C
    y_num = np.r_[ukn2XY1[:, -1], ukn2XY2[:, -1]].astype(np.int8)
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    F = standard(np.r_[ukn2XY1[:, 2:-1], ukn2XY2[:, 2:-1]].astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'musk', labels)
    np.savetxt(sglPrepDst + 'muskCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'muskF.csv', F, delimiter=',', fmt='%s')
    return 'musk'


# 预处理原始数据文件odtTe&odtTr.csv
def optdigits(arff=False):
    # 读取文件并存放在矩阵中
    xYTe = np.loadtxt(sglSrc + 'odtTe.csv', dtype=np.float32, delimiter=',')
    xYTr = np.loadtxt(sglSrc + 'odtTr.csv', dtype=np.float32, delimiter=',')
    # 获取矩阵C
    y_num = np.r_[xYTe[:, -1], xYTr[:, -1]].astype(np.int8)
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    F = standard(np.r_[xYTe[:, :-1], xYTr[:, :-1]])
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'odt', labels)
    np.savetxt(sglPrepDst + 'odtCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'odtF.csv', F, delimiter=',', fmt='%s')
    return 'odt'


# 预处理原始数据文件old.data
def oznLvlDet(arff=False):
    # 读取文件并存放在矩阵中
    ukn1XY = np.loadtxt(sglSrc + 'old.data', dtype=np.str, delimiter=',')
    # 获取矩阵C
    y_num = ukn1XY[:, -1].astype(np.int8)
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    xStr = ukn1XY[:, 1:-1]
    xStr[xStr == '?'] = '0'
    F = standard(xStr.astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'old', labels)
    np.savetxt(sglPrepDst + 'oldCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'oldF.csv', F, delimiter=',', fmt='%s')
    return 'old'


# 预处理原始数据文件plr.txt
def planRlx(arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + 'plr.txt', dtype=np.float32)
    # 获取矩阵C
    y_num = xY[:, -1].astype(np.int8)
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i] - 1] = 1
    # 获取矩阵F
    F = standard(xY[:, :-1])
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'plr', labels)
    np.savetxt(sglPrepDst + 'plrCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'plrF.csv', F, delimiter=',', fmt='%s')
    return 'plr'


# 预处理原始数据文件qbg.csv
def qsarBdg(arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + 'qbg.csv', dtype=np.str, delimiter=',')
    # 获取矩阵C
    y_ch = xY[:, -1]
    n, labels = len(y_ch), sorted(set(y_ch))
    c = len(labels)
    y_num = chToNum(y_ch, labels, c)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    F = standard(xY[:, :-1].astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'qbg', labels)
    np.savetxt(sglPrepDst + 'qbgCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'qbgF.csv', F, delimiter=',', fmt='%s')
    return 'qbg'


# 预处理原始数据文件sbs.data
def soybeanSmall(arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + 'sbs.data', dtype=np.str, delimiter=',')
    # 获取矩阵C
    y_ch = xY[:, -1]
    n, labels = len(y_ch), sorted(set(y_ch))
    c = len(labels)
    y_num = chToNum(y_ch, labels, c)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    F = standard(xY[:, :-1].astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'sbs', labels)
    np.savetxt(sglPrepDst + 'sbsCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'sbsF.csv', F, delimiter=',', fmt='%s')
    return 'sbs'


# 预处理原始数据文件sblTr&sblTe.data
def soybeanLarge(arff=False):
    # 读取文件并存放在矩阵中
    yXTr = np.loadtxt(sglSrc + 'sblTr.data', dtype=np.str, delimiter=',')
    yXTe = np.loadtxt(sglSrc + 'sblTe.data', dtype=np.str, delimiter=',')
    # 获取矩阵C
    y_ch = np.r_[yXTr[:, 0], yXTe[:, 0]]
    n, labels = len(y_ch), sorted(set(y_ch))
    c = len(labels)
    y_num = chToNum(y_ch, labels, c)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    xStr = np.r_[yXTr[:, 1:], yXTe[:, 1:]]
    xStr[xStr == '?'] = '0'
    F = standard(xStr.astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'sbl', labels)
    np.savetxt(sglPrepDst + 'sblCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'sblF.csv', F, delimiter=',', fmt='%s')
    return 'sbl'


# 预处理原始数据文件ukl.csv
def userKnowlde(arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + 'ukl.csv', dtype=np.str, delimiter=',', skiprows=1)
    # 获取矩阵C
    y_ch = xY[:, -1]
    n, labels = len(y_ch), sorted(set(y_ch))
    c = len(labels)
    y_num = chToNum(y_ch, labels, c)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i]] = 1
    # 获取矩阵F
    F = standard(xY[:, :-1].astype(np.float32))
    # 生成arff文件
    if arff:
        savearff(F, y_ch, 'ukl', labels)
    np.savetxt(sglPrepDst + 'uklCY.csv', np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + 'uklF.csv', F, delimiter=',', fmt='%s')
    return 'ukl'


# 预处理原始数据文件filename.csv
# filename->wqr/wqw
def wineQuality(filename='wqr', arff=False):
    # 读取文件并存放在矩阵中
    xY = np.loadtxt(sglSrc + '%s.csv' % filename, dtype=np.float32, delimiter=',', skiprows=1)
    # 获取矩阵C
    y_num = xY[:, -1].astype(np.int8)
    n, y_ch = len(y_num), numToCh(y_num)
    labels = sorted(set(y_ch))
    c = len(labels)
    C = np.zeros((n, c), dtype=np.int8)
    for i in range(n):
        C[i][y_num[i] - 3] = 1
    # 获取矩阵F
    F = standard(xY[:, :-1])
    # 生成arff文件
    if arff:
        savearff(F, y_ch, filename, labels)
    np.savetxt(sglPrepDst + '%sCY.csv' % filename, np.c_[C, y_num], delimiter=',', fmt='%d')
    np.savetxt(sglPrepDst + '%sF.csv' % filename, F, delimiter=',', fmt='%s')
    return filename


# 获取单标签数据集数据(C & y_num & F)
def getCYF(filename):
    CY = np.loadtxt(sglPrepDst + '%sCY.csv' % filename, np.int8, delimiter=',')
    C, y_num = CY[:, :-1], CY[:, -1]
    F = np.loadtxt(sglPrepDst + '%sF.csv' % filename, np.float32, delimiter=',')
    return C, y_num, F


# 处理多标签数据(无需生成arff文件)
# 从原始数据文件brsTr&brsTe.csv中获取基本输入信息C&X(连续)
def birds():
    cXTr = np.loadtxt(mulSrc + 'brsTr.csv', dtype=np.float32, delimiter=',', skiprows=1)
    cXTe = np.loadtxt(mulSrc + 'brsTe.csv', dtype=np.float32, delimiter=',', skiprows=1)
    C, X = np.r_[cXTr[:, :19], cXTe[:, :19]].astype(np.int8), np.r_[cXTr[:, 19:], cXTe[:, 19:]]
    F = standard(X)
    return C, F


# 从原始数据文件CAL500.csv中获取基本输入信息C&F(连续)
def CAL500():
    cX = np.loadtxt(mulSrc + 'CAL500.csv', dtype=np.float32, delimiter=',', skiprows=1)
    C, X = cX[:, :174].astype(np.int8), cX[:, 174:]
    F = standard(X)
    return C, F


# 从原始数据文件emtAl.csv(连续)中获取基本输入信息C&F
def emotions():
    cX = np.loadtxt(mulSrc + 'emtAl.csv', dtype=np.float32, delimiter=',', skiprows=1)
    C, X = cX[:, :6].astype(np.int8), cX[:, 6:]
    F = standard(X)
    return C, F


# 从原始数据文件ernAl.csv(离散)中获取基本输入信息C&F
def enron():
    cX = np.loadtxt(mulSrc + 'ernAl.csv', dtype=np.float32, delimiter=',', skiprows=1)
    C, X = cX[:, :53].astype(np.int8), cX[:, 53:]
    F = standard(X)
    return C, F


# 从原始数据文件gbsAl.csv(离散)中获取基本输入信息C&F
def genbase():
    cX = np.loadtxt(mulSrc + 'gbsAl.csv', dtype=np.str, delimiter=',', skiprows=1)
    C, xStr = cX[:, :27].astype(np.int8), cX[:, 28:]
    xStr[xStr == 'NO'] = '0'
    xStr[xStr == 'YES'] = '1'
    F = standard(xStr.astype(np.float32))
    return C, F


# 从原始数据文件mdcAl.csv(离散)中获取基本输入信息C&F
def medical():
    cX = np.loadtxt(mulSrc + 'mdcAl.csv', dtype=np.float32, delimiter=',', skiprows=1)
    C, X = cX[:, :45].astype(np.int8), cX[:, 45:]
    F = standard(X)
    return C, F


# 从原始数据文件scnAl.csv(连续)中获取基本输入信息C&F
def scene():
    cX = np.loadtxt(mulSrc + 'scnAl.csv', dtype=np.float32, delimiter=',', skiprows=1)
    C, X = cX[:, :6].astype(np.int8), cX[:, 6:]
    F = standard(X)
    return C, F


# 从原始数据文件ystAl.csv(连续)中获取基本输入信息C&F
def yeast():
    cX = np.loadtxt(mulSrc + 'ystAl.csv', dtype=np.float32, delimiter=',', skiprows=1)
    C, X = cX[:, :14].astype(np.int8), cX[:, 14:]
    F = standard(X)
    return C, F


# 获取多标签数据集数据(C & F)
def getCF(filename):
    if filename == 'brs':
        return birds()
    if filename == 'CAL500':
        return CAL500()
    if filename == 'emt':
        return emotions()
    if filename == 'ern':
        return enron()
    if filename == 'gbs':
        return genbase()
    if filename == 'mdc':
        return medical()
    if filename == 'scn':
        return scene()
    return yeast()


# 单标签
# 预处理数据
arff = True
# arff = False

# filename = arrhythmia(arff=arff)
# filename = bachCrlHrmy(arff=arff)
# filename = chess(arff=arff)
# filename = frtOdrTrm(arff=arff)
# filename = hillValley('hvy', arff=arff)
# filename = hillValley('hvyN', arff=arff)
# filename = ionosphere(arff=arff)
# filename = libMove(arff=arff)
# filename = madelon(arff=arff)
# filename = multFea(arff=arff)
# filename = musk(arff=arff)
# filename = optdigits(arff=arff)
# filename = oznLvlDet(arff=arff)
# filename = qsarBdg(arff=arff)
# filename = planRlx(arff=arff)
# filename = soybeanSmall(arff=arff)
# filename = soybeanLarge(arff=arff)
# filename = userKnowlde(arff=arff)
# filename = wineQuality('wqr', arff=arff)
# filename = wineQuality('wqw', arff=arff)

print('--------------------')

# 多标签(直接获取数据)
# C, F = birds()
# C, F = CAL500()
# C, F = emotions()
# C, F = enron()
# C, F = genbase()
# C, F = medical()
# C, F = scene()
# C, F = yeast()
