import numpy as np
from CIP import start, cip, evaluate
from dataDeal import getCYF, getCF, savearffs
from ofs import fisher_laplace, reliefF, spec, startTr, trace_ratio

sglRst = 'D:/data/fs/output/single/rst/'
mulRst = 'D:/data/fs/output/multi/rst/'


def saveRst(filename, rsts, multi=False):
    dst = sglRst
    fss = ['cip', 'cipm', 'fisher', 'laplace', 'reliefF', 'spec', 'trcRto']
    if multi:
        dst = mulRst
        fss = ['cip', 'cipm', 'laplace', 'spec', 'trcRto']
    n = len(fss)
    with open(dst + '%srst.csv' % filename, 'w') as wf:
        wf.write('@rst %s\n' % filename)
        for i in range(n):
            wf.write('@%s\n' % fss[i])
            rst = rsts[i]
            for ie in rst:
                m = len(ie)
                buffs = '%s' % ie[0]
                for j in range(1, m):
                    buffs += ',%s' % ie[j]
                wf.write(buffs + '\n')
        wf.close()


tr, fl = True, False
# 单标签数据信息
sglRuns = [tr, tr, tr, tr, tr,
           tr, tr, tr, tr, tr,
           tr, tr, tr, tr, tr,
           tr, tr, tr, tr]
sglFns = ['aya', 'bch', 'fot', 'hvy', 'hvyN',
          'ise', 'lbm', 'mdl', 'mtf', 'musk',
          'odt', 'old', 'plr', 'qbg', 'sbs',
          'sbl', 'ukl', 'wqr', 'wqw']
# 多标签数据信息
mulRuns = [tr, tr, tr, tr, tr,
           tr, tr, tr, tr]
mulFns = ['brs', 'CAL500', 'emt', 'ern',
          'gbs', 'mdc', 'scn', 'yst']
# print('--------------------')
minI = 0
maxK = 55
maxKs = []
indexses = []
# multi = True
multi = False
# 测试多标签数据
if multi:
    n = len(mulFns)
    for i in range(minI, n):
        if not mulRuns[i]:
            continue
        # 获取数据
        filename = mulFns[i]
        print('MULTI: %s fs start!' % filename)
        C, F = getCF(filename)
        maxKs.append(min(maxK, np.shape(F)[1]))
        if mulRuns[i]:
            continue
        indexs = []
        ks = range(1, min(maxK, np.shape(F)[1]))

        # 特征选取
        # 原cip方法
        print('From %s select %s features with cip!' % (filename, ks))
        idx0, A, A0, CT, Gamma, S = start(C, F, mutual=False, multi=True)
        for k in ks:
            t, p, index = cip(idx0, A, A0, CT, F, Gamma, k)
            indexs.append(index)
            print('select %d features(迭代次数: %d)!' % (k, t))
            print('p: \n', p)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='cip', multi=True, C=C)
        print('End CIP!')

        # 互信息cip方法
        indexs = []
        print('From %s select %s features with cipm!' % (filename, ks))
        idx0, A, A0, CT, Gamma, S = start(C, F, mutual=True, multi=True)
        for k in ks:
            t, p, index = cip(idx0, A, A0, CT, F, Gamma, k)
            indexs.append(index)
            print('select %d features(迭代次数: %d)!' % (k, t))
            print('p: \n', p)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='cipm', multi=True, C=C)
        print('End CIPM!')

        # laplace
        # 由于fisher score采用监督方法构建矩阵W,需要用到标签信息Lbls,因此多标签不能使用
        # laplacian score,参数score必须为laplace!!!(或除fisher之外的任意字符串)
        print('From %s select %s features with laplace!' % (filename, ks))
        indexs = fisher_laplace(F.copy(), ks, score='laplace')
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='laplace', multi=True, C=C)
        print('End LAPLACE!')

        # spec
        print('From %s select %s features with spec!' % (filename, ks))
        indexs = spec(F, ks)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='spec', multi=True, C=C)
        print('End SPEC!')

        # trace ratio,参数style必须为laplacian!!!(或除fisher之外的任意字符串)
        indexs = []
        sb, sw = startTr(F, style='laplacian')
        print('From %s select %s features with trcRto!' % (filename, ks))
        for k in ks:
            index = trace_ratio(k, sb, sw)
            indexs.append(index)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='trcRto', multi=True, C=C)
        print('End TRACE RATIO!')

        rsts = evaluate(C, F, indexses, multi=True, S=S)
        saveRst(filename, rsts, multi=True)
        indexses = []
# 测试单标签数据
else:
    n = len(sglFns)
    for i in range(minI, n):
        if not sglRuns[i]:
            continue
        # 获取数据
        filename = sglFns[i]
        print('SINGL: %s fs start!' % filename)
        C, y_num, F = getCYF(filename)
        # maxKs.append(min(maxK, np.shape(F)[1]))
        # if sglRuns[i]:
        #     continue
        indexs = []
        ks = range(1, min(maxK, np.shape(F)[1]))

        # 特征选取
        # 原cip方法
        print('From %s select %s features with cip!' % (filename, ks))
        idx0, A, A0, CT, Gamma = start(C, F, mutual=False, multi=False)
        for k in ks:
            t, p, index = cip(idx0, A, A0, CT, F, Gamma, k)
            indexs.append(index)
            print('select %d features(迭代次数: %d)!' % (k, t))
            print('p: \n', p)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='cip', multi=False, y_num=y_num)
        print('End CIP!')

        # 互信息cip方法
        indexs = []
        print('From %s select %s features with cipm!' % (filename, ks))
        idx0, A, A0, CT, Gamma = start(C, F, mutual=True, multi=False)
        for k in ks:
            t, p, index = cip(idx0, A, A0, CT, F, Gamma, k)
            indexs.append(index)
            print('select %d features(迭代次数: %d)!' % (k, t))
            print('p: \n', p)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='cipm', multi=False, y_num=y_num)
        print('End CIPM!')

        # fisher score
        print('From %s select %s features with fisher!' % (filename, ks))
        indexs = fisher_laplace(F, ks, score='fisher', Lbls=y_num)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='fisher', multi=False, y_num=y_num)
        print('End FISHER!')

        # laplacian score
        print('From %s select %s features with laplace!' % (filename, ks))
        indexs = fisher_laplace(F.copy(), ks, score='laplace')
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='laplace', multi=False, y_num=y_num)
        print('End LAPLACE!')

        # reliefF
        print('From %s select %s features with reliefF!' % (filename, ks))
        indexs = reliefF(F, ks, y_num)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='reliefF', multi=False, y_num=y_num)
        print('End RELIEFF!')

        # spec
        print('From %s select %s features with spec!' % (filename, ks))
        indexs = spec(F, ks)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='spec', multi=False, y_num=y_num)
        print('End SPEC!')

        # trace ratio
        indexs = []
        sb, sw = startTr(F, Lbls=y_num)
        print('From %s select %s features with trcRto!' % (filename, ks))
        for k in ks:
            index = trace_ratio(k, sb, sw)
            indexs.append(index)
        indexses.append(indexs)
        savearffs(F, indexs, filename, fs='trcRto', multi=False, y_num=y_num)
        print('End TRACE RATIO!')

        rsts = evaluate(C, F, indexses, multi=False)
        saveRst(filename, rsts, multi=False)
        indexses = []

print(maxKs)
