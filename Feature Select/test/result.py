import numpy as np

sglRst = 'D:/data/fs/output/single/rst/'
mulRst = 'D:/data/fs/output/multi/rst/'
sglWeka = 'D:/data/fs/output/single/weka/'
mulMeka = 'D:/data/fs/output/multi/Meka/'
rstPath = 'D:/data/fs/rst/'

# 单标签数据信息
sglFns = ['aya', 'bch', 'fot', 'hvy', 'hvyN',
          'ise', 'lbm', 'mdl', 'mtf', 'musk',
          'odt', 'old', 'plr', 'qbg', 'sbs',
          'sbl', 'ukl', 'wqr', 'wqw']
sglKs = [55, 14, 51, 55, 55, 33, 55, 55, 55, 55, 55, 55, 12, 41, 21, 35, 5, 11, 11]
# 多标签数据信息
mulFns = ['brs', 'CAL500', 'emt', 'ern',
          'gbs', 'mdc', 'scn', 'yst']


# sgl:CR(F),特征子集的平均分类冗余性(越小越好)
# mul:CC(F),特征子集的平均残留类间相关性(越小越好)
def getRst(multi=False):
    if multi:
        n = len(mulFns)
        cc_mat = np.zeros((n, 5))
        for j in range(n):
            print('GET MULTI rst @%s' % mulFns[j])
            with open(mulRst + '%srst.csv' % mulFns[j], 'r') as rf:
                lines = rf.readlines()
                for i in range(5):
                    cc = 0.0
                    for k in range(53):
                        # print(lines[55 * i + k + 3])
                        ie = lines[55 * i + k + 3].split(',')
                        cc += float(ie[k + 2])
                        # print(ie[k + 2])
                    cc_mat[j][i] = cc / 53
        print('cc @rst:\n', cc_mat)
        with open(rstPath + 'mulCC.csv', 'w') as wf:
            wf.write('data\\cc\\fs,cip,cipm,laplace,spec,trcRto\n')
            for j in range(n):
                buffs = '%s' % mulFns[j]
                for i in range(5):
                    buffs += ',%s' % cc_mat[j][i]
                wf.write('%s\n' % buffs)
    else:
        n = len(sglFns)
        cr_mat, red_mat = np.zeros((n, 7)), np.zeros((n, 7))
        for j in range(n):
            print('GET SINGLE rst @%s' % sglFns[j])
            crs, reds = [], []
            with open(sglRst + '%srst.csv' % sglFns[j], 'r') as rf:
                lines = rf.readlines()
                m, maxK = len(lines), sglKs[j]
                for i in range(3, m, maxK):
                    cr, red = 0.0, 0.0
                    for k in range(maxK - 2):
                        # print(lines[i + k])
                        ie = lines[i + k].split(',')
                        cr += float(ie[k + 2])
                        red += float(ie[k + 3])
                        # print(ie[k + 2], ie[k + 3])
                    crs.append(cr / (maxK - 2))
                    reds.append(red / (maxK - 2))
            cr_mat[j], red_mat[j] = crs, reds
        print('cr @rst:\n', cr_mat)
        print('red @rst:\n', red_mat)
        with open(rstPath + 'sglCR.csv', 'w') as wf:
            wf.write('data\\cr\\fs,cip,cipm,fisher,laplace,reliefF,spec,trcRto\n')
            for j in range(n):
                buffs = '%s' % sglFns[j]
                for i in range(7):
                    buffs += ',%s' % cr_mat[j][i]
                wf.write('%s\n' % buffs)


# 最高分类精度
def getWeka():
    n = len(sglFns)
    precs = [np.zeros((n, 7)), np.zeros((n, 7)), np.zeros((n, 7))]
    for j in range(n):
        print('GET weka @%s' % sglFns[j])
        with open(sglWeka + '%sweka.csv' % sglFns[j], 'r') as rf:
            lines = rf.readlines()
            for i in range(3):
                tmp = np.zeros((7, sglKs[j] - 1))
                for k in range(7):
                    tmp[k] = lines[8 * i + k + 2].split(',')[1:]
                precs[i][j] = np.max(tmp, axis=1)
    classifiers = ['SMO', 'NB', 'KNN']
    for (cls, prec) in zip(classifiers, precs) :
        print('%s:\n' % cls, prec)
        with open(rstPath + 'sglWeka_%s.csv' % cls, 'w') as wf:
            wf.write('data\\prec\\fs,cip,cipm,fisher,laplace,reliefF,spec,trcRto\n')
            for j in range(n):
                buffs = '%s' % sglFns[j]
                for k in range(7):
                    buffs += ',%s' % prec[j][k]
                wf.write('%s\n' % buffs)


# 平均分类精度(越大越好)
# 平均Hamming Loss(越小越好)
# mean±var,avg
def getMeka():
    n = len(mulFns)
    evas_means, evas_stds = [], []
    for j in range(n):
        print('GET Meka @%s' % mulFns[j])
        with open(mulMeka + '%sMeka.txt' % mulFns[j], 'r') as rf:
            evas_mean, evas_std = np.zeros((5, 6)), np.zeros((5, 6))
            lines = rf.readlines()
            for i in range(5):
                evas, tmp = np.zeros((54, 6)), np.zeros((10, 6))
                for k in range(54):
                    for fold in range(10):
                        tmp[fold] = lines[fold + 541 * i + 10 * k + 3].split(',')
                    evas[k] = np.mean(tmp, axis=0)
                evas_mean[i] = np.mean(evas, axis=0)
                evas_std[i] = np.std(evas, axis=0)
            evas_means.append(evas_mean)
            evas_stds.append(evas_std)
    eva_nms = ['rankLoss', 'avgPrec', 'jcdDist', 'zeroLoss,', 'accuracy', 'hammingLoss']
    for k in range(6):
        with open(rstPath + 'mulMeka_%s.csv' % eva_nms[k], 'w') as wf:
            wf.write('data\\%s\\fs,cip,cipm,laplace,spec,trcRto\n' % eva_nms[k])
            for j in range(n):
                buffs = '%s' % mulFns[j]
                for i in range(5):
                    buffs += ',%f±%f' % (evas_means[j][i][k], evas_stds[j][i][k])
                wf.write('%s\n' % buffs)


if __name__ == '__main__':
    tr, fl = True, False
    sgl, mul = fl, tr
    # 单标签
    if sgl:
        weka, rst = tr, tr
        if weka:
            getWeka()
        if rst:
            getRst(multi=False)
    # 多标签
    if mul:
        meka, rst = fl, tr
        if meka:
            getMeka()
        if rst:
            getRst(multi=True)
