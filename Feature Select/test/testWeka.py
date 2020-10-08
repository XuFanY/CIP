import numpy as np
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.classes import Random
from weka.classifiers import Classifier
from weka.classifiers import Evaluation

sglArffSrc = 'D:/data/fs/output/single/'
sglWeka = 'D:/data/fs/output/single/weka/'


# fs(str){cip,fisher,laplace,reliefF,spec,trcRto}
# classifier(str){SMO,NB,K-NN}
def getPrecs(classifier, ks, filename, fss):
    # 建立分类器
    if classifier == 'SMO':
        cls = Classifier("weka.classifiers.functions.SMO")
    elif classifier == 'NB':
        cls = Classifier("weka.classifiers.bayes.NaiveBayes")
    elif classifier == 'K-NN':
        cls = Classifier("weka.classifiers.lazy.IBk")
    else:
        cls = Classifier()
    print('Finish building classifier!')
    loader = Loader("weka.core.converters.ArffLoader")
    precs = []
    for fs in fss:
        print('@%s' % fs)
        path = sglArffSrc + '%s/%s-' % (fs, filename)
        prec = []
        for k in ks:
            print('curK: %d' % k)
            # 加载arff文件,生成数据实例
            arffPath = path + '%s.arff' % k
            data = loader.load_file(arffPath)
            # 必须指明类标签对应的列,否则会报错
            data.class_index = data.num_attributes - 1
            # 使用分类器
            cls.build_classifier(data)
            # 用交叉验证评价分类器
            evlt = Evaluation(data)
            # crossvalidate_model(classifier, data, 交叉验证次数, ?)
            evlt.crossvalidate_model(cls, data, 10, Random(1))
            prec.append(evlt.percent_correct)
        precs.append(prec)
    return precs


def savePrecs(filename, classifiers, precses, fss):
    with open(sglWeka + '%sweka.csv' % filename, 'w') as wf:
        wf.write('@weka %s' % filename)
        buffs = ''
        n = len(classifiers)
        for i in range(n):
            buffs += '\n@%s' % classifiers[i]
            m = len(fss)
            precs = precses[i]
            for j in range(m):
                buffs += '\n@%s' % fss[j]
                for p in precs[j]:
                    buffs += ',%s' % p
        wf.write(buffs)
        wf.close()


minI = 1
maxK = 55
ms = [262, 14, 51, 100, 100, 33, 90,
      500, 649, 166, 62, 72, 12,
      41, 21, 35, 5, 11, 11]
sglFns = ['aya', 'bch', 'fot', 'hvy', 'hvyN', 'ise', 'lbm', 
          'mdl', 'mtf', 'musk', 'odt', 'old', 'plr', 
          'qbg', 'sbs', 'sbl', 'ukl', 'wqr', 'wqw']
fss = ['cip', 'cipm', 'fisher', 'laplace', 'reliefF', 'spec', 'trcRto']
classifiers = ['SMO', 'NB', 'K-NN']
n = len(sglFns)
jvm.start()
for i in range(minI, n):
    filename = sglFns[i]
    print('SINGL: %s weka start!' % filename)
    ks = range(1, min(maxK, ms[i]))
    precses = []
    for classifier in classifiers:
        print(classifier)
        precs = getPrecs(classifier, ks, filename, fss)
        for (fs, prec) in zip(fss, precs):
            idx = np.argsort(prec)[-1]
            print('Best (k,precision) for %s is (%s,%s)' % (fs, idx + 1, prec[idx]))
            print('Precs:\n', prec)
        precses.append(precs)
    savePrecs(filename, classifiers, precses, fss)
    print('End saving precses!')
jvm.stop()
