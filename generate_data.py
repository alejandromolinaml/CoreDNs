import os

import numpy
from sklearn.model_selection import KFold

from mlutils import datasets
from mlutils.benchmarks import Chrono

numpy.random.seed(1337)


def getSamples(data, w=None, pct=10):
    if w is None:
        sampleIndexes = numpy.random.choice(data.shape[0], size=int(data.shape[0] * (pct / 100.0)), replace=False, )
    else:
        sampleIndexes = numpy.random.choice(data.shape[0], size=int(data.shape[0] * (pct / 100.0)), p=w, replace=False)

    sampleIndexes = numpy.sort(sampleIndexes)

    sampleWeights = None
    if w is not None:
        sampleWeights = w[sampleIndexes]
        sampleWeights = sampleWeights / numpy.sum(sampleWeights)

    return (sampleIndexes, sampleWeights)

def coreset(dataIn, pct=10):
    def getWeights(data):
        q, _ = numpy.linalg.qr(data)

        # get sample weights
        qnorm = numpy.linalg.norm(q, 2, 1)
        w = qnorm / numpy.sum(qnorm)
        return w

    (sampleIndexes, w) = getSamples(dataIn, getWeights(dataIn), pct)
    return sampleIndexes, w

def mkfolders(fname):
    os.makedirs(fname, exist_ok=True)
    return fname


def prepareExp(folder, fold, train, test, weights, family, base_time, cmds, fulltrain):

    prefix = folder + "fold_" + str(fold) + "_"
    trainfname = prefix + "train.npz"
    weightsfname = prefix + "weights.npz"
    testfname = prefix + "test.npz"
    coefffname = prefix + "coeff.mmp"
    predfname = prefix + "pred.mmp"
    timefname = prefix + "time.mmp"
    trainllfname = prefix + "trainll.mmp"
    testllfname = prefix + "testll.mmp"
    fulltrainllfname = prefix + "fulltrainll.mmp"
    fulltrainfname = prefix + "full_train.npz"

    numpy.savez_compressed(trainfname, train)
    numpy.savez_compressed(fulltrainfname, fulltrain)
    numpy.savez_compressed(testfname, test)

    weightscmd = ""
    if weights is not None:
        numpy.savez_compressed(weightsfname, weights)
        weightscmd = "--weights '%s'" % (weightsfname)


    numpy.memmap(coefffname, dtype='float32', mode='w+', shape=(train.shape[1], train.shape[1] + 1)).flush()
    numpy.memmap(predfname, dtype='float32', mode='w+', shape=(test.shape[0], test.shape[1])).flush()
    numpy.memmap(trainllfname, dtype='float32', mode='w+', shape=(train.shape[0], train.shape[1])).flush()
    numpy.memmap(testllfname, dtype='float32', mode='w+', shape=(test.shape[0], test.shape[1])).flush()
    numpy.memmap(fulltrainllfname, dtype='float32', mode='w+', shape=(fulltrain.shape[0], fulltrain.shape[1])).flush()

    timetable = numpy.memmap(timefname, dtype='float32', mode='w+', shape=(train.shape[1], 1))
    timetable += base_time
    timetable.flush()

    for feature_idx in range(train.shape[1]):

        cmds.append(
            "python3 learnDN.py --train '%s' --full_train '%s' --test '%s' %s --family '%s' --feature '%s' --out_coeff '%s' --out_preds '%s' --out_time '%s' --out_train_loglikelihood '%s' --out_test_loglikelihood '%s' --out_full_train_loglikelihood '%s'" % (
            trainfname, fulltrainfname, testfname, weightscmd, family, feature_idx, coefffname, predfname, timefname, trainllfname, testllfname, fulltrainllfname))


def generateInput(dsname,  data, family):
    data = data[numpy.var(data, 1) > 0, :]

    fullcmd = []
    cscmd = []
    rndcmd = []

    for fold, (train_index, test_index) in enumerate(KFold(n_splits=10, random_state=1).split(data)):

        train, test = data[train_index, :], data[test_index, :]

        prepareExp(mkfolders("data/" + dsname + "/full/"), fold, train, test, None, family, 0.0, fullcmd, train)

        for pct in [5, 10, 20, 30, 40]:
            cschrono = Chrono().start()
            sampleIndexes, csweight = coreset(train, pct)
            cstrain = train[sampleIndexes, :]
            cschrono.end()

            prepareExp(mkfolders("data/" + dsname + "/cs/" + str(pct) + "/"), fold, cstrain, test, csweight, family, cschrono.elapsed(), cscmd, train)


            rndchrono = Chrono().start()
            sampleIndexes, _ = getSamples(train, None, pct)
            rtrain = train[sampleIndexes, :]
            rndchrono.end()

            prepareExp(mkfolders("data/" + dsname + "/rnd/" + str(pct) + "/"), fold, rtrain, test, None, family, rndchrono.elapsed(), rndcmd, train)


    allcmds = []
    allcmds.extend(fullcmd)
    allcmds.extend(cscmd)
    allcmds.extend(rndcmd)

    with open("cmds_"+dsname+".txt", "w") as text_file:
        text_file.write("\n".join(allcmds))


(dsname, data, features, target) = datasets.getMNIST()
print(dsname)
generateInput(dsname, data, "gaussian")

#(dsname, data, features, times, hours) = datasets.getTrafficLarge()
#print(dsname)
#generateInput(dsname, data, "poisson")