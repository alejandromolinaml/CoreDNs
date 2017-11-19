import os
import platform


if platform.system() == 'Darwin':
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"


import argparse
import numpy
import time

from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

numpy2ri.activate()
robjects.r["library"]("glm2")


class Chrono(object):
    def __init__(self):
        self.started = 0
        self.finished = 0

    def start(self):
        self.started = time.perf_counter()
        return self

    def end(self):
        self.finished = time.perf_counter()
        return self

    def elapsed(self):
        return self.finished - self.started


parser = argparse.ArgumentParser()
parser.add_argument('--train')
parser.add_argument('--test')
parser.add_argument('--weights')
parser.add_argument('--family')
parser.add_argument('--feature', type=int)
parser.add_argument('--out_coeff')
parser.add_argument('--out_preds')
parser.add_argument('--out_time')
parser.add_argument('--out_train_loglikelihood')
parser.add_argument('--out_test_loglikelihood')
parser.add_argument('--out_full_train_loglikelihood')
parser.add_argument('--full_train')



args = parser.parse_args()

train = numpy.load(args.train)['arr_0']
test = numpy.load(args.test)['arr_0']
full_train = numpy.load(args.full_train)['arr_0']

weights = None
if args.weights is not None:
    weights = numpy.load(args.weights)['arr_0']

family = args.family
feature = args.feature
out_coeff = numpy.memmap(args.out_coeff, dtype='float32', mode='r+', shape=(train.shape[1], train.shape[1] + 1))
out_preds = numpy.memmap(args.out_preds, dtype='float32', mode='r+', shape=(test.shape[0], test.shape[1]))
out_time = numpy.memmap(args.out_time, dtype='float32', mode='r+', shape=(train.shape[1], 1))
out_train_loglikelihood = numpy.memmap(args.out_train_loglikelihood, dtype='float32', mode='r+', shape=(train.shape[0], train.shape[1]))
out_test_loglikelihood = numpy.memmap(args.out_test_loglikelihood, dtype='float32', mode='r+', shape=(test.shape[0], test.shape[1]))
out_full_train_loglikelihood = numpy.memmap(args.out_full_train_loglikelihood, dtype='float32', mode='r+', shape=(full_train.shape[0], full_train.shape[1]))


def learn(train):
    traindf = robjects.r["as.data.frame"](train)

    start = robjects.r["rep"](numpy.max(train)/train.shape[1], train.shape[1])

    robjects.r["set.seed"](1337)
    try:
        #0/0
        t = Chrono().start()
        if weights is not None:
            glm = robjects.r["glm"](data=traindf, family=family, formula=robjects.r["as.formula"]("V%s ~ ." % (feature+1)), maxit = 100, weights=weights)
        else:
            glm = robjects.r["glm"](data=traindf, family=family, formula=robjects.r["as.formula"]("V%s ~ ." % (feature+1)), maxit = 100)

        traintime = t.end().elapsed()
    except:
        #print("WHAAAAAA")
        numpy.random.seed(1337)
        rn = numpy.random.standard_normal(size=train.shape)
        rn = rn + numpy.abs(numpy.min(rn))
        data = train + rn

        df = robjects.r["as.data.frame"](data)
        t = Chrono().start()
        if weights is not None:
            glm = robjects.r["glm"](data=traindf, family=family, formula=robjects.r["as.formula"]("V%s ~ ." % (feature+1)), maxit = 100, weights=weights)
        else:
            glm = robjects.r["glm"](data=traindf, family=family, formula=robjects.r["as.formula"]("V%s ~ ." % (feature+1)), maxit = 100)
        traintime = t.end().elapsed()

        #print(glm)

    coeffs = numpy.asarray(glm[0])
    coeffs = numpy.insert(coeffs, feature, [0])
    #print(coeffs)
    return glm, coeffs, traintime

glm, coeffs, traintime = learn(train)

out_time[feature] += traintime

testdf = robjects.r["as.data.frame"](test)
traindf = robjects.r["as.data.frame"](train)
fulltraindf = robjects.r["as.data.frame"](full_train)

preds = robjects.r["predict"](glm, testdf, type="response")

predsTrain = robjects.r["predict"](glm, traindf, type="response")

predsFullTrain = robjects.r["predict"](glm, fulltraindf, type="response")

out_preds[:, feature] = preds

out_coeff[feature, :] = coeffs

densityfunc = "dnorm"

if family == "poisson":
    densityfunc = "dpois"

out_full_train_loglikelihood[:, feature] = robjects.r[densityfunc](full_train[:, feature], predsFullTrain, log = True)
out_train_loglikelihood[:, feature] = robjects.r[densityfunc](train[:, feature], predsTrain, log = True)
out_test_loglikelihood[:, feature] = robjects.r[densityfunc](test[:, feature], preds, log = True)


out_train_loglikelihood.flush()
out_test_loglikelihood.flush()
out_full_train_loglikelihood.flush()
out_preds.flush()
out_coeff.flush()
out_time.flush()

#print(out_preds)