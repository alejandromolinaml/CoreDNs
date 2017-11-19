import argparse
import numpy

import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument('--train')
parser.add_argument('--test')
parser.add_argument('--weights')
parser.add_argument('--family')
parser.add_argument('--feature', type=int)
parser.add_argument('--out_coeff')
parser.add_argument('--out_preds')
parser.add_argument('--out_time')

args = parser.parse_args()

train = numpy.load(args.train)['arr_0']
test = numpy.load(args.test)['arr_0']
weights = None
if args.weights is not None:
    weights = numpy.load(args.weights)['arr_0']

family = args.family
feature = args.feature
out_coeff = numpy.memmap(args.out_coeff, dtype='float32', mode='w+', shape=(train.shape[1], train.shape[1] + 1))
out_preds = numpy.memmap(args.out_preds, dtype='float32', mode='w+', shape=(test.shape[0], test.shape[1]))
out_time = numpy.memmap(args.out_time, dtype='float32', mode='w+', shape=(train.shape[1], 1))






print(numpy.sum(numpy.var(train,1)==0))
print(numpy.sum(numpy.var(test,1)==0))

dataIn = numpy.c_[train, numpy.ones((train.shape[0]))]

assert numpy.all(dataIn[:, -1] == 1), "missing intercept in data"

indepFeatures = list(range(dataIn.shape[1]))  # intercept at the last column
indepFeatures.remove(feature)


if family == "gaussian":
    fam = sm.families.Gaussian()
elif family == 'poisson':
    fam = sm.families.Poisson()

if weights is None:
    glm = sm.GLM(dataIn[:, feature], dataIn[:, indepFeatures], family=fam)
else:
    glm = sm.GLM(dataIn[:, feature], dataIn[:, indepFeatures], family=fam, freq_weights=weights)

glmcoeff = glm.fit(method="bfgs").params
# print(glmcoeff, glmcoeff.shape)
out_coeff[feature, :] = glmcoeff
# print(glmcoeff)

dataTest = numpy.c_[test, numpy.ones((test.shape[0]))]

out_preds[:, feature] = sm.GLM.predict(glmcoeff, dataTest[:, indepFeatures])
out_preds.flush()
out_coeff.flush()


print(out_preds)