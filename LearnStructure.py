'''
Created on 7 Sep 2017

@author: alejomc
'''


import argparse
import os
import platform
import time
if platform.system() == 'Darwin':
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"


from joblib.memory import Memory
import numpy
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

from mlutils import datasets
import networkx as nx
import json
import re


numpy2ri.activate()
robjects.r["library"]("glm2")


memory = Memory(cachedir="structure", verbose=0, compress=9)


def getSamples(data, w=None, pct=10):
    if w is None:
        sampleIndexes = numpy.random.choice(data.shape[0], size=int(data.shape[0] * (pct / 100.0)), replace=False,)
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

def getAllGraphs(data):
    graphs = []
    for pct in [40, 70, 100]:
    #for pct in [100]:
        numpy.random.seed(4442)
        csweight = None
        cstrain = numpy.copy(data)
        utrain = numpy.copy(data)
        if pct < 100:
            sampleIndexes, csweight = coreset(data, pct)
            cstrain = data[sampleIndexes, :]
            usampleIndexes, _ = getSamples(data, w=None, pct=pct)
            utrain = data[usampleIndexes, :]
            
        graphs.append((pct, 
                       getGraph(cstrain, csweight, pct, family="poisson", prep="x+1"), 
                       getGraph(utrain, None, pct, family="poisson", prep="x+1"),
                       getGraph(cstrain, csweight, pct, family="gaussian", prep="log(x+1)"), 
                       getGraph(utrain, None, pct, family="gaussian", prep="log(x+1)"),
                       
                       ))
        # 0/0
    return graphs


def learnGlm(train, weights, feature, family="gaussian", prep="log(x+1)"):
    #print(train)
    #train = train + numpy.reshape(numpy.abs(numpy.random.normal(0, 1, train.shape[0] * train.shape[1])), train.shape)
    
    #train = train + numpy.abs(numpy.random.normal(0, 0.1, 100))
    #0 / 0
    #feature = 0
    #train = numpy.copy(train)
     
    #train[:, feature] = numpy.log(train[:, feature])
    print(prep, family)
    
    if prep == "log(x+1)":
        train = numpy.log(train+1.0)
    if prep == "x+1":
        #pass
        train = train + 1
    
    #print(train)
    
    traindf = robjects.r["as.data.frame"](train)

    robjects.r["set.seed"](421)
 
    if weights is not None:
        glm = robjects.r["glm2"](data=traindf, family=family, formula=robjects.r["as.formula"]("V%s ~ . " % (feature + 1)), maxit=1000, weights=weights)
    else:
        glm = robjects.r["glm"](data=traindf, family=family, formula=robjects.r["as.formula"]("V%s ~ . " % (feature + 1)), maxit=1000)

    #print(glm)
    
    coeffs = numpy.asarray(glm[0])
    
    coeffs = numpy.insert(coeffs[1:], feature, [0])
    #coeffs = numpy.insert(coeffs, feature, [0])
    # print(coeffs)
    
    #print(coeffs)
    #0/0
    return coeffs

#@memory.cache
def getGraph(cstrain, csweight, pct, family="gaussian", prep="log(x+1)"):
    ncols = cstrain.shape[1]
    result = numpy.zeros((ncols, ncols))
    for c in range(ncols):
        print(pct, c)
        coefs = learnGlm(cstrain, csweight, c, family=family, prep=prep)
        
        result[c, :] = coefs
    # print(result)
    
    return result

#@memory.cache
def maxunigraph(g):
    gabs = numpy.abs(g)
    # gabs=g
    gmax = numpy.zeros_like(gabs)
    
    #return gabs
    for i in range(gabs.shape[0]):
        for j in range(gabs.shape[0]):
            gmax[i, j] = max(gabs[i, j], gabs[j, i])
    return gmax

        
def getGraphFromAdj(adj):
    #print(numpy.any(numpy.isnan(adj)))
    
    G = nx.DiGraph()
    G = nx.from_numpy_matrix(filteradj(adj), create_using=G)
    
    labels = {}
    for i, w in enumerate(words):
        labels[i] = w
    G = nx.relabel_nodes(G, labels)
    
    deg = G.degree()
    to_keep = [n for n in deg if deg[n] > 0]
    G = G.subgraph(to_keep)
    
    return G

def saveGraph(adj, filename):
    adj = filteradj(adj)
    G = getGraphFromAdj(adj)
    
    with open('nips_layout.graphml', 'r') as myfile:
        layout=myfile.read()
        
    renodes = re.compile(r'<node id="([^"]+)">')
    
    existingNodes = renodes.findall(layout)
    
    edgesStr = ""
    i = 0
    for e in G.edges():
        if e[0] not in existingNodes or e[1] not in existingNodes:
            continue
        
        edgesStr += """
        <edge id="%s" source="%s" target="%s">
        <data key="weight">%s</data>
        </edge>
    """% (i, e[0], e[1], G[e[0]][e[1]]["weight"])
    
        i+=1
    
    with open(filename, "w") as text_file:
        text_file.write(layout % (edgesStr))


def filteradj(adj):
    adj[adj < 0.02] = 0
    #adj[adj < 0.052] = 0
    return adj 
    
    
(dsname, data, words) = datasets.getNips()

graphs = getAllGraphs(data)


fullPDN = graphs[-1][1]
fullGDN = graphs[-1][3]

saveGraph(fullPDN, "structure_graphs/PDN_100.graphml")
saveGraph(fullGDN, "structure_graphs/GDN_100.graphml")

def computeDistances(a, gold, th=0.0):
    r = numpy.linalg.norm(a - gold, "fro")
    return numpy.round(r, 4)

    d1 = numpy.linalg.norm((a > th) - (gold > th), "fro")
    d2 = numpy.sum(numpy.abs((a > th) - (gold > th)))
    d3 = numpy.linalg.norm(numpy.abs(a - gold), "fro")
    d4 = numpy.sum(numpy.abs(a - gold))
    d5 = numpy.sum(numpy.abs( numpy.abs(a) - numpy.abs(gold) ))
    
    return numpy.round([d1, d2, d3, d4, d5], 3)
    
for pct, pcdn, pudn, gcdn, gudn in graphs:
    pcdn[numpy.isnan(pcdn)] = 0.0
    pudn[numpy.isnan(pudn)] = 0.0
    gcdn[numpy.isnan(gcdn)] = 0.0
    gudn[numpy.isnan(gudn)] = 0.0
    
    if pct < 100:
        saveGraph(pcdn, "structure_graphs/PCDN_%s.graphml" % (pct))
        #saveGraph(pudn, "structure_graphs/PUDN_%s.graphml" % (pct))
        saveGraph(gcdn, "structure_graphs/GCDN_%s.graphml" % (pct))
        #saveGraph(gudn, "structure_graphs/GUDN_%s.graphml" % (pct))
    

    isnan = numpy.any(numpy.isnan(pcdn)) or numpy.any(numpy.isnan(pudn)) or numpy.any(numpy.isnan(gcdn)) or numpy.any(numpy.isnan(gudn))
    print(pct, isnan, 
          computeDistances(pcdn, fullPDN), 
          computeDistances(pudn, fullPDN), 
          computeDistances(gcdn, fullGDN), 
          computeDistances(gudn, fullGDN))




