import numpy as np
from numpy import linalg as LA
import networkx as nx
from networkx import normalized_laplacian_matrix
import math    
import ot
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity, BallTree
from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve, GridSearchCV
from scipy.stats import iqr
from scipy.linalg import eigh
from itertools import groupby
import pandas as pd
import random
from scipy.io import loadmat
import os.path
import sys

kn = 30
nhop = int(sys.argv[3])
iteration = int(sys.argv[2])
dataset = sys.argv[1]

def makeGraph(G,A,B,start):
    for i in range(0,A.shape[0]):
        p = A[i]
        q = B[i]
        if start==0:
            G[p,q] = 1
            G[q,p] = 1
        elif start==1:
            G[p-1,q-1] = 1
            G[q-1,p-1] = 1
    return G

def knn(n,ids,G):
    que = []
    que.append(n)
    que.extend(ids[0])
    count=1
    while len(que)<kn:
        node = que[count]
        row = G[node,:]
        nhood = np.nonzero(row)
        ngh=[]
        for nh in nhood[0]:
            if nh not in que:
                ngh.append(nh)
        que.extend(ngh)
        count = count+1
    return que

def khop(n,ids,G):
    que = []
    que.append([n])
    que.append(ids[0])
    count=1
    hop=1
    while hop<nhop:
        allngh=[]
        for nh in que[hop]:
            row = G[nh,:]
            nhood = np.nonzero(row)
            if len(nhood[0])>0:
                ngh=[]
                for nh in nhood[0]:
                    if nh not in que[hop] and nh not in allngh:
                        ngh.append(nh)
                allngh.extend(ngh)
        if len(allngh)>0:
            que.append(allngh)
        hop=hop+1
    nodes = [j for i in que for j in i]
    nodes = list(set(nodes))
    return nodes


def subGraph(nodes,G):
    subG = G[np.ix_(nodes,nodes)]
    return subG

def Spectrum(G):
    A = np.matrix(G)
    G = nx.from_numpy_matrix(A)
    N = normalized_laplacian_matrix(G)
    b, v = eigh(N.todense(),eigvals=(N.shape[0]-1,N.shape[0]-1))
    a, v = eigh(N.todense(),eigvals=(0,0))
    L = N.todense()
    I = np.identity(L.shape[0])
    Ns = (L-((b+a)/2)*I)/((b-a)/2)
    w, v = eigh(Ns)    
    return w,v

def Binning(EIG,binsize,E):
    mat = np.empty([0,binsize])
    for i in range(len(EIG)):
        row = EIG[i]
        vec = np.histogram(row, bins = binsize, range=(np.array(E).min(),np.array(E).max()))
        mat = np.concatenate((mat,np.array([vec[0]])),axis=0)
    return mat

def HistSize(EIG,N):
    IQR = iqr(EIG)
    h = (2*IQR)/(N**(1./3))
    binsize = (np.array(EIG).max()-np.array(EIG).min())/h
    return int(round(binsize))
#'''
def mydist(x, y):
    n = x.shape[0]
    M = ot.dist(np.array([x]), np.array([y]))
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    G = ot.emd2(a, b, M)
    return G
#'''
'''
def mydist(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
'''
'''
def KLD(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
 
def mydist(a, b):
    m = (a+b)/2
    kl_pm = KLD(a, m)
    kl_qm = KLD(b, m)
    #a = np.asarray(a, dtype=np.float)
    #b = np.asarray(b, dtype=np.float)
    js = (kl_pm + kl_qm)/2
    return js
    #return np.sum(np.where(a != 0, a * np.log(a / b), 0))

'''

def metropolis(G,data,i,Closest,iter=iteration):
    prob = 0.1
    knr=10
    ii = i
    idxs = np.nonzero(G[ii,:])
    if len(idxs)>1:
        idxs = idxs[1]
    else:
        idxs=idxs[0]
    ball = data[idxs,:]
    d_x = len(idxs)
    x = data[i,:]
    x = np.array([x])
    samples = np.zeros((iter, data.shape[1]))
    RW = np.zeros((iter,),dtype=int)
    samples[0]=np.array([x])
    RW[0] = ii   	
    for i in range(1,iter):
        pick = np.random.rand()
        if  pick <= prob:
            ngh = [int(item) for item in Closest[int(ii)]]
            num = random.sample(set(ngh),1)
        elif pick > prob:
            num = random.sample(set(np.arange(d_x)),1)
        x_prime = ball[num,:]
        x = x_prime
        ii = idxs[num]
        idxs = np.nonzero(G[ii,:])
        if len(idxs)>1:
            idxs = idxs[1]
        else:
            idxs=idxs[0]
        ball = data[idxs,:]
        d_x = len(idxs)
        samples[i] = np.array([x])
        RW[i] = ii   	
    return samples,RW


def RemDups(samples):
    Uni_samples = [x[0] for x in groupby(samples)]
    return Uni_samples
NODES = set()
def parse_line(line):
    line = line.strip().split()
    e1, e2 = line[0].strip(), line[1].strip()
    return e1, e2

mat = np.loadtxt('../data/Splits/'+str(dataset)+'/90/train_pos.txt')
A = mat[:,0]
B = mat[:,1]
A = A.astype(int)
B = B.astype(int)
N = np.union1d(A,B)
start=min(N)
adj = np.zeros((N.shape[0],N.shape[0]),dtype = int)
G = makeGraph(adj,A,B,start)
deg = G.sum(axis=1)
EIG=[]
alleig=[]
for i in range(0,N.shape[0]):
    row = G[i,:]
    ids = np.nonzero(row)
    nodes = khop(i,ids,G)
    subG = subGraph(nodes,G)
    eigvals,eigvecs = Spectrum(subG)
    eigs = np.sort(eigvals)
    eigvals = eigs[::-1]
    EIG.append(eigvals)
    alleig.extend(eigvals)
binsize = HistSize(alleig,N.shape[0])
mat = Binning(EIG,binsize,alleig)

RWalks = []
knr=10
Closest=[]
for i in range(0,mat.shape[0]):
    idxs = np.nonzero(G[i,:])
    if len(idxs)>1:
        idxs = idxs[1]
    else:
        idxs=idxs[0]
    d_x = len(idxs)
    ball = mat[idxs,:]
    tree = BallTree(ball,metric='pyfunc', func=mydist)
    x = mat[i,:]
    x = np.array([x])
    if d_x < knr:
        topk = d_x
    else:
        topk = knr
    dist, ind = tree.query(x, k=topk)
    Closest.append(list(ind[0]))
for i in range(0,mat.shape[0]):
    for j in range(50):
        samples,RW = metropolis(G,mat,i,Closest,iter=iteration)
        walk = RemDups(RW)
        RWalks.append(walk)

with open('walks/'+str(dataset)+'_SW_50_4.txt', 'w') as f:
    for _list in RWalks:
        for _string in _list:
            f.write(str(_string) + ' ')
        f.write('\n')
