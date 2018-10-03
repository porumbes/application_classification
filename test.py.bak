from __future__ import print_function, division

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# --
# IO

data_vertex    = pd.read_csv('./data/georgiyData.Vertex.csv', skiprows=1, sep=' ', header=None)
pattern_vertex = pd.read_csv('./data/georgiyPattern.Vertex.csv', skiprows=1, sep=' ', header=None)

data_edges    = pd.read_csv('./data/georgiyData.Edges.csv', skiprows=1, sep=' ', header=None)
pattern_edges = pd.read_csv('./data/georgiyPattern.Edges.csv', skiprows=1, sep=' ', header=None)

assert (data_vertex[0] == data_vertex.index).all()
assert (pattern_vertex[0] == pattern_vertex.index).all()

data_vertex      = data_vertex.values[:,1:]
data_edges_table = data_edges[list(range(2, data_edges.shape[1]))].values
data_edges       = data_edges[[0, 1]].values

pattern_vertex      = pattern_vertex.values[:,1:]
pattern_edges_table = pattern_edges[list(range(2, pattern_edges.shape[1]))].values
pattern_edges       = pattern_edges[[0, 1]].values

num_dv = data_vertex.shape[0]
num_pv = pattern_vertex.shape[0]

# --
# Init

def normprob(x):
    return np.log(np.exp(x) / np.exp(x).sum(axis=0, keepdims=True))

def maxblock():
    pass

# Init_CV_MU
cv = cdist(data_vertex, pattern_vertex)
mu = -cv

cv = normprob(cv)
mu = normprob(mu)

vr = mu[:,pattern_edges[:,0]]
vf = mu[:,pattern_edges[:,1]]

ce = cdist(data_edges_table, pattern_edges_table)
re = normprob(-ce)
fe = normprob(-ce)
ce = normprob(ce)

cnull = np.sqrt((pattern_edges_table ** 2).sum(axis=-1))
cnull = np.maximum(cnull, ce.max(axis=0))
# >>
# cnull = normprob(cnull)
cnull = np.zeros(cnull.shape[0]) # bug in code?
# <<


vf_max = vf.max(axis=0)
vr_max = vr.max(axis=0)
fmax = np.tile(-cnull + vr_max, num_dv).reshape(num_dv, -1)
rmax = np.tile(-cnull + vf_max, num_dv).reshape(num_dv, -1)
for edge_idx, (src, dst) in enumerate(data_edges):
    fmax[dst] = np.maximum(fmax[dst], fe[edge_idx])
    rmax[src] = np.maximum(rmax[src], re[edge_idx])


for _ in range(num_pv):
    vf = mu[:,pattern_edges[:,1]] - fmax
    vr = mu[:,pattern_edges[:,0]] - rmax
    
    re = normprob(vf[data_edges[:,0]] - ce)
    fe = normprob(vr[data_edges[:,0]] - ce)
    
    vf_max = vf.max(axis=0)
    vr_max = vr.max(axis=0)
    fmax = np.tile(-cnull + vr_max, num_dv).reshape(num_dv, -1)
    rmax = np.tile(-cnull + vf_max, num_dv).reshape(num_dv, -1)
    for edge_idx, (src, dst) in enumerate(data_edges):
        rmax[src] = np.maximum(rmax[src], re[edge_idx])
        fmax[dst] = np.maximum(fmax[dst], fe[edge_idx])
        
    mu = -cv
    for edge_idx, (src, dst) in enumerate(pattern_edges):
        mu[:,dst] += fmax[:,edge_idx]
        mu[:,src] += rmax[:,edge_idx]
    
    mu = normprob(mu)


mu = np.hstack(mu)
np.savetxt('python_result', mu)
