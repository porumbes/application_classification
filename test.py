#!/bin/bash

"""
    test.py
"""

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

num_dv   = data_vertex.shape[0]
num_pv   = pattern_vertex.shape[0]

num_de   = data_edges.shape[0]
num_pe   = pattern_edges.shape[0]

edge_dim = pattern_edges.shape[1]

print({
    "num_dv" : num_dv,
    "num_pv" : num_pv,

    "num_de" : num_de,
    "num_pe" : num_pe,
})

# --
# Init

def normprob(x):
    x = (x - x.max(axis=0, keepdims=True)).copy()
    return np.log(np.exp(x) / np.exp(x).sum(axis=0, keepdims=True))

def l2_norm(x):
    return np.sqrt((x ** 2).sum())


# --
# Vertex similarity

cv = cdist(data_vertex, pattern_vertex)  # num_dv x num_pv

mu = normprob(-cv) # num_dv x num_pv
cv = normprob(cv)  # num_dv x num_pv

v_fwd_max = np.zeros(num_pe) # num_dv x num_pe
v_bak_max = np.zeros(num_pe) # num_dv x num_pe
mu_max = mu.max(axis=0)

for i, (src, dst) in enumerate(pattern_edges):
    v_bak_max[i] = mu_max[src]
    v_fwd_max[i] = mu_max[dst]

# --
# Edge similarity

ce = cdist(data_edges_table, pattern_edges_table) # num_de x num_pe
xe = normprob(-ce) # num_de x num_pe
ce = normprob(ce)  # num_de x num_pe

# --
# Combine

# >>
# cnull = np.sqrt((pattern_edges_table ** 2).sum(axis=-1))
# cnull = np.maximum(cnull, ce.max(axis=0))
# cnull = normprob(cnull)
# --
cnull = np.zeros(num_pe) # bug in code?
# <<

fwd_max = np.zeros((num_dv, num_pe))
bak_max = np.zeros((num_dv, num_pe))

fwd_touched = set([])
bak_touched = set([])
for edge_idx, (src, dst) in enumerate(data_edges):
    if dst not in fwd_touched:
        fwd_max[dst] = np.maximum(v_bak_max, xe[edge_idx])
        fwd_touched.add(dst)
    else:
        fwd_max[dst] = np.maximum(fwd_max[dst], xe[edge_idx])
    
    if src not in bak_touched:
        bak_max[src] = np.maximum(v_fwd_max, xe[edge_idx])
        bak_touched.add(src)
    else:
        bak_max[src] = np.maximum(bak_max[src], xe[edge_idx])


v_fwd = np.zeros((num_dv, num_pe)) # num_dv x num_pe
v_bak = np.zeros((num_dv, num_pe)) # num_dv x num_pe
for _ in range(num_pv):
    
    for p_edge_idx, (src, dst) in enumerate(pattern_edges):
        v_fwd[:,p_edge_idx] = mu[:,dst] - fwd_max[:,p_edge_idx]
        v_bak[:,p_edge_idx] = mu[:,src] - bak_max[:,p_edge_idx]
        
    v_fwd_max = v_fwd.max(axis=0)
    v_bak_max = v_bak.max(axis=0)
    
    e_bak = v_fwd[data_edges[:,0]] - ce
    e_fwd = v_bak[data_edges[:,0]] - ce
    e_bak_norm = np.log(np.exp(e_bak).sum(axis=0, keepdims=True))
    e_fwd_norm = np.log(np.exp(e_fwd).sum(axis=0, keepdims=True))
    
    fwd_max = np.zeros((num_dv, num_pe)) - np.inf # num_dv x num_pe
    bak_max = np.zeros((num_dv, num_pe)) - np.inf # num_dv x num_pe
    
    sel = np.argsort(data_edges[:,0],  kind='mergesort')
    for d_edge_idx, (src, dst) in enumerate(data_edges[sel]):
        bak_max[src] = np.maximum(bak_max[src], e_bak[d_edge_idx])
    
    for d_edge_idx, (src, dst) in enumerate(data_edges[sel]):
        fwd_max[dst] = np.maximum(fwd_max[dst], e_fwd[d_edge_idx])
    
    fwd_max -= e_fwd_norm
    bak_max -= e_bak_norm
    
    fwd_max = np.maximum(fwd_max, (v_bak_max - cnull).reshape(1, -1))
    bak_max = np.maximum(bak_max, (v_fwd_max - cnull).reshape(1, -1))
    
    mu = -cv
    for p_edge_idx, (src, dst) in enumerate(pattern_edges):
        mu[:,dst] += fwd_max[:,p_edge_idx]
        mu[:,src] += bak_max[:,p_edge_idx]
    
    mu = normprob(mu)

np.savetxt('python_result', np.hstack(mu))
