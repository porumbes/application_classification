#!/bin/bash

"""
    prob2bin.py
"""

import numpy as np
import pandas as pd
from functools import partial

def read_csv(inpath):
  return pd.read_csv(inpath, skiprows=1, sep=' ', header=None)

# --
# IO

dv_inpath = './data/rmat18.Vertex.csv'
de_inpath = './data/rmat18.Edges.csv'

pv_inpath = './data/georgiyPattern.Vertex.csv'
pe_inpath = './data/georgiyPattern.Edges.csv'

data_vertex    = read_csv(dv_inpath)
pattern_vertex = read_csv(pv_inpath)
data_edges     = read_csv(de_inpath)
pattern_edges  = read_csv(pe_inpath)

assert (data_vertex[0] == data_vertex.index).all()
assert (pattern_vertex[0] == pattern_vertex.index).all()

# --

data_vertex      = data_vertex.values[:,1:]
data_edges_table = data_edges[list(range(2, data_edges.shape[1]))].values
data_edges       = data_edges[[0, 1]].values

pattern_vertex      = pattern_vertex.values[:,1:]
pattern_edges_table = pattern_edges[list(range(2, pattern_edges.shape[1]))].values
pattern_edges       = pattern_edges[[0, 1]].values

print(data_vertex.shape)
print(data_edges_table.shape)
print(pattern_vertex.shape)
print(pattern_edges_table.shape)

with open('data/data.bin', 'wb') as f:
  for xx in [
    np.hstack([data_vertex.shape, data_edges_table.shape]).astype(np.uint64),
    data_vertex.ravel().astype(np.float64),
    data_edges_table.ravel().astype(np.float64),
    data_edges[:,0].ravel().astype(np.uint64),
    data_edges[:,1].ravel().astype(np.uint64),
  ]:
    f.write(bytearray(xx))

with open('data/pattern.bin', 'wb') as f:
  for xx in [
    np.hstack([pattern_vertex.shape, pattern_edges_table.shape]).astype(np.uint64),
    pattern_vertex.ravel().astype(np.float64),
    pattern_edges_table.ravel().astype(np.float64),
    pattern_edges[:,0].ravel().astype(np.uint64),
    pattern_edges[:,1].ravel().astype(np.uint64),
  ]:
    f.write(bytearray(xx))