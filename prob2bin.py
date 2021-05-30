#!/bin/bash

"""
    prob2bin.py
"""

import sys
import argparse

import numpy as np
import pandas as pd
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_vertex', type=str, help='vertex csv file, e.g., georgiyData.Vertex.csv')
    parser.add_argument('data_edges', type=str, help='edge csv file, e.g., georgiyData.Edge.csv')
    parser.add_argument('data_pattern_vertex', type=str, help='vertex pattern csv file, e.g., georgiyPattern.Vertex.csv')
    parser.add_argument('data_pattern_edges', type=str, help='edge csv file, e.g., georgiyPattern.Edge.csv')
    parser.add_argument('--out_data_prefix', type=str, help='value will result in <out_data_prefix>_data.bin', default='data')
    parser.add_argument('--out_pattern_prefix', type=str, help='value will result in <out_pattern_prefix>_pattern.bin', default='pattern')
    parser.add_argument('--out_path', type=str, default='./data')
    args = parser.parse_args()

    return args

def read_csv(inpath):
    print(f'prob2bin.py reading csv: {inpath}')
    return pd.read_csv(inpath, skiprows=1, sep=' ', header=None)

def main():
    args = parse_args()

    # --
    # IO
    
    dv_inpath = args.data_vertex
    de_inpath = args.data_edges
    
    pv_inpath = args.data_pattern_vertex
    pe_inpath = args.data_pattern_edges

    d_prefix = args.out_data_prefix
    p_prefix = args.out_pattern_prefix
    out_path = args.out_path

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
    
    #with open('data/data.bin', 'wb') as f:
    with open(f'{out_path}/{d_prefix}_data.bin', 'wb') as f:
      for xx in [
        np.hstack([data_vertex.shape, data_edges_table.shape]).astype(np.uint64),
        data_vertex.ravel().astype(np.float64),
        data_edges_table.ravel().astype(np.float64),
        data_edges[:,0].ravel().astype(np.uint64),
        data_edges[:,1].ravel().astype(np.uint64),
      ]:
        f.write(bytearray(xx))
    
    #with open('data/pattern.bin', 'wb') as f:
    with open(f'{out_path}/{p_prefix}_pattern.bin', 'wb') as f:
      for xx in [
        np.hstack([pattern_vertex.shape, pattern_edges_table.shape]).astype(np.uint64),
        pattern_vertex.ravel().astype(np.float64),
        pattern_edges_table.ravel().astype(np.float64),
        pattern_edges[:,0].ravel().astype(np.uint64),
        pattern_edges[:,1].ravel().astype(np.uint64),
      ]:
        f.write(bytearray(xx))

if __name__ == "__main__":
    main()
