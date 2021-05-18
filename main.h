#ifndef __MAIN_H
#define __MAIN_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <algorithm>
#include <math.h>
#include <cfloat>

typedef uint64_t IntT;
typedef double FloatT;

typedef IntT Int;
typedef FloatT Real;

typedef struct Graph {
  IntT    num_nodes;
  IntT    node_feat_dim;
  FloatT* node_feats;

  IntT    num_edges;
  IntT    edge_feat_dim;
  FloatT* edge_feats;

  IntT* srcs;
  IntT* dsts;

  IntT* srcs_r;
  IntT* dsts_r;
  IntT* map_r;
} Graph;


namespace ac {
  namespace host {

    void ColumnMax2(Int, Int, Real*, Real*);
    void ColumnSoftmax2(Int, Int, Real*);
    void ColumnSoftmax2_prealloc(Int, Int, Real*, Real*);
    void EdgeMaxReduce2(Int, Int, Int, Real*, Real*, Real*, Int*);
    void ComputeMU2(
      Int row_in,
      Int col_in,
      Int row_out,
      Int col_out,
      Real* CV,
      Real* FMax,
      Real* RMax,
      Int* srcs,
      Int* dsts,
      Real* MU
    );
    
    void SortEdges(
      IntT*, IntT*, IntT*, IntT*, IntT*, IntT);

    void ColumnMax(
      IntT, IntT, FloatT*, FloatT*);

    void ColumnSoftmax(
      IntT, IntT, FloatT*);

    void EdgeMaxReduce(
      IntT, IntT, IntT, FloatT*, FloatT*, FloatT*, IntT*, IntT*);

    void ComputeMU(
      Graph*, IntT, FloatT*, FloatT*, FloatT*, FloatT*);
  }
  namespace device {

    __global__ void NodePairwiseNorm(
      IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*, IntT);

    __global__ void EdgePairwiseNorm(
      IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*, FloatT*, IntT);

    __global__ void RepeatColumnsByPatternEdges(
      IntT, IntT, IntT, FloatT*, FloatT*, FloatT*, IntT*, IntT*);

    __global__ void RepeatColumnsByPatternEdgesSubtract(
      IntT, IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*, FloatT*, IntT*, IntT*);

    __global__ void RepeatColumnsByDataEdges(
      IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*, FloatT*, IntT*);

  }
}

#endif