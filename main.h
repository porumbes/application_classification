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

    void SortEdges(
      IntT*, IntT*, IntT*, IntT*, IntT*, IntT);

    void RowMax(
      IntT, IntT, FloatT*, FloatT*, IntT*);

    void RowSoftmax(
      IntT, IntT, FloatT*, IntT*);

    void EdgeMaxReduce(
      IntT, IntT, IntT, FloatT*, FloatT*, FloatT*, IntT*, IntT*, FloatT*, FloatT*);

    void ComputeMU(
      Graph*, IntT, FloatT*, FloatT*, FloatT*, FloatT*);

  }
  namespace device {

    __global__ void NodePairwiseNorm(
      IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*, IntT);

    __global__ void EdgePairwiseNorm(
      IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*, FloatT*, IntT);

    __global__ void RepeatRowsByPatternEdges(
      IntT, IntT, IntT, FloatT*, FloatT*, FloatT*, IntT*, IntT*);

    __global__ void RepeatRowsByPatternEdgesSubtract(
      IntT, IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*, FloatT*, IntT*, IntT*);

    __global__ void RepeatRowsByDataEdges(
      IntT, IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*, FloatT*, IntT*);

  }
}

#endif