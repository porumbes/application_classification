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
} Graph;

namespace ac {
  namespace host {

    void RowMax2(IntT, IntT, FloatT*, FloatT*);

    void RowSoftmax2(Int, Int, Real*);
    void RowSoftmax2_prealloc(Int, Int, Real*, Real*);
    
    void EdgeMaxReduce2_t(Int, Int, Int, Real*, Real*, Real*, Int*);
    
    void ComputeMU2_t(
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
    
  }

}

#endif