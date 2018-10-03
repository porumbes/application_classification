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
  IntT* srcs_flipped;
  IntT* dsts_flipped;
} Graph;


namespace ac {
  void ColumnSoftmax(IntT, IntT, FloatT*);

  void Init_CV_MU(Graph*, Graph*, FloatT*, FloatT*);
  void Init_CE_RE_FE(Graph*, Graph*, FloatT*, FloatT*, FloatT*);

  void Init_VR_VF(Graph*, IntT, FloatT*, FloatT*, FloatT*);
  void VFmax_VRmax(IntT, IntT, FloatT*, FloatT*, FloatT*, FloatT*);
  void VF_VR(Graph*, IntT, FloatT*, FloatT*, FloatT*, FloatT*, FloatT*);
  void UpdateMU(Graph*, IntT, FloatT*, FloatT*, FloatT*, FloatT*);
  void FE_RE(Graph *, IntT, FloatT*, FloatT*, FloatT*, FloatT*, FloatT*);
  void FMax(Graph *, IntT, FloatT*, FloatT*, FloatT*, FloatT*);
  void RMax(Graph *, IntT, FloatT*, FloatT*, FloatT*, FloatT*);
}

#endif