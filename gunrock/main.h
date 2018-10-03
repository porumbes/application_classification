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

typedef struct WorkArrays {
  FloatT* CV;
  FloatT* CE;
  FloatT* Cnull;
  FloatT* MU;
  FloatT* RE;
  FloatT* FE;
  FloatT* VR;
  FloatT* VF;
  FloatT* VRmax;
  FloatT* VFmax;
  FloatT* RMax;
  FloatT* FMax;
} WorkArrays;

void initializeWorkArrays(Graph *, Graph *, WorkArrays&);
void run_iteration(Graph *, Graph *, WorkArrays&);

#endif