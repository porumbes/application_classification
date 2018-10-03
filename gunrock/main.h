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

typedef struct Table {
  uint64_t num_rows;
  uint64_t num_cols;
  uint64_t *srcs;
  uint64_t *dsts;
  uint64_t *srcs_r;
  uint64_t *dsts_r;
  uint64_t *table;
} Table;

typedef struct Graph {
  uint64_t num_edges;
  uint64_t num_vertices;
  Table Vtable;
  Table Etable;
} Graph;

typedef struct WorkArrays {
  double * CV;
  double * CE;
  double * Cnull;
  double * MU;
  double * RE;
  double * FE;
  double * VR;
  double * VF;
  double * VRmax;
  double * VFmax;
  double * RMax;
  double * FMax;
} WorkArrays;

double my_timer();
double atomic_fmax(double *, double);
void NormProb(uint64_t, uint64_t, double *);

Table readEdgeTable(char *);
Table readVertexTable(char *);
Graph constructGraph(Table *, Table *);
void initializeWorkArrays(Graph *, Graph *, WorkArrays&);

void VF_VR(Graph *, Graph *, double *, double *, double *, double *, double *);
void VFmax_VRmax(Graph *, Graph *, double *, double *, double *, double *);
void FE_RE(Graph *, Graph *, double *, double *, double *, double *, double *);
void FMax(Graph *, Graph *, double *, double *, double *, double *);
void RMax(Graph *, Graph *, double *, double *, double *, double *);
void MU(Graph *, Graph *, double *, double *, double *, double *);

void run_iteration(Graph *, Graph *, WorkArrays&);

#endif