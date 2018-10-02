#include <iostream>
#include "main.h"
#include "kernels.cuh"

double norm(uint64_t num_attr, double * vec1, double * vec2) {
  double sum = 0.0;

  if (vec2 == NULL) {
     for (uint64_t i = 0; i < num_attr; i ++) {
      sum += (vec1[i] * vec1[i]);
    }
  } else {
     for (uint64_t i = 0; i < num_attr; i ++) {
      sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
  }

  return sqrt(sum);
}


// double atomic_fmax(double * elem, double value) {
//   double old_value;
//   int64_t * ptr = (int64_t *) elem;

//   while (true) {
//      old_value = * (double *) ptr;
//      double new_value = fmax(old_value, value);
//      int64_t * old_value_ptr = (int64_t *) & old_value;
//      int64_t * new_value_ptr = (int64_t *) & new_value;
//      if (__sync_bool_compare_and_swap(ptr, * old_value_ptr, * new_value_ptr)) break;
//   }

//   return old_value;
// }


// void NormProb(int num_DE, int num_PE, double * Prob) {
//   double * Probmax    = (double *) malloc(num_PE * sizeof(double));
//   double * Probglobal = (double *) malloc(num_PE * sizeof(double));
//   for (int j = 0; j < num_PE; j ++) {
//       Probmax[j] = - DBL_MAX;
//       Probglobal[j] = 0.0;
//   }

// // #pragma omp parallel for
//   for (int k = 0; k < num_DE * num_PE; k ++) {
//       int j = k % num_PE;
//       atomic_fmax(Probmax + j, Prob[k]);
//   }

// // #pragma omp parallel for
//   for (int k = 0; k < num_DE * num_PE; k ++) {
//       int j = k % num_PE;
//       Prob[k] = exp(Prob[k] - Probmax[j]);

//       // #pragma omp atomic
//       Probglobal[j] += Prob[k];
//   }

//   for (int j = 0; j < num_PE; j ++) Probglobal[j] = log(Probglobal[j]);

// // #pragma omp parallel for
//   for (int k = 0; k < num_DE * num_PE; k ++) {
//       int j = k % num_PE;
//       Prob[k] = log(Prob[k]) - Probglobal[j];
//   }

//   free(Probmax);
//   free(Probglobal);
// }


// Pairwise euclidean distance between features
void Init_CV_MU(Graph * Data_Graph, Graph * Pattern_Graph, double * CV, double * MU) {
  uint64_t num_DV = Data_Graph->num_vertices;
  uint64_t num_PV = Pattern_Graph->num_vertices;
  uint64_t num_AT = Data_Graph->Vtable.num_cols;               // number of vertex attributes
  double * DAttr  = (double *) Data_Graph->Vtable.table;       // reading only attributes, so use double *
  double * PAttr  = (double *) Pattern_Graph->Vtable.table;    // reading only attributes, so use double *

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PV; k ++) {
      uint64_t i = k / num_PV;
      uint64_t j = k % num_PV;
      CV[k] = norm(num_AT - 1, PAttr + j * num_AT + 1, DAttr + i * num_AT + 1);
      MU[k] = - CV[k];
  }
}



// void Init_Cnull(Graph * Data_Graph, Graph * Pattern_Graph, double * CE, double * Cnull) {
//   int num_DE = Data_Graph->num_edges;
//   int num_PE = Pattern_Graph->num_edges;
//   int num_AT = Data_Graph->Etable.num_cols;                // number of edge attributes
//   double * PAttr  = (double *) Pattern_Graph->Etable.table;     // reading only attributes, so use double *

//   for (int j = 0; j < num_PE; j ++) {
//       Cnull[j] = norm(num_AT - 2, PAttr + j * num_AT + 2, NULL);
//   }

// // #pragma omp parallel for
//   for (int k = 0; k < num_DE * num_PE; k ++) {
//       int j = k % num_PE;
//       atomic_fmax(Cnull + j, CE[k]);
// } }


void initializeWorkArrays(
  Graph * h_Data_Graph, Graph * h_Pattern_Graph,
  Graph * d_Data_Graph, Graph * d_Pattern_Graph,
  WorkArrays &h_WA, WorkArrays &d_WA) {

  const uint64_t DV = h_Data_Graph->num_vertices;
  const uint64_t DE = h_Data_Graph->num_edges;
  const uint64_t PV = h_Pattern_Graph->num_vertices;
  const uint64_t PE = h_Pattern_Graph->num_edges;

  // CPU allocation
  h_WA.CV       = (double *) malloc(DV * PV * sizeof(double));
  h_WA.CE       = (double *) malloc(DE * PE * sizeof(double));
  h_WA.Cnull    = (double *) malloc(PE *      sizeof(double));
  h_WA.MU       = (double *) malloc(DV * PV * sizeof(double));
  h_WA.RE       = (double *) malloc(DE * PE * sizeof(double));
  h_WA.FE       = (double *) malloc(DE * PE * sizeof(double));
  h_WA.VR       = (double *) malloc(DV * PE * sizeof(double));
  h_WA.VF       = (double *) malloc(DV * PE * sizeof(double));
  h_WA.VRmax    = (double *) malloc(PE *      sizeof(double));
  h_WA.VFmax    = (double *) malloc(PE *      sizeof(double));
  h_WA.RMax     = (double *) malloc(DV * PE * sizeof(double));
  h_WA.FMax     = (double *) malloc(DV * PE * sizeof(double));

  // GPU allocation
  cudaMalloc((void **)&d_WA.CV,    DV * PV * sizeof(double));
  cudaMalloc((void **)&d_WA.CE,    DE * PE * sizeof(double));
  cudaMalloc((void **)&d_WA.Cnull, PE *      sizeof(double));
  cudaMalloc((void **)&d_WA.MU,    DV * PV * sizeof(double));
  cudaMalloc((void **)&d_WA.RE,    DE * PE * sizeof(double));
  cudaMalloc((void **)&d_WA.FE,    DE * PE * sizeof(double));
  cudaMalloc((void **)&d_WA.VR,    DV * PE * sizeof(double));
  cudaMalloc((void **)&d_WA.VF,    DV * PE * sizeof(double));
  cudaMalloc((void **)&d_WA.VRmax, PE *      sizeof(double));
  cudaMalloc((void **)&d_WA.VFmax, PE *      sizeof(double));
  cudaMalloc((void **)&d_WA.RMax,  DV * PE * sizeof(double));
  cudaMalloc((void **)&d_WA.FMax,  DV * PE * sizeof(double));

// look_here
  // error: need to copy `data_graph` and `pattern_graph` to GPU
  // also, need to be careful about `uint64_t` stuff.

  // Pairwise distances
  device2host(h_WA, d_WA, DV, DE, PV, PE);
    // Pairwise distance computation
    Init_CV_MU(h_Data_Graph, h_Pattern_Graph, h_WA.CV, h_WA.MU);
  host2device(h_WA, d_WA, DV, DE, PV, PE);

  d_NormProb(DV, PV, d_WA.CV);
  d_NormProb(DV, PV, d_WA.MU);
  d_Init_VR_VF(d_Data_Graph, d_Pattern_Graph, d_WA.MU, d_WA.VR, d_WA.VF);
  d_Init_CE_RE_FE(d_Data_Graph, d_Pattern_Graph, d_WA.CE, d_WA.RE, d_WA.FE);
  d_NormProb(DE, PE, d_WA.CE);
  d_NormProb(DE, PE, d_WA.RE);
  d_NormProb(DE, PE, d_WA.FE);
  cudaMemset(d_WA.Cnull, 0, PE * sizeof(double));
  d_VFmax_VRmax(d_Data_Graph, d_Pattern_Graph, d_WA.VF, d_WA.VR, d_WA.VFmax, d_WA.VRmax);

  device2host(h_WA, d_WA, DV, DE, PV, PE);
    // Max reduction over data edges
    FMax(h_Data_Graph, h_Pattern_Graph, h_WA.Cnull, h_WA.VRmax, h_WA.FE, h_WA.FMax);
    RMax(h_Data_Graph, h_Pattern_Graph, h_WA.Cnull, h_WA.VFmax, h_WA.RE, h_WA.RMax);
  host2device(h_WA, d_WA, DV, DE, PV, PE);

  device2host(h_WA, d_WA, DV, DE, PV, PE);
  host2device(h_WA, d_WA, DV, DE, PV, PE);
}
