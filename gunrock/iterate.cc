#include <iostream>
#include "main.h"
#include "kernels.cuh"

double atomic_fmax(double * elem, double value) {
  double old_value;
  int64_t * ptr = (int64_t *) elem;

  while (true) {
     old_value = * (double *) ptr;
     double new_value = fmax(old_value, value);
     int64_t * old_value_ptr = (int64_t *) & old_value;
     int64_t * new_value_ptr = (int64_t *) & new_value;
     if (__sync_bool_compare_and_swap(ptr, * old_value_ptr, * new_value_ptr)) break;
  }

  return old_value;
}

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

// void VF_VR(Graph * Data_Graph, Graph * Pattern_Graph,
//            double * MU, double * FMax, double * RMax, double * VF, double * VR) {
//   uint64_t num_DV  = Data_Graph->num_vertices;
//   uint64_t num_PV  = Pattern_Graph->num_vertices;
//   uint64_t num_PE  = Pattern_Graph->num_edges;
//   uint64_t num_AT  = Pattern_Graph->Etable.num_cols;               // number of edge attributes
//   uint64_t * PAttr = (uint64_t *) Pattern_Graph->Etable.table;     // reading only src and dst, so use uint64_t *

// // #pragma omp parallel for
//   for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
//     uint64_t i = k / num_PE;
//     uint64_t j = k % num_PE;
//     uint64_t src = PAttr[j * num_AT];
//     uint64_t dst = PAttr[j * num_AT + 1];
//     VF[k] = MU[i * num_PV + dst] - FMax[k];
//     VR[k] = MU[i * num_PV + src] - RMax[k];
// } }


// void VFmax_VRmax(Graph * Data_Graph, Graph * Pattern_Graph,
//                  double * VF, double * VR, double * VFmax, double * VRmax) {
//   uint64_t num_DV = Data_Graph->num_vertices;
//   uint64_t num_PE = Pattern_Graph->num_edges;

//   for (uint64_t j = 0; j < num_PE; j ++) {
//       VFmax[j] = - DBL_MAX; VRmax[j] = - DBL_MAX;
//   }

// // #pragma omp parallel for
//   for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
//       uint64_t j = k % num_PE;
//       atomic_fmax(VFmax + j, VF[k]);
//       atomic_fmax(VRmax + j, VR[k]);
// } }


// void FE_RE(Graph * Data_Graph, Graph * Pattern_Graph,
//            double * CE, double * VF, double * VR, double * FE, double * RE) {
//   uint64_t num_DE  = Data_Graph->num_edges;
//   uint64_t num_PE  = Pattern_Graph->num_edges;
//   uint64_t num_AT  = Data_Graph->Etable.num_cols;               // number of edge attributes
//   uint64_t * DAttr = (uint64_t *) Data_Graph->Etable.table;     // reading only src, so use uint64_t *

// // #pragma omp parallel for
//   for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
//       uint64_t ij  = k / num_PE;
//       uint64_t km  = k % num_PE;
//       uint64_t src = DAttr[ij * num_AT];
//       FE[k] = - CE[k] + VR[src * num_PE + km];
//       RE[k] = - CE[k] + VF[src * num_PE + km];
// } }


void FMax(Graph * Data_Graph, Graph * Pattern_Graph, double * Cnull, double * VRmax, double * FE, double * FMax) {
  uint64_t num_DE  = Data_Graph->num_edges;
  uint64_t num_PE  = Pattern_Graph->num_edges;
  uint64_t num_DV  = Data_Graph->num_vertices;
  uint64_t num_AT  = Data_Graph->Etable.num_cols;               // number of edge attributes
  uint64_t * DAttr = (uint64_t *) Data_Graph->Etable.table;     // reading only dst, so use uint64_t *

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
      uint64_t j = k % num_PE;
      FMax[k] = -Cnull[j] + VRmax[j];
  }

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t ij  = k / num_PE;
      uint64_t km  = k % num_PE;
      uint64_t dst = DAttr[ij * num_AT + 1];
      atomic_fmax(FMax + dst * num_PE + km, FE[k]);
} }


void RMax(Graph * Data_Graph, Graph * Pattern_Graph, double * Cnull, double * VFmax, double * RE, double * RMax) {
  uint64_t num_DE  = Data_Graph->num_edges;
  uint64_t num_PE  = Pattern_Graph->num_edges;
  uint64_t num_DV  = Data_Graph->num_vertices;
  uint64_t num_AT  = Data_Graph->Etable.num_cols;               // number of edge attributes
  uint64_t * DAttr = (uint64_t *) Data_Graph->Etable.table;     // reading only src, so use uint64_t *

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
      uint64_t j = k % num_PE;
      RMax[k] = -Cnull[j] + VFmax[j];
  }

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t ij  = k / num_PE;
      uint64_t km  = k % num_PE;
      uint64_t src = DAttr[ij * num_AT];
      atomic_fmax(RMax + src * num_PE + km, RE[k]);
} }


void run_iteration(
  Graph* h_Data_Graph, Graph* h_Pattern_Graph,
  Graph* d_Data_Graph, Graph* d_Pattern_Graph,
  WorkArrays& h_WA, WorkArrays& d_WA) {

  int DV = h_Data_Graph->num_vertices;
  int DE = h_Data_Graph->num_edges;
  int PV = h_Pattern_Graph->num_vertices;
  int PE = h_Pattern_Graph->num_edges;

  d_VF_VR(d_Data_Graph, d_Pattern_Graph, d_WA.MU, d_WA.FMax, d_WA.RMax, d_WA.VF, d_WA.VR);
  d_VFmax_VRmax(d_Data_Graph, d_Pattern_Graph, d_WA.VF, d_WA.VR, d_WA.VFmax, d_WA.VRmax);
  d_FE_RE(d_Data_Graph, d_Pattern_Graph, d_WA.CE, d_WA.VF, d_WA.VR, d_WA.FE, d_WA.RE);
  d_NormProb(DE, PE, d_WA.FE);
  d_NormProb(DE, PE, d_WA.RE);

  d_FMax(d_Data_Graph, d_Pattern_Graph, d_WA.Cnull, d_WA.VRmax, d_WA.FE, d_WA.FMax);
  d_RMax(d_Data_Graph, d_Pattern_Graph, d_WA.Cnull, d_WA.VFmax, d_WA.RE, d_WA.RMax);
  d_UpdateMU(d_Data_Graph, d_Pattern_Graph, d_WA.CV, d_WA.FMax, d_WA.RMax, d_WA.MU);
  d_NormProb(DV, PV, d_WA.MU);

  device2host(h_WA, d_WA, DV, DE, PV, PE);
  host2device(h_WA, d_WA, DV, DE, PV, PE);
}