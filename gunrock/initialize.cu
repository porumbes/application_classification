#include <iostream>
#include "main.h"

#include "kernels.cu"

double norm(uint64_t num_attr, double * vec1, double * vec2) {
  double sum = 0.0;

  if (vec2 == NULL)
     for (uint64_t i = 0; i < num_attr; i ++) sum += (vec1[i] * vec1[i]);
  else
     for (uint64_t i = 0; i < num_attr; i ++) sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);

  return sqrt(sum);
}


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


void NormProb(uint64_t num_DE, uint64_t num_PE, double * Prob) {
  double * Probmax    = (double *) malloc(num_PE * sizeof(double));
  double * Probglobal = (double *) malloc(num_PE * sizeof(double));
  for (uint64_t j = 0; j < num_PE; j ++) {
      Probmax[j] = - DBL_MAX;
      Probglobal[j] = 0.0;
  }

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t j = k % num_PE;
      atomic_fmax(Probmax + j, Prob[k]);
  }

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t j = k % num_PE;
      Prob[k] = exp(Prob[k] - Probmax[j]);

      // #pragma omp atomic
      Probglobal[j] += Prob[k];
  }

  for (uint64_t j = 0; j < num_PE; j ++) Probglobal[j] = log(Probglobal[j]);

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t j = k % num_PE;
      Prob[k] = log(Prob[k]) - Probglobal[j];
  }

  free(Probmax);
  free(Probglobal);
}


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
} }


// Initialize VR, VF
void Init_VR_VF(Graph * Data_Graph, Graph * Pattern_Graph, double * MU, double * VR, double * VF) {
  uint64_t num_DV  = Data_Graph->num_vertices;
  uint64_t num_PV  = Pattern_Graph->num_vertices;
  uint64_t num_PE  = Pattern_Graph->num_edges;
  uint64_t num_AT  = Pattern_Graph->Etable.num_cols;               // number of pattern edge attributes
  uint64_t * PAttr = (uint64_t *) Pattern_Graph->Etable.table;     // reading only vertex ids, so use uint64_t *

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
      uint64_t i = k / num_PE;
      uint64_t j = k % num_PE;
      uint64_t src = PAttr[j * num_AT];         // src id
      uint64_t dst = PAttr[j * num_AT + 1];     // dst id
      VR[k] = MU[i * num_PV + src];
      VF[k] = MU[i * num_PV + dst];
} }


// Initialize CE, RE, FE
void Init_CE_RE_FE(Graph * Data_Graph, Graph * Pattern_Graph, double * CE, double * RE, double * FE) {
  uint64_t num_DE = Data_Graph->num_edges;
  uint64_t num_PE = Pattern_Graph->num_edges;
  uint64_t num_AT = Data_Graph->Etable.num_cols;                // number of edge attributes
  double * DAttr  = (double *) Data_Graph->Etable.table;        // reading only attributes, so use double *
  double * PAttr  = (double *) Pattern_Graph->Etable.table;     // reading only attributes, so use double *

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t i = k / num_PE;
      uint64_t j = k % num_PE;
      CE[k] = norm(num_AT - 2, PAttr + j * num_AT + 2, DAttr + i * num_AT + 2);
      RE[k] = - CE[k];
      FE[k] = - CE[k];
  }
}


void Init_Cnull(Graph * Data_Graph, Graph * Pattern_Graph, double * CE, double * Cnull) {
  uint64_t num_DE = Data_Graph->num_edges;
  uint64_t num_PE = Pattern_Graph->num_edges;
  uint64_t num_AT = Data_Graph->Etable.num_cols;                // number of edge attributes
  double * PAttr  = (double *) Pattern_Graph->Etable.table;     // reading only attributes, so use double *

  for (uint64_t j = 0; j < num_PE; j ++) {
      Cnull[j] = norm(num_AT - 2, PAttr + j * num_AT + 2, NULL);
  }

// #pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t j = k % num_PE;
      atomic_fmax(Cnull + j, CE[k]);
} }


WorkArrays initializeWorkArrays(Graph * Data_Graph, Graph * Pattern_Graph) {
  WorkArrays h_WA;
  WorkArrays d_WA;

  uint64_t DV  = Data_Graph->num_vertices;
  uint64_t DE  = Data_Graph->num_edges;
  uint64_t PV  = Pattern_Graph->num_vertices;
  uint64_t PE  = Pattern_Graph->num_edges;

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

  double* CV2;
  cudaMalloc((void **)&CV2, DV * PV * sizeof(double));

  // Pairwise distances
  Init_CV_MU(Data_Graph, Pattern_Graph, h_WA.CV, h_WA.MU);
  cudaMemcpy(d_WA.CV, h_WA.CV, DV * PV * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(CV2, h_WA.CV, DV * PV * sizeof(double), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_WA.MU, h_WA.MU, DV * PV * sizeof(double), cudaMemcpyHostToDevice);

  // Normalize CV
  // NormProb(DV, PV, h_WA.CV);
  // NormProb(DV, PV, h_WA.MU);
  // Init_VR_VF(Data_Graph, Pattern_Graph, h_WA.MU, h_WA.VR, h_WA.VF);
  // Init_CE_RE_FE(Data_Graph, Pattern_Graph, h_WA.CE, h_WA.RE, h_WA.FE);
  // NormProb(DE, PE, h_WA.CE);
  // NormProb(DE, PE, h_WA.RE);
  // NormProb(DE, PE, h_WA.FE);
  // Init_Cnull(Data_Graph, Pattern_Graph, h_WA.CE, h_WA.Cnull);
  // NormProb(1, PE, h_WA.Cnull);

  d_NormProb(DV, PV, d_WA.CV);
  d_NormProb(DV, PV, d_WA.MU);
  d_Init_VR_VF(Data_Graph, Pattern_Graph, d_WA.MU, d_WA.VR, d_WA.VF);
  d_Init_CE_RE_FE(Data_Graph, Pattern_Graph, h_WA.CE, h_WA.RE, h_WA.FE);
  d_NormProb(DE, PE, d_WA.CE);
  d_NormProb(DE, PE, d_WA.RE);
  d_NormProb(DE, PE, d_WA.FE);
  cudaMemset(d_WA.Cnull, 0, PE * sizeof(double));

  cudaMemcpy(h_WA.CV, d_WA.CV, DV * PV * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.MU, d_WA.MU, DV * PV * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.VR, d_WA.VR, DV * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.VF, d_WA.VF, DV * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.CE, d_WA.CE, DE * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.RE, d_WA.RE, DE * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.FE, d_WA.FE, DE * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.Cnull, d_WA.Cnull, PE * sizeof(double), cudaMemcpyDeviceToHost);

  // vv WIP vv

  VFmax_VRmax(Data_Graph, Pattern_Graph, h_WA.VF, h_WA.VR, h_WA.VFmax, h_WA.VRmax);
  FMax(Data_Graph, Pattern_Graph, h_WA.Cnull, h_WA.VRmax, h_WA.FE, h_WA.FMax);
  RMax(Data_Graph, Pattern_Graph, h_WA.Cnull, h_WA.VFmax, h_WA.RE, h_WA.RMax);

  cudaFree(d_WA.CV);
  cudaFree(d_WA.CE);
  cudaFree(d_WA.Cnull);
  cudaFree(d_WA.MU);
  cudaFree(d_WA.RE);
  cudaFree(d_WA.FE);
  cudaFree(d_WA.VR);
  cudaFree(d_WA.VF);
  cudaFree(d_WA.VRmax);
  cudaFree(d_WA.VFmax);
  cudaFree(d_WA.RMax);
  cudaFree(d_WA.FMax);

  return h_WA;
}
