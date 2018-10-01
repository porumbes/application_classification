#include "main.h"

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


void NormProb(uint64_t num_DE,uint64_t num_PE, double * Prob) {
  double * Probmax    = (double *) malloc(num_PE * sizeof(double));
  double * Probglobal = (double *) malloc(num_PE * sizeof(double));
  for (uint64_t j = 0; j < num_PE; j ++) {
      Probmax[j] = - DBL_MAX;
      Probglobal[j] = 0.0;
  }

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t j = k % num_PE;
      atomic_fmax(Probmax + j, Prob[k]);
  }

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t j = k % num_PE;
      Prob[k] = exp(Prob[k] - Probmax[j]);

      #pragma omp atomic
      Probglobal[j] += Prob[k];
  }

  for (uint64_t j = 0; j < num_PE; j ++) Probglobal[j] = log(Probglobal[j]);

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t j = k % num_PE;
      Prob[k] = log(Prob[k]) - Probglobal[j];
  }

  free(Probmax);
  free(Probglobal);
}


// Initialize CV, MU
void Init_CV_MU(Graph * Data_Graph, Graph * Pattern_Graph, double * CV, double * MU) {
  uint64_t num_DV = Data_Graph->num_vertices;
  uint64_t num_PV = Pattern_Graph->num_vertices;
  uint64_t num_AT = Data_Graph->Vtable.num_cols;               // number of vertex attributes
  double * DAttr  = (double *) Data_Graph->Vtable.table;       // reading only attributes, so use double *
  double * PAttr  = (double *) Pattern_Graph->Vtable.table;    // reading only attributes, so use double *

#pragma omp parallel for
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

#pragma omp parallel for
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

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t i = k / num_PE;
      uint64_t j = k % num_PE;
      CE[k] = norm(num_AT - 2, PAttr + j * num_AT + 2, DAttr + i * num_AT + 2);
      RE[k] = - CE[k];
      FE[k] = - CE[k];
} }


void Init_Cnull(Graph * Data_Graph, Graph * Pattern_Graph, double * CE, double * Cnull) {
  uint64_t num_DE = Data_Graph->num_edges;
  uint64_t num_PE = Pattern_Graph->num_edges;
  uint64_t num_AT = Data_Graph->Etable.num_cols;                // number of edge attributes
  double * PAttr  = (double *) Pattern_Graph->Etable.table;     // reading only attributes, so use double *

  for (uint64_t j = 0; j < num_PE; j ++) {
      Cnull[j] = norm(num_AT - 2, PAttr + j * num_AT + 2, NULL);
  }

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t j = k % num_PE;
      atomic_fmax(Cnull + j, CE[k]);
} }


WorkArrays initializeWorkArrays(Graph * Data_Graph, Graph * Pattern_Graph) {
  WorkArrays WA;
  uint64_t DV  = Data_Graph->num_vertices;
  uint64_t DE  = Data_Graph->num_edges;
  uint64_t PV  = Pattern_Graph->num_vertices;
  uint64_t PE  = Pattern_Graph->num_edges;

  WA.CV       = (double *) malloc(DV * PV * sizeof(double));
  WA.CE       = (double *) malloc(DE * PE * sizeof(double));
  WA.Cnull    = (double *) malloc(PE *      sizeof(double));
  WA.MU       = (double *) malloc(DV * PV * sizeof(double));
  WA.RE       = (double *) malloc(DE * PE * sizeof(double));
  WA.FE       = (double *) malloc(DE * PE * sizeof(double));
  WA.VR       = (double *) malloc(DV * PE * sizeof(double));
  WA.VF       = (double *) malloc(DV * PE * sizeof(double));
  WA.VRmax    = (double *) malloc(PE *      sizeof(double));
  WA.VFmax    = (double *) malloc(PE *      sizeof(double));
  WA.RMax     = (double *) malloc(DV * PE * sizeof(double));
  WA.FMax     = (double *) malloc(DV * PE * sizeof(double));

  Init_CV_MU(Data_Graph, Pattern_Graph, WA.CV, WA.MU);
  NormProb(DV, PV, WA.CV);
  NormProb(DV, PV, WA.MU);

  Init_VR_VF(Data_Graph, Pattern_Graph, WA.MU, WA.VR, WA.VF);
  Init_CE_RE_FE(Data_Graph, Pattern_Graph, WA.CE, WA.RE, WA.FE);
  NormProb(DE, PE, WA.CE);
  NormProb(DE, PE, WA.RE);
  NormProb(DE, PE, WA.FE);

  Init_Cnull(Data_Graph, Pattern_Graph, WA.CE, WA.Cnull);
  NormProb(1, PE, WA.Cnull);

  VFmax_VRmax(Data_Graph, Pattern_Graph, WA.VF, WA.VR, WA.VFmax, WA.VRmax);
  FMax(Data_Graph, Pattern_Graph, WA.Cnull, WA.VRmax, WA.FE, WA.FMax);
  RMax(Data_Graph, Pattern_Graph, WA.Cnull, WA.VFmax, WA.RE, WA.RMax);
  return WA;
}
