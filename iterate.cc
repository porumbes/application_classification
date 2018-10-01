#include "main.h"

void VF_VR(Graph * Data_Graph, Graph * Pattern_Graph,
           double * MU, double * FMax, double * RMax, double * VF, double * VR) {
  uint64_t num_DV  = Data_Graph->num_vertices;
  uint64_t num_PV  = Pattern_Graph->num_vertices;
  uint64_t num_PE  = Pattern_Graph->num_edges;
  uint64_t num_AT  = Pattern_Graph->Etable.num_cols;               // number of edge attributes
  uint64_t * PAttr = (uint64_t *) Pattern_Graph->Etable.table;     // reading only src and dst, so use uint64_t *

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
      uint64_t i = k / num_PE;
      uint64_t j = k % num_PE;
      uint64_t src = PAttr[j * num_AT];         // src id
      uint64_t dst = PAttr[j * num_AT + 1];     // dst id
      VF[k] = MU[i * num_PV + dst] - FMax[k];
      VR[k] = MU[i * num_PV + src] - RMax[k];
} }


void VFmax_VRmax(Graph * Data_Graph, Graph * Pattern_Graph,
                 double * VF, double * VR, double * VFmax, double * VRmax) {
  uint64_t num_DV = Data_Graph->num_vertices;
  uint64_t num_PE = Pattern_Graph->num_edges;

  for (uint64_t j = 0; j < num_PE; j ++) {
      VFmax[j] = - DBL_MAX; VRmax[j] = - DBL_MAX;
  }

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
      uint64_t j = k % num_PE;
      atomic_fmax(VFmax + j, VF[k]);
      atomic_fmax(VRmax + j, VR[k]);
} }


void FE_RE(Graph * Data_Graph, Graph * Pattern_Graph,
           double * CE, double * VF, double * VR, double * FE, double * RE) {
  uint64_t num_DE  = Data_Graph->num_edges;
  uint64_t num_PE  = Pattern_Graph->num_edges;
  uint64_t num_AT  = Data_Graph->Etable.num_cols;               // number of edge attributes
  uint64_t * DAttr = (uint64_t *) Data_Graph->Etable.table;     // reading only src, so use uint64_t *

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t ij  = k / num_PE;
      uint64_t km  = k % num_PE;
      uint64_t src = DAttr[ij * num_AT];
      FE[k] = - CE[k] + VR[src * num_PE + km];
      RE[k] = - CE[k] + VF[src * num_PE + km];
} }


void FMax(Graph * Data_Graph, Graph * Pattern_Graph, double * Cnull, double * VRmax, double * FE, double * FMax) {
  uint64_t num_DE  = Data_Graph->num_edges;
  uint64_t num_PE  = Pattern_Graph->num_edges;
  uint64_t num_DV  = Data_Graph->num_vertices;
  uint64_t num_AT  = Data_Graph->Etable.num_cols;               // number of edge attributes
  uint64_t * DAttr = (uint64_t *) Data_Graph->Etable.table;     // reading only dst, so use uint64_t *

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
      uint64_t j = k % num_PE;
      FMax[k] = -Cnull[j] + VRmax[j];
  }

#pragma omp parallel for
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

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
      uint64_t j = k % num_PE;
      RMax[k] = -Cnull[j] + VFmax[j];
  }

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DE * num_PE; k ++) {
      uint64_t ij  = k / num_PE;
      uint64_t km  = k % num_PE;
      uint64_t src = DAttr[ij * num_AT];
      atomic_fmax(RMax + src * num_PE + km, RE[k]);
} }


void MU(Graph * Data_Graph, Graph * Pattern_Graph, double * CV, double * FMax, double * RMax, double * MU) {
  uint64_t num_DV  = Data_Graph->num_vertices;
  uint64_t num_PV  = Pattern_Graph->num_vertices;
  uint64_t num_PE  = Pattern_Graph->num_edges;
  uint64_t num_AT  = Pattern_Graph->Etable.num_cols;               // number of edge attributes
  uint64_t * PAttr = (uint64_t *) Pattern_Graph->Etable.table;     // reading only src and dst, so use uint64_t *

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PV; k ++) MU[k] = - CV[k];

#pragma omp parallel for
  for (uint64_t k = 0; k < num_DV * num_PE; k ++) {
          uint64_t i   = k / num_PE;
          uint64_t j   = k % num_PE;
          uint64_t src = PAttr[j * num_AT];         // src id
          uint64_t dst = PAttr[j * num_AT + 1];     // dst id
          #pragma omp atomic
          MU[i * num_PV + dst] += FMax[k];
          #pragma omp atomic
          MU[i * num_PV + src] += RMax[k];
} }
