#include <iostream>
#include "main.h"
#include "kernels.cuh"

void initializeWorkArrays(Graph * d_Data_Graph, Graph * d_Pattern_Graph, WorkArrays &d_WA) {

  const uint64_t DV = d_Data_Graph->num_vertices;
  const uint64_t DE = d_Data_Graph->num_edges;
  const uint64_t PV = d_Pattern_Graph->num_vertices;
  const uint64_t PE = d_Pattern_Graph->num_edges;

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

  d_Init_CV_MU(d_Data_Graph, d_Pattern_Graph, d_WA.CV, d_WA.MU);
  d_NormProb(DV, PV, d_WA.CV);
  d_NormProb(DV, PV, d_WA.MU);
  d_Init_VR_VF(d_Data_Graph, d_Pattern_Graph, d_WA.MU, d_WA.VR, d_WA.VF);
  d_Init_CE_RE_FE(d_Data_Graph, d_Pattern_Graph, d_WA.CE, d_WA.RE, d_WA.FE);
  d_NormProb(DE, PE, d_WA.CE);
  d_NormProb(DE, PE, d_WA.RE);
  d_NormProb(DE, PE, d_WA.FE);
  cudaMemset(d_WA.Cnull, 0, PE * sizeof(double));
  d_VFmax_VRmax(d_Data_Graph, d_Pattern_Graph, d_WA.VF, d_WA.VR, d_WA.VFmax, d_WA.VRmax);
  d_FMax(d_Data_Graph, d_Pattern_Graph, d_WA.Cnull, d_WA.VRmax, d_WA.FE, d_WA.FMax);
  d_RMax(d_Data_Graph, d_Pattern_Graph, d_WA.Cnull, d_WA.VFmax, d_WA.RE, d_WA.RMax);
}
