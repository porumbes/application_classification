#include <iostream>
#include "main.h"
#include "kernels.cuh"

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

  // Pairwise distances
  device2host(h_WA, d_WA, DV, DE, PV, PE);
    // Pairwise distance computation
    // Init_CV_MU(h_Data_Graph, h_Pattern_Graph, h_WA.CV, h_WA.MU);
  host2device(h_WA, d_WA, DV, DE, PV, PE);

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

  device2host(h_WA, d_WA, DV, DE, PV, PE);
  host2device(h_WA, d_WA, DV, DE, PV, PE);
}
