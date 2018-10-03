#include <iostream>
#include "main.h"
#include "kernels.cuh"

void run_iteration(Graph* d_Data_Graph, Graph* d_Pattern_Graph, WorkArrays& d_WA) {
  const uint64_t DV = d_Data_Graph->num_vertices;
  const uint64_t DE = d_Data_Graph->num_edges;
  const uint64_t PV = d_Pattern_Graph->num_vertices;
  const uint64_t PE = d_Pattern_Graph->num_edges;

  d_VF_VR(d_Data_Graph, d_Pattern_Graph, d_WA.MU, d_WA.FMax, d_WA.RMax, d_WA.VF, d_WA.VR);
  d_VFmax_VRmax(d_Data_Graph, d_Pattern_Graph, d_WA.VF, d_WA.VR, d_WA.VFmax, d_WA.VRmax);
  d_FE_RE(d_Data_Graph, d_Pattern_Graph, d_WA.CE, d_WA.VF, d_WA.VR, d_WA.FE, d_WA.RE);
  d_NormProb(DE, PE, d_WA.FE);
  d_NormProb(DE, PE, d_WA.RE);

  d_FMax(d_Data_Graph, d_Pattern_Graph, d_WA.Cnull, d_WA.VRmax, d_WA.FE, d_WA.FMax);
  d_RMax(d_Data_Graph, d_Pattern_Graph, d_WA.Cnull, d_WA.VFmax, d_WA.RE, d_WA.RMax);
  d_UpdateMU(d_Data_Graph, d_Pattern_Graph, d_WA.CV, d_WA.FMax, d_WA.RMax, d_WA.MU);
  d_NormProb(DV, PV, d_WA.MU);
}