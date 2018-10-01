#include "main.h"

int main ( int argc, char * argv[] ) {

  Table Data_Vtable    = readVertexTable(argv[2]);
  Table Data_Etable    = readEdgeTable(argv[3]);
  Table Pattern_Vtable = readVertexTable(argv[4]);
  Table Pattern_Etable = readEdgeTable(argv[5]);

  Graph Data_Graph    = constructGraph(& Data_Vtable, & Data_Etable);
  Graph Pattern_Graph = constructGraph(& Pattern_Vtable, & Pattern_Etable);

  WorkArrays WA = initializeWorkArrays(& Data_Graph, & Pattern_Graph);

  uint64_t DV = Data_Graph.num_vertices;
  uint64_t DE = Data_Graph.num_edges;
  uint64_t PV = Pattern_Graph.num_vertices;
  uint64_t PE = Pattern_Graph.num_edges;

  for (uint64_t iter = 0; iter < PV; iter ++) {
      VF_VR(& Data_Graph, & Pattern_Graph, WA.MU, WA.FMax, WA.RMax, WA.VF, WA.VR);

      VFmax_VRmax(& Data_Graph, & Pattern_Graph, WA.VF, WA.VR, WA.VFmax, WA.VRmax);

      FE_RE(& Data_Graph, & Pattern_Graph, WA.CE, WA.VF, WA.VR, WA.FE, WA.RE);
      NormProb(DE, PE, WA.FE);
      NormProb(DE, PE, WA.RE);

      FMax(& Data_Graph, & Pattern_Graph, WA.Cnull, WA.VRmax, WA.FE, WA.FMax);

      RMax(& Data_Graph, & Pattern_Graph, WA.Cnull, WA.VFmax, WA.RE, WA.RMax);

      MU(& Data_Graph, & Pattern_Graph, WA.CV, WA.FMax, WA.RMax, WA.MU);
      NormProb(DV, PV, WA.MU);
  }

  for (uint64_t i = 0; i < DV * PV; i ++) {
    printf("%e\n", WA.MU[i]);
  }
  return 0;
}
