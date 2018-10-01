#include "main.h"

int main ( int argc, char * argv[] ) {

// READ FILE NAME FROM COMMAND LINE
  if (argc != 6) {printf("Usage: <number of threads> <data vertices> <data edges> <pattern vertices> <pattern edges> \n"); return 1;}

  int number_of_threads = atoi(argv[1]);
  if ((number_of_threads < 1) && (number_of_threads > 8)) {printf("Number of threads must [1..8]\n"); exit(1);}

  omp_set_dynamic(0);
  omp_set_num_threads(number_of_threads);

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
