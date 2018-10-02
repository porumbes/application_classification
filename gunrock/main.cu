#ifndef __MAIN_CU
#define __MAIN_CU

#include <iostream>
#include "main.h"

int main ( int argc, char * argv[] ) {

  // --
  // IO

  Table Data_Vtable    = readVertexTable(argv[2]);
  Table Data_Etable    = readEdgeTable(argv[3]);
  Table Pattern_Vtable = readVertexTable(argv[4]);
  Table Pattern_Etable = readEdgeTable(argv[5]);

  Graph Data_Graph    = constructGraph(& Data_Vtable, & Data_Etable);
  Graph Pattern_Graph = constructGraph(& Pattern_Vtable, & Pattern_Etable);

  // --
  // Init
  WorkArrays h_WA;
  WorkArrays d_WA;
  initializeWorkArrays(&Data_Graph, &Pattern_Graph, h_WA, d_WA);

  // --
  // Run

  uint64_t DV = Data_Graph.num_vertices;
  uint64_t PV = Pattern_Graph.num_vertices;

  // for (uint64_t iter = 0; iter < PV; iter ++) {
  //     run_iteration(&Data_Graph, &Pattern_Graph, h_WA, d_WA);
  // }

  // --
  // Print results

  for (uint64_t i = 0; i < DV * PV; i ++) {
    printf("%e\n", h_WA.MU[i]);
  }
  return 0;
}

#endif