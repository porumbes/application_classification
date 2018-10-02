#ifndef __MAIN_CU
#define __MAIN_CU

#include <iostream>
#include "main.h"

Graph constructGraph(Table * Vtable, Table * Etable) {
  Graph graph;
  graph.Vtable       = * Vtable;
  graph.Etable       = * Etable;
  graph.num_edges    = Etable->num_rows;
  graph.num_vertices = Vtable->num_rows;
  return graph;
}



void table2device(Table* d_table, Table* h_table) {
  d_table->num_rows = h_table->num_rows;
  d_table->num_cols = h_table->num_cols;

  cudaMalloc((void**)&d_table->table, h_table->num_rows * h_table->num_cols * sizeof(uint64_t));
  cudaMemcpy(d_table->table, h_table->table,
    h_table->num_rows * h_table->num_cols * sizeof(uint64_t), cudaMemcpyHostToDevice);

  uint64_t srcs[d_table->num_rows];
  uint64_t dsts[d_table->num_rows];
  for(uint64_t i = 0; i < d_table->num_rows; i++) {
    srcs[i] = h_table->table[i * d_table->num_cols];
    dsts[i] = h_table->table[i * d_table->num_cols + 1];
  }

  cudaMalloc((void**)&d_table->srcs, h_table->num_rows * sizeof(uint64_t));
  cudaMalloc((void**)&d_table->dsts, h_table->num_rows * sizeof(uint64_t));

  cudaMemcpy(d_table->srcs, srcs, h_table->num_rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_table->dsts, dsts, h_table->num_rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
}


int main ( int argc, char * argv[] ) {

  // --
  // IO

  Table h_Data_Vtable    = readVertexTable(argv[2]);
  Table h_Data_Etable    = readEdgeTable(argv[3]);
  Table h_Pattern_Vtable = readVertexTable(argv[4]);
  Table h_Pattern_Etable = readEdgeTable(argv[5]);

  Table d_Data_Vtable;
  table2device(&d_Data_Vtable, &h_Data_Vtable);

  Table d_Data_Etable;
  table2device(&d_Data_Etable, &h_Data_Etable);

  Table d_Pattern_Vtable;
  table2device(&d_Pattern_Vtable, &h_Pattern_Vtable);

  Table d_Pattern_Etable;
  table2device(&d_Pattern_Etable, &h_Pattern_Etable);


  Graph h_Data_Graph    = constructGraph(&h_Data_Vtable, &h_Data_Etable);
  Graph h_Pattern_Graph = constructGraph(&h_Pattern_Vtable, &h_Pattern_Etable);

  Graph d_Data_Graph    = constructGraph(&d_Data_Vtable, &d_Data_Etable);
  Graph d_Pattern_Graph = constructGraph(&d_Pattern_Vtable, &d_Pattern_Etable);

  // --
  // Init
  WorkArrays h_WA;
  WorkArrays d_WA;
  initializeWorkArrays(
    &h_Data_Graph, &h_Pattern_Graph,
    &d_Data_Graph, &d_Pattern_Graph,
    h_WA, d_WA
  );

  // --
  // Run

  uint64_t DV = h_Data_Graph.num_vertices;
  uint64_t PV = h_Pattern_Graph.num_vertices;

  for (uint64_t i = 0; i < PV; i++) {
      run_iteration(
        &h_Data_Graph, &h_Pattern_Graph,
        &d_Data_Graph, &d_Pattern_Graph,
        h_WA, d_WA
      );
  }

  // --
  // Print results

  for (uint64_t i = 0; i < DV * PV; i ++) {
    printf("%e\n", h_WA.MU[i]);
  }
  return 0;
}

#endif