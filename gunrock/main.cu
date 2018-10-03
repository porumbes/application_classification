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

Table readEdgeTable(char * filename) {
  uint64_t num_rows, num_cols;
  FILE * tableFile = fopen(filename, "r");
  if (! tableFile) {printf("Cannot open file %s\n", filename); exit(1);}

  fscanf(tableFile, "%lu %lu", & num_rows, & num_cols);
  uint64_t * table = (uint64_t *) malloc(num_rows * num_cols * sizeof(uint64_t));

  for (uint64_t i = 0; i < num_rows * num_cols; i += num_cols) {
      fscanf(tableFile, "%lu", table + i);                          // read src id
      fscanf(tableFile, "%lu", table + i + 1);                      // read dst id

      for (uint64_t j = 2; j < num_cols; j ++) {
          fscanf(tableFile, "%lf", (double *) (table + i + j));     // read attribute
  }   }

  Table edgeTable;
  edgeTable.num_rows = num_rows;
  edgeTable.num_cols = num_cols;
  edgeTable.table    = table;
  return edgeTable;
}

Table readVertexTable(char * filename) {
  uint64_t num_rows, num_cols;
  FILE * tableFile = fopen(filename, "r");
  if (! tableFile) {printf("Cannot open file %s\n", filename); exit(1);}

  fscanf(tableFile, "%lu %lu", & num_rows, & num_cols);
  uint64_t * table = (uint64_t *) malloc(num_rows * num_cols * sizeof(uint64_t));

  for (uint64_t i = 0; i < num_rows * num_cols; i += num_cols) {
      fscanf(tableFile, "%lu", table + i);                          // read id

      for (uint64_t j = 1; j < num_cols; j ++) {
          fscanf(tableFile, "%lf", (double *) (table + i + j));     // read attribute
  }   }

  Table vertexTable;
  vertexTable.num_rows = num_rows;
  vertexTable.num_cols = num_cols;
  vertexTable.table    = table;
  return vertexTable;
}


void table2device(Table* d_table, Table* h_table) {
  d_table->num_rows = h_table->num_rows;
  d_table->num_cols = h_table->num_cols;

  cudaMalloc((void**)&d_table->table, h_table->num_rows * h_table->num_cols * sizeof(uint64_t));
  cudaMemcpy(d_table->table, h_table->table,
    h_table->num_rows * h_table->num_cols * sizeof(uint64_t), cudaMemcpyHostToDevice);

  uint64_t h_srcs[d_table->num_rows];
  uint64_t h_dsts[d_table->num_rows];
  for(uint64_t i = 0; i < d_table->num_rows; i++) {
    h_srcs[i] = h_table->table[i * d_table->num_cols];
    h_dsts[i] = h_table->table[i * d_table->num_cols + 1];
  }

  cudaMalloc((void**)&d_table->srcs, h_table->num_rows * sizeof(uint64_t));
  cudaMalloc((void**)&d_table->dsts, h_table->num_rows * sizeof(uint64_t));
  cudaMemcpy(d_table->srcs, h_srcs, h_table->num_rows * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_table->dsts, h_dsts, h_table->num_rows * sizeof(uint64_t), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_table->srcs_r, h_table->num_rows * sizeof(uint64_t));
  cudaMalloc((void**)&d_table->dsts_r, h_table->num_rows * sizeof(uint64_t));

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

  Graph d_Data_Graph    = constructGraph(&d_Data_Vtable, &d_Data_Etable);
  Graph d_Pattern_Graph = constructGraph(&d_Pattern_Vtable, &d_Pattern_Etable);

  // --
  // Init

  WorkArrays d_WA;
  initializeWorkArrays(&d_Data_Graph, &d_Pattern_Graph, d_WA);

  // --
  // Run

  const uint64_t DV = d_Data_Graph.num_vertices;
  const uint64_t PV = d_Pattern_Graph.num_vertices;

  for (uint64_t i = 0; i < PV; i++) {
      run_iteration(&d_Data_Graph, &d_Pattern_Graph, d_WA);
  }

  // --
  // Print results

  double *h_MU = (double *) malloc(DV * PV * sizeof(double));
  cudaMemcpy(h_MU, d_WA.MU, DV * PV * sizeof(double), cudaMemcpyDeviceToHost);

  for (uint64_t i = 0; i < DV * PV; i ++) {
    printf("%e\n", h_MU[i]);
  }
  return 0;
}

#endif