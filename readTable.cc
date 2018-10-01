#include "main.h"

// Data files format assumed to be
//
//      Row 0:			numbers_of_rows, number_of_cols 
//      Row 1..number_of_rows:	id, dst id, attr 0, attr 1, ....
//
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


// Data files format assumed to be
//
//      Row 0:			numbers_of_rows, number_of_cols 
//      Row 1..number_of_rows:	id, dst id, attr 0, attr 1, ....
//
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
