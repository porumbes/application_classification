#include "main.h"

Graph constructGraph(Table * Vtable, Table * Etable) {
  Graph graph;
  graph.Vtable       = * Vtable;
  graph.Etable       = * Etable;
  graph.num_edges    = Etable->num_rows;
  graph.num_vertices = Vtable->num_rows;
  graph.starts = (uint64_t *) malloc((graph.num_vertices + 1) * sizeof(uint64_t));

// edge list of vertices [0 .. first vertex] start at 0
  uint64_t first_vertex = Etable->table[0];
  for (uint64_t j = 0; j <= first_vertex; j ++) graph.starts[j] = 0;

// for each edge i whose src id is different than the src id of the previous edge
//     edge list of vertices [prev_src + 1 .. this_src] start at i
// #pragma omp parallel for
  for (uint64_t i = 1; i < graph.num_edges; i ++) {
      uint64_t this_src = Etable->table[i * Etable->num_cols];
      uint64_t prev_src = Etable->table[(i - 1) * Etable->num_cols];

      if (this_src != prev_src) {
         for (uint64_t j = prev_src + 1; j <= this_src; j ++) graph.starts[j] = i;
  }   }

// edge list of vertices [last_vertex + 1 .. num_vertices] start at num_edges
  uint64_t last_vertex = Etable->table[(Etable->num_rows - 1) * Etable->num_cols];
  for (uint64_t j = last_vertex + 1; j <= graph.num_vertices; j ++) graph.starts[j] = graph.num_edges;

  return graph;
}
