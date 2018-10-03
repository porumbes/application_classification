#ifndef __MAIN_CU
#define __MAIN_CU

#include <iostream>
#include "main.h"
#include "assert.h"

void loadGraph(char * node_filename, char * edge_filename, Graph* d_graph) {

  // -------------------------
  // Read nodes

  IntT num_nodes, node_line_length, node_feat_dim;
  FILE * nodeFile = fopen(node_filename, "r");
  if (! nodeFile) {printf("Cannot open node file %s\n", node_filename); exit(1);}

  fscanf(nodeFile, "%lu %lu", & num_nodes, & node_line_length);
  node_feat_dim = node_line_length - 1;

  FloatT * node_feats = (FloatT *) malloc(num_nodes * node_feat_dim * sizeof(FloatT));
  IntT * node_ids = (IntT *) malloc(num_nodes * sizeof(IntT));

  for (IntT node_idx = 0; node_idx < num_nodes; node_idx++) {
    fscanf(nodeFile, "%lu", node_ids + node_idx);
    assert(node_ids[node_idx] == node_idx);
    for (IntT feat_idx = 0; feat_idx < node_feat_dim; feat_idx ++) {
        fscanf(nodeFile, "%lf", (FloatT *) (node_feats + (node_feat_dim * node_idx) + feat_idx));
    }
  }

  // -------------------------
  // Read edges

  IntT num_edges, edge_line_length, edge_feat_dim;
  FILE * edgeFile = fopen(edge_filename, "r");
  if (! edgeFile) {printf("Cannot open file %s\n", edge_filename); exit(1);}

  fscanf(edgeFile, "%lu %lu", & num_edges, & edge_line_length);
  edge_feat_dim = edge_line_length - 2;

  FloatT * edge_feats = (FloatT *) malloc(num_edges * edge_feat_dim * sizeof(FloatT));
  IntT * srcs         = (IntT *) malloc(num_edges * sizeof(IntT));
  IntT * dsts         = (IntT *) malloc(num_edges * sizeof(IntT));

  for (IntT edge_idx = 0; edge_idx < num_edges; edge_idx++) {
    fscanf(edgeFile, "%lu", srcs + edge_idx); // read src
    fscanf(edgeFile, "%lu", dsts + edge_idx); // read dst
    for (IntT feat_idx = 0; feat_idx < edge_feat_dim; feat_idx ++) {
        fscanf(edgeFile, "%lf", (FloatT *) (edge_feats + (edge_feat_dim * edge_idx) + feat_idx));
    }
  }

  // -------------------------
  // Build graph

  d_graph->num_nodes     = num_nodes;
  d_graph->num_edges     = num_edges;
  d_graph->node_feat_dim = node_feat_dim;
  d_graph->edge_feat_dim = edge_feat_dim;

  cudaMalloc((void**)&d_graph->node_feats, num_nodes * node_feat_dim * sizeof(FloatT));
  cudaMalloc((void**)&d_graph->edge_feats, num_edges * edge_feat_dim * sizeof(FloatT));
  cudaMalloc((void**)&d_graph->srcs, num_edges * sizeof(IntT));
  cudaMalloc((void**)&d_graph->dsts, num_edges * sizeof(IntT));

  cudaMemcpy(d_graph->node_feats, node_feats, num_nodes * node_feat_dim * sizeof(FloatT), cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph->edge_feats, edge_feats, num_edges * edge_feat_dim * sizeof(FloatT), cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph->srcs, srcs, num_edges * sizeof(IntT), cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph->dsts, dsts, num_edges * sizeof(IntT), cudaMemcpyHostToDevice);
}


int main ( int argc, char * argv[] ) {

  // --
  // IO

  char* data_node_path = argv[2];
  char* data_edge_path = argv[3];
  char* patt_node_path = argv[4];
  char* patt_edge_path = argv[5];

  Graph data;
  loadGraph(data_node_path, data_edge_path, &data);

  Graph patt;
  loadGraph(patt_node_path, patt_edge_path, &patt);

  // --
  // Allocate memory

  FloatT *CV,
         *CE,
         *Cnull,
         *MU,
         *RE,
         *FE,
         *VR,
         *VF,
         *VRmax,
         *VFmax,
         *RMax,
         *FMax;

  cudaMalloc((void **)&CV,    data.num_nodes * patt.num_nodes * sizeof(FloatT));
  cudaMalloc((void **)&CE,    data.num_edges * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&Cnull, patt.num_edges *                  sizeof(FloatT));
  cudaMalloc((void **)&MU,    data.num_nodes * patt.num_nodes * sizeof(FloatT));
  cudaMalloc((void **)&RE,    data.num_edges * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&FE,    data.num_edges * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&VR,    data.num_nodes * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&VF,    data.num_nodes * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&VRmax, patt.num_edges *                  sizeof(FloatT));
  cudaMalloc((void **)&VFmax, patt.num_edges *                  sizeof(FloatT));
  cudaMalloc((void **)&RMax,  data.num_nodes * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&FMax,  data.num_nodes * patt.num_edges * sizeof(FloatT));

  // --
  // Initialize algorithm

  // Node-node distance matrix
  ac::Init_CV_MU(&data, &patt, CV, MU);

  // Edge-edge distance matrix
  ac::Init_CE_RE_FE(&data, &patt, CE, RE, FE);

  // Normalize distance matrices (could all happen in parallel)
  ac::ColumnSoftmax(data.num_nodes, patt.num_nodes, CV);
  ac::ColumnSoftmax(data.num_nodes, patt.num_nodes, MU);
  ac::ColumnSoftmax(data.num_edges, patt.num_edges, CE);
  ac::ColumnSoftmax(data.num_edges, patt.num_edges, RE);
  ac::ColumnSoftmax(data.num_edges, patt.num_edges, FE);

  // Repeat columns of MU by pattern edgelist
  ac::Init_VR_VF(&patt, data.num_nodes, MU, VR, VF);

  // Hardcode to 0
  cudaMemset(Cnull, 0, patt.num_edges * sizeof(FloatT));

  // Compute max over columns of VF/VR
  ac::VFmax_VRmax(data.num_nodes, patt.num_edges, VF, VR, VFmax, VRmax);

  // Max reduce over edges adjacent to data nodes
  ac::FMax(&data, patt.num_edges, Cnull, VRmax, FE, FMax);
  ac::RMax(&data, patt.num_edges, Cnull, VFmax, RE, RMax);

  // --
  // Run

  for (IntT i = 0; i < patt.num_nodes; i++) {
    // Repeat columns of (MU - FMax) by pattern edgelist
    ac::VF_VR(&patt, data.num_nodes, MU, FMax, RMax, VF, VR);

    // Compute max over columns of VF/VR
    ac::VFmax_VRmax(data.num_nodes, patt.num_edges, VF, VR, VFmax, VRmax);

    ac::FE_RE(&data, patt.num_edges, CE, VF, VR, FE, RE);
    ac::ColumnSoftmax(data.num_edges, patt.num_edges, FE);
    ac::ColumnSoftmax(data.num_edges, patt.num_edges, RE);

    // Max aggregation over edges adjacent to data nodes
    ac::FMax(&data, patt.num_edges, Cnull, VRmax, FE, FMax);
    ac::RMax(&data, patt.num_edges, Cnull, VFmax, RE, RMax);

    // Sum reduce over edges adjacent to pattern nodes
    ac::UpdateMU(&patt, data.num_nodes, CV, FMax, RMax, MU);
    ac::ColumnSoftmax(data.num_nodes, patt.num_nodes, MU);
  }

  // --
  // Copy results to host and print

  FloatT *h_MU = (FloatT *) malloc(data.num_nodes * patt.num_nodes * sizeof(FloatT));
  cudaMemcpy(h_MU, MU, data.num_nodes * patt.num_nodes * sizeof(FloatT), cudaMemcpyDeviceToHost);

  for (IntT i = 0; i < data.num_nodes * patt.num_nodes; i ++) {
    printf("%e\n", h_MU[i]);
  }

  // --
  // Free memory

  cudaFree(CV);
  cudaFree(CE);
  cudaFree(Cnull);
  cudaFree(MU);
  cudaFree(RE);
  cudaFree(FE);
  cudaFree(VR);
  cudaFree(VF);
  cudaFree(VRmax);
  cudaFree(VFmax);
  cudaFree(RMax);
  cudaFree(FMax);
  free(h_MU);

  return 0;
}

#endif