#ifndef __MAIN_CU
#define __MAIN_CU

#include <iostream>
#include "main.h"
#include "kernels.cuh"
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

  Graph d_data_graph;
  loadGraph(data_node_path, data_edge_path, &d_data_graph);

  Graph d_patt_graph;
  loadGraph(patt_node_path, patt_edge_path, &d_patt_graph);

  const IntT DV = d_data_graph.num_nodes;
  const IntT DE = d_data_graph.num_edges;
  const IntT PV = d_patt_graph.num_nodes;
  const IntT PE = d_patt_graph.num_edges;

  // --
  // Allocation

  WorkArrays d_WA;
  cudaMalloc((void **)&d_WA.CV,    DV * PV * sizeof(FloatT));
  cudaMalloc((void **)&d_WA.CE,    DE * PE * sizeof(FloatT));
  cudaMalloc((void **)&d_WA.Cnull, PE *      sizeof(FloatT));
  cudaMalloc((void **)&d_WA.MU,    DV * PV * sizeof(FloatT));
  cudaMalloc((void **)&d_WA.RE,    DE * PE * sizeof(FloatT));
  cudaMalloc((void **)&d_WA.FE,    DE * PE * sizeof(FloatT));
  cudaMalloc((void **)&d_WA.VR,    DV * PE * sizeof(FloatT));
  cudaMalloc((void **)&d_WA.VF,    DV * PE * sizeof(FloatT));
  cudaMalloc((void **)&d_WA.VRmax, PE *      sizeof(FloatT));
  cudaMalloc((void **)&d_WA.VFmax, PE *      sizeof(FloatT));
  cudaMalloc((void **)&d_WA.RMax,  DV * PE * sizeof(FloatT));
  cudaMalloc((void **)&d_WA.FMax,  DV * PE * sizeof(FloatT));

  // --
  // Initialization

  d_Init_CV_MU(&d_data_graph, &d_patt_graph, d_WA.CV, d_WA.MU);
  d_NormProb(DV, PV, d_WA.CV);
  d_NormProb(DV, PV, d_WA.MU);
  d_Init_VR_VF(&d_data_graph, &d_patt_graph, d_WA.MU, d_WA.VR, d_WA.VF);
  d_Init_CE_RE_FE(&d_data_graph, &d_patt_graph, d_WA.CE, d_WA.RE, d_WA.FE);
  d_NormProb(DE, PE, d_WA.CE);
  d_NormProb(DE, PE, d_WA.RE);
  d_NormProb(DE, PE, d_WA.FE);
  cudaMemset(d_WA.Cnull, 0, PE * sizeof(FloatT));
  d_VFmax_VRmax(&d_data_graph, &d_patt_graph, d_WA.VF, d_WA.VR, d_WA.VFmax, d_WA.VRmax);
  d_FMax(&d_data_graph, &d_patt_graph, d_WA.Cnull, d_WA.VRmax, d_WA.FE, d_WA.FMax);
  d_RMax(&d_data_graph, &d_patt_graph, d_WA.Cnull, d_WA.VFmax, d_WA.RE, d_WA.RMax);

  // --
  // Run

  for (IntT i = 0; i < PV; i++) {
    d_VF_VR(&d_data_graph, &d_patt_graph, d_WA.MU, d_WA.FMax, d_WA.RMax, d_WA.VF, d_WA.VR);
    d_VFmax_VRmax(&d_data_graph, &d_patt_graph, d_WA.VF, d_WA.VR, d_WA.VFmax, d_WA.VRmax);
    d_FE_RE(&d_data_graph, &d_patt_graph, d_WA.CE, d_WA.VF, d_WA.VR, d_WA.FE, d_WA.RE);
    d_NormProb(DE, PE, d_WA.FE);
    d_NormProb(DE, PE, d_WA.RE);

    d_FMax(&d_data_graph, &d_patt_graph, d_WA.Cnull, d_WA.VRmax, d_WA.FE, d_WA.FMax);
    d_RMax(&d_data_graph, &d_patt_graph, d_WA.Cnull, d_WA.VFmax, d_WA.RE, d_WA.RMax);
    d_UpdateMU(&d_data_graph, &d_patt_graph, d_WA.CV, d_WA.FMax, d_WA.RMax, d_WA.MU);
    d_NormProb(DV, PV, d_WA.MU);
  }

  // --
  // Copy results to host and print

  FloatT *h_MU = (FloatT *) malloc(DV * PV * sizeof(FloatT));
  cudaMemcpy(h_MU, d_WA.MU, DV * PV * sizeof(FloatT), cudaMemcpyDeviceToHost);

  for (IntT i = 0; i < DV * PV; i ++) {
    printf("%e\n", h_MU[i]);
  }
  return 0;
}

#endif