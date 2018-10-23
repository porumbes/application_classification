#ifndef __MAIN_CU
#define __MAIN_CU

#include <iostream>
#include "main.h"
#include "assert.h"

#define THREAD 1024

void loadGraph(char * node_filename, char * edge_filename, Graph* d_graph) {

  // -------------------------
  // Read nodes

  IntT num_nodes, node_line_length;
  FILE * nodeFile = fopen(node_filename, "r");
  if (! nodeFile) {printf("Cannot open node file %s\n", node_filename); exit(1);}

  fscanf(nodeFile, "%lu %lu", & num_nodes, & node_line_length);
  IntT node_feat_dim = node_line_length - 1;

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

  IntT num_edges, edge_line_length;
  FILE * edgeFile = fopen(edge_filename, "r");
  if (! edgeFile) {printf("Cannot open file %s\n", edge_filename); exit(1);}

  fscanf(edgeFile, "%lu %lu", & num_edges, & edge_line_length);
  IntT edge_feat_dim = edge_line_length - 2;

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

  cudaMalloc((void**)&d_graph->srcs_r, num_edges * sizeof(IntT));
  cudaMalloc((void**)&d_graph->dsts_r, num_edges * sizeof(IntT));
  cudaMalloc((void**)&d_graph->map_r, num_edges * sizeof(IntT));
}


int main ( int argc, char * argv[] ) {

  // --
  // IO


  Graph data;
  char* data_node_path = argv[1];
  char* data_edge_path = argv[2];
  loadGraph(data_node_path, data_edge_path, &data);

  Graph patt;
  char* patt_node_path = argv[3];
  char* patt_edge_path = argv[4];
  loadGraph(patt_node_path, patt_edge_path, &patt);

  // --
  // Allocate memory

  FloatT *CV,
         *CE,
         *MU,
         *RE,
         *FE,
         *VR,
         *VF,
         *VRmax,
         *VFmax,
         *RMax,
         *FMax;

  // FloatT *Cnull; // Ignoring for now

  cudaMalloc((void **)&CV,    data.num_nodes * patt.num_nodes * sizeof(FloatT));
  cudaMalloc((void **)&MU,    data.num_nodes * patt.num_nodes * sizeof(FloatT));

  cudaMalloc((void **)&CE,    data.num_edges * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&RE,    data.num_edges * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&FE,    data.num_edges * patt.num_edges * sizeof(FloatT));

  cudaMalloc((void **)&VR,    data.num_nodes * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&VF,    data.num_nodes * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&VRmax,                  patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&VFmax,                  patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&RMax,  data.num_nodes * patt.num_edges * sizeof(FloatT));
  cudaMalloc((void **)&FMax,  data.num_nodes * patt.num_edges * sizeof(FloatT));
  // cudaMalloc((void **)&Cnull,                  patt.num_edges * sizeof(FloatT));

  IntT block_vv = 1 + (data.num_nodes * patt.num_nodes) / THREAD;
  IntT block_ee = 1 + (data.num_edges * patt.num_edges) / THREAD;
  IntT block_ve = 1 + (data.num_nodes * patt.num_edges) / THREAD;

  // --
  // Start timer

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // --
  // Initialize algorithm

  ac::host::SortEdges(data.srcs, data.dsts, data.srcs_r, data.dsts_r, data.map_r, data.num_edges);
  ac::host::SortEdges(patt.srcs, patt.dsts, patt.srcs_r, patt.dsts_r, patt.map_r, patt.num_edges);

  // Node-node distance matrix
  ac::device::NodePairwiseNorm<<<block_vv, THREAD>>>(
    data.num_nodes,
    patt.num_nodes,
    CV,
    MU,
    data.node_feats,
    patt.node_feats,
    data.node_feat_dim
  );


  // Edge-edge distance matrix
  ac::device::EdgePairwiseNorm<<<block_ee, THREAD>>>(
    data.num_edges,
    patt.num_edges,
    CE,
    RE,
    FE,
    data.edge_feats,
    patt.edge_feats,
    data.edge_feat_dim
  );

  // Normalize distance matrices (could all happen in parallel)
  ac::host::RowSoftmax(patt.num_nodes, data.num_nodes, CV);
  ac::host::RowSoftmax(patt.num_nodes, data.num_nodes, MU);
  ac::host::RowSoftmax(patt.num_edges, data.num_edges, CE);
  ac::host::RowSoftmax(patt.num_edges, data.num_edges, RE);
  ac::host::RowSoftmax(patt.num_edges, data.num_edges, FE);

  // Repeat columns of MU by pattern edgelist
  ac::device::RepeatRowsByPatternEdges<<<block_ve, THREAD>>>(
    data.num_nodes,
    patt.num_edges,
    patt.num_nodes,
    MU,
    VR,
    VF,
    patt.srcs,
    patt.dsts
  );

  // // Hardcode Cnull to 0
  // // cudaMemset(Cnull, 0, patt.num_edges * sizeof(FloatT));

  // Compute max over columns of VF/VR
  ac::host::RowMax(patt.num_edges, data.num_nodes, VF, VFmax);
  ac::host::RowMax(patt.num_edges, data.num_nodes, VR, VRmax);

  // Max reduce over edges adjacent to data nodes
  ac::host::EdgeMaxReduce(data.num_edges, data.num_nodes, patt.num_edges,
    VRmax, FE, FMax,
    data.dsts_r, data.map_r
  );

  ac::host::EdgeMaxReduce(
    data.num_edges, data.num_nodes, patt.num_edges,
    VFmax, RE, RMax,
    data.srcs, NULL
  );

  // --
  // Run

  for (IntT i = 0; i < patt.num_nodes; i++) {
    // Repeat columns of (MU - FMax) by pattern edgelist
    ac::device::RepeatRowsByPatternEdgesSubtract<<<block_ve, THREAD>>>(
      data.num_nodes,
      patt.num_edges,
      patt.num_nodes,
      MU,
      VR,
      VF,
      FMax,
      RMax,
      patt.srcs,
      patt.dsts
    );

    // Compute max over columns of VF/VR
    ac::host::RowMax(patt.num_edges, data.num_nodes, VF, VFmax);
    ac::host::RowMax(patt.num_edges, data.num_nodes, VR, VRmax);

    // Repeat rows of VF/VR by data srcs
    ac::device::RepeatRowsByDataEdges<<<block_ee, THREAD>>>(
      data.num_edges,
      patt.num_edges,
      data.num_nodes,
      CE,
      VR,
      VF,
      FE,
      RE,
      data.srcs
    );
    ac::host::RowSoftmax(patt.num_edges, data.num_edges, FE);
    ac::host::RowSoftmax(patt.num_edges, data.num_edges, RE);

    // Max aggregation over edges adjacent to data nodes
    ac::host::EdgeMaxReduce(data.num_edges, data.num_nodes, patt.num_edges,
      VRmax, FE, FMax,
      data.dsts_r, data.map_r
    );

    ac::host::EdgeMaxReduce(
      data.num_edges, data.num_nodes, patt.num_edges,
      VFmax, RE, RMax,
      data.srcs, NULL
    );

    // Replace columns of MU w/ sum over FMax/RMax of adjacent edges + subtract CV
    ac::host::ComputeMU(&patt, data.num_nodes, CV, FMax, RMax, MU);
    ac::host::RowSoftmax(patt.num_nodes, data.num_nodes, MU);
    // break;
  }

  // --
  // Stop timer

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cerr << "elapsed_ms=" << milliseconds << std::endl;

  // --
  // Copy results to host and print

  FloatT *h_MU = (FloatT *) malloc(data.num_nodes * patt.num_nodes * sizeof(FloatT));
  cudaMemcpy(h_MU, MU, data.num_nodes * patt.num_nodes * sizeof(FloatT), cudaMemcpyDeviceToHost);
  for (IntT i = 0; i < data.num_nodes * patt.num_nodes; i ++) printf("%f\n", h_MU[i]);

  // FloatT *h_CE = (FloatT *) malloc(data.num_edges * patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_CE, CE, data.num_edges * patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < data.num_edges * patt.num_edges; i ++) printf("%f\n", h_CE[i]);

  // FloatT *h_FE = (FloatT *) malloc(data.num_edges * patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_FE, FE, data.num_edges * patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < data.num_edges * patt.num_edges; i ++) printf("%f\n", h_FE[i]);

  // FloatT *h_RE = (FloatT *) malloc(data.num_edges * patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_RE, RE, data.num_edges * patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < data.num_edges * patt.num_edges; i ++) printf("%f\n", h_RE[i]);

  // FloatT *h_VF = (FloatT *) malloc(data.num_nodes * patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_VF, VF, data.num_nodes * patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < data.num_nodes * patt.num_edges; i ++) printf("%f\n", h_VF[i]);

  // FloatT *h_VR = (FloatT *) malloc(data.num_nodes * patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_VR, VR, data.num_nodes * patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < data.num_nodes * patt.num_edges; i ++) printf("%f\n", h_VR[i]);

  // FloatT *h_VRmax = (FloatT *) malloc(patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_VRmax, VRmax, patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < patt.num_edges; i ++) printf("%f\n", h_VRmax[i]);

  // FloatT *h_VFmax = (FloatT *) malloc(patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_VFmax, VFmax, patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < patt.num_edges; i ++) printf("%f\n", h_VFmax[i]);

  // FloatT *h_RMax = (FloatT *) malloc(data.num_nodes * patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_RMax, RMax, data.num_nodes * patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < data.num_nodes * patt.num_edges; i ++) printf("%f\n", h_RMax[i]);

  // FloatT *h_FMax = (FloatT *) malloc(data.num_nodes * patt.num_edges * sizeof(FloatT));
  // cudaMemcpy(h_FMax, FMax, data.num_nodes * patt.num_edges * sizeof(FloatT), cudaMemcpyDeviceToHost);
  // for (IntT i = 0; i < data.num_nodes * patt.num_edges; i ++) printf("%f\n", h_FMax[i]);

  // --
  // Free memory

  cudaFree(CV);
  cudaFree(CE);
  // cudaFree(Cnull);
  cudaFree(MU);
  cudaFree(RE);
  cudaFree(FE);
  cudaFree(VR);
  cudaFree(VF);
  cudaFree(VRmax);
  cudaFree(VFmax);
  cudaFree(RMax);
  cudaFree(FMax);
  // free(h_MU);

  return 0;
}

#endif