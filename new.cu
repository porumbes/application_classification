#ifndef __MAIN_CU
#define __MAIN_CU

#include <iostream>
#include "main.h"
#include "assert.h"

#include "thrust/device_vector.h"

#define THREAD 1024
// #define VERBOSE

typedef Int Int;
typedef Real Real;

struct cuda_timer_t {
  float time;

  cuda_timer_t() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~cuda_timer_t() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_); }
  
  float stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&time, start_, stop_);

    return microseconds();
  }
  
  float microseconds() { return (long long)(1000 * time); }

 private:
  cudaEvent_t start_, stop_;
};

void loadGraph(std::string inpath, Graph* d_graph) {

  FILE *ptr;
  ptr = fopen(inpath.c_str(), "rb");
    
  Int num_nodes;
  Int node_feat_dim;
  Int num_edges;
  Int edge_feat_dim;
  
  fread(&num_nodes,     sizeof(Int), 1, ptr);
  fread(&node_feat_dim, sizeof(Int), 1, ptr);
  fread(&num_edges,     sizeof(Int), 1, ptr);
  fread(&edge_feat_dim, sizeof(Int), 1, ptr);
  
  Real* node_feats = (Real*)malloc(num_nodes * node_feat_dim * sizeof(Real));
  Real* edge_feats = (Real*)malloc(num_edges * edge_feat_dim * sizeof(Real));
  Int*  srcs       = (Int*)malloc(num_edges * sizeof(Int));
  Int*  dsts       = (Int*)malloc(num_edges * sizeof(Int));
  
  fread(node_feats, sizeof(Real), num_nodes * node_feat_dim, ptr); 
  fread(edge_feats, sizeof(Real), num_edges * edge_feat_dim, ptr); 
  fread(srcs,       sizeof(Int),  num_edges,                 ptr); 
  fread(dsts,       sizeof(Int),  num_edges,                 ptr); 
  
#ifdef VERBOSE
        printf("----------------------------\n");
        std::cout << inpath << std::endl;
        printf("num_nodes     = %lu \n", num_nodes);
        printf("node_feat_dim = %lu \n", node_feat_dim);
        printf("num_edges     = %lu \n", num_edges);
        printf("edge_feat_dim = %lu \n", edge_feat_dim);
        printf("----------------------------\n");
#endif

  // -------------------------
  // Build graph

  d_graph->num_nodes     = num_nodes;
  d_graph->num_edges     = num_edges;
  d_graph->node_feat_dim = node_feat_dim;
  d_graph->edge_feat_dim = edge_feat_dim;

  cudaMalloc((void**)&d_graph->node_feats, num_nodes * node_feat_dim * sizeof(Real));
  cudaMalloc((void**)&d_graph->edge_feats, num_edges * edge_feat_dim * sizeof(Real));
  cudaMalloc((void**)&d_graph->srcs,       num_edges                 * sizeof(Int));
  cudaMalloc((void**)&d_graph->dsts,       num_edges                 * sizeof(Int));

  cudaMemcpy(d_graph->node_feats, node_feats, num_nodes * node_feat_dim * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph->edge_feats, edge_feats, num_edges * edge_feat_dim * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph->srcs,       srcs,       num_edges                 * sizeof(Int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph->dsts,       dsts,       num_edges                 * sizeof(Int), cudaMemcpyHostToDevice);
 
  cudaMalloc((void**)&d_graph->srcs_r, num_edges * sizeof(Int));
  cudaMalloc((void**)&d_graph->dsts_r, num_edges * sizeof(Int));
}


int main ( int argc, char * argv[] ) {

  // --
  // IO

  Graph data;
  Graph patt;
  
  loadGraph(argv[1], &data);
  loadGraph(argv[2], &patt);

  // --
  // Allocate memory

  Real *CV, *CE, *MU, *RE, *FE, *VR, *VF, *VRmax, *VFmax, *RMax, *FMax;

  // Real *Cnull; // Ignoring for now

  cudaMalloc((void **)&CV,    data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMalloc((void **)&MU,    data.num_nodes * patt.num_nodes * sizeof(Real));

  cudaMalloc((void **)&CE,    data.num_edges * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&RE,    data.num_edges * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&FE,    data.num_edges * patt.num_edges * sizeof(Real));

  cudaMalloc((void **)&VR,    data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VF,    data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VRmax,                  patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VFmax,                  patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&RMax,  data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&FMax,  data.num_nodes * patt.num_edges * sizeof(Real));
  // cudaMalloc((void **)&Cnull,                  patt.num_edges * sizeof(Real));

  // --
  // Initialize algorithm

  cuda_timer_t timer;
  timer.start();

  auto cdist_node = [=] __device__(Int const& offset) {
    Int i = offset / patt.num_nodes;
    Int j = offset % patt.num_nodes;

    Int dim = patt.node_feat_dim;

    Real* vec1 = patt.node_feats + (j * dim);
    Real* vec2 = data.node_feats + (i * dim);

    Real dist = 0.0;
    for (int i = 0; i < dim; i++)
      dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    dist = sqrt(dist);

    CV[offset] = dist;
    MU[offset] = -dist;
  };
  
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator<Int>(0),
    thrust::make_counting_iterator<Int>(patt.num_nodes * data.num_nodes),
    cdist_node
  );

  auto cdist_edge = [=] __device__(Int const& offset) {
    Int i     = offset / patt.num_edges;
    Int j     = offset % patt.num_edges;
    Int dim    = patt.edge_feat_dim;
    Real* vec1 = patt.edge_feats + j * dim;
    Real* vec2 = data.edge_feats + i * dim;

    Real dist = 0.0;
    for (int i = 0; i < dim; i++)
      dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    dist = sqrt(dist);

    CE[offset] = dist;
    RE[offset] = - dist;
    FE[offset] = - dist;
  };
  
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator<Int>(0),
    thrust::make_counting_iterator<Int>(patt.num_edges * data.num_edges),
    cdist_edge
  );

  // >>
  cuda_timer_t timer_;
  timer_.start();

  // Normalize distance matrices (could all happen in parallel)
  ac::host::ColumnSoftmax2(data.num_nodes, patt.num_nodes, CV);
  ac::host::ColumnSoftmax2(data.num_nodes, patt.num_nodes, MU);
  ac::host::ColumnSoftmax2(data.num_edges, patt.num_edges, CE);
  ac::host::ColumnSoftmax2(data.num_edges, patt.num_edges, RE);
  ac::host::ColumnSoftmax2(data.num_edges, patt.num_edges, FE);

  auto RepeatColumnsByPatternEdges_op = [=] __device__(Int const& offset) {
      Int i     = offset / patt.num_edges;
      Int j     = offset % patt.num_edges;
      VR[offset] = MU[i * patt.num_nodes + patt.srcs[j]];
      VF[offset] = MU[i * patt.num_nodes + patt.dsts[j]];
  };
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator<Int>(0),
    thrust::make_counting_iterator<Int>(data.num_nodes * patt.num_edges),
    RepeatColumnsByPatternEdges_op
  );

  // Hardcode Cnull to 0
  // cudaMemset(Cnull, 0, patt.num_edges * sizeof(Real));

  // Compute max over columns of VF/VR
  ac::host::ColumnMax2(data.num_nodes, patt.num_edges, VF, VFmax);
  ac::host::EdgeMaxReduce2(
    data.num_edges, data.num_nodes, patt.num_edges,
    VFmax,
    RE,
    RMax,
    data.srcs
  );
  
  ac::host::ColumnMax2(data.num_nodes, patt.num_edges, VR, VRmax);
  ac::host::EdgeMaxReduce2(
    data.num_edges, data.num_nodes, patt.num_edges,
    VRmax,
    FE,
    FMax,
    data.dsts
  );

  long long elapsed_ = timer_.stop();
  std::cerr << "elapsed_=" << elapsed_ << std::endl;

  // --
  // Run

  Real* MU_tmp;
  Real* RE_tmp;
  Real* FE_tmp;
  cudaMalloc(&MU_tmp, patt.num_nodes * sizeof(Real));
  cudaMalloc(&RE_tmp, patt.num_edges * sizeof(Real));
  cudaMalloc(&FE_tmp, patt.num_edges * sizeof(Real));

  for (Int i = 0; i < patt.num_nodes; i++) {
    
    auto RepeatColumnsByPatternEdgesSubtract_op = [=] __device__(Int const& offset) {
      Int i      = offset / patt.num_edges;
      Int j      = offset % patt.num_edges;
      VF[offset] = MU[i * patt.num_nodes + patt.dsts[j]] - FMax[offset];
      VR[offset] = MU[i * patt.num_nodes + patt.srcs[j]] - RMax[offset];
    };
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      thrust::make_counting_iterator<Int>(data.num_nodes * patt.num_edges),
      RepeatColumnsByPatternEdgesSubtract_op
    );

    auto RepeatColumnsByDataEdges_op = [=] __device__(Int const& offset) {
      Int ij     = offset / patt.num_edges;
      Int km     = offset % patt.num_edges;
      FE[offset] = VR[data.srcs[ij] * patt.num_edges + km] - CE[offset];
      RE[offset] = VF[data.srcs[ij] * patt.num_edges + km] - CE[offset];
    };
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      thrust::make_counting_iterator<Int>(data.num_edges * patt.num_edges),
      RepeatColumnsByDataEdges_op
    );
    
    ac::host::ColumnMax2(data.num_nodes, patt.num_edges, VF, VFmax);               // x.max(axis=0)
    ac::host::ColumnSoftmax2_prealloc(data.num_edges, patt.num_edges, RE, RE_tmp); // x.softmax(axis=0)
    ac::host::EdgeMaxReduce2(
      data.num_edges, data.num_nodes, patt.num_edges,
      VFmax, RE, RMax,
      data.srcs
    );
    
    ac::host::ColumnMax2(data.num_nodes, patt.num_edges, VR, VRmax);               // x.max(axis=0)
    ac::host::ColumnSoftmax2_prealloc(data.num_edges, patt.num_edges, FE, FE_tmp); // x.softmax(axis=0)
    ac::host::EdgeMaxReduce2(
      data.num_edges, data.num_nodes, patt.num_edges,
      VRmax, FE, FMax,
      data.dsts
    );

    ac::host::ComputeMU2(
      data.num_nodes, patt.num_edges,
      data.num_nodes, patt.num_nodes,
      CV,
      FMax,
      RMax,
      patt.srcs,
      patt.dsts,
      MU
    );
    
    ac::host::ColumnSoftmax2_prealloc(data.num_nodes, patt.num_nodes, MU, MU_tmp);
  }

  long long elapsed = timer.stop();
  std::cerr << "elapsed=" << elapsed << std::endl;

  // --
  // Copy results to host and print

  Real *h_MU = (Real *) malloc(data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMemcpy(h_MU, MU, data.num_nodes * patt.num_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
  for (Int i = 0; i < data.num_nodes * patt.num_nodes; i ++) printf("%e\n", h_MU[i]);
}

#endif