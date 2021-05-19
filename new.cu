#ifndef __MAIN_CU
#define __MAIN_CU

#include <iostream>
#include "main.h"
#include "nvToolsExt.h"

#include "thrust/device_vector.h"

#define THREAD 1024
// #define VERBOSE

template <typename val>
void transpose(val* in, val* out, Int num_rows, Int num_cols) {

  auto op = [=]__device__(Int const& offset) {
    Int src_row = offset / num_cols;
    Int src_col = offset % num_cols;
    out[src_col * num_rows + src_row] = in[offset];
  };

  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator<Int>(0),
    thrust::make_counting_iterator<Int>(num_rows * num_cols),
    op
  );
}


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

  Real *MU, *VRmax, *VFmax;
  Real *CV_t, *CE_t, *MU_t, *RE_t, *FE_t, *VR_t, *VF_t, *RMax_t, *FMax_t;

  // Real *Cnull; // Ignoring for now

  cudaMalloc((void **)&MU,    data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMalloc((void **)&VRmax,                  patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VFmax,                  patt.num_edges * sizeof(Real));

  cudaMalloc((void **)&CV_t,    data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMalloc((void **)&MU_t,    data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMalloc((void **)&CE_t,    data.num_edges * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&RE_t,    data.num_edges * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&FE_t,    data.num_edges * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VR_t,    data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VF_t,    data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&RMax_t,  data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&FMax_t,  data.num_nodes * patt.num_edges * sizeof(Real));

  // --
  // Initialize algorithm

  cuda_timer_t timer;
  timer.start();
  
  nvtxRangePushA("prep");
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

    CV_t[j * data.num_nodes + i] = dist;
    MU_t[j * data.num_nodes + i] = -dist;
  };
  
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator<Int>(0),
    thrust::make_counting_iterator<Int>(patt.num_nodes * data.num_nodes),
    cdist_node
  );

  auto cdist_edge = [=] __device__(Int const& offset) {
    Int i      = offset / patt.num_edges;
    Int j      = offset % patt.num_edges;
    Int dim    = patt.edge_feat_dim;
    Real* vec1 = patt.edge_feats + j * dim;
    Real* vec2 = data.edge_feats + i * dim;

    Real dist = 0.0;
    for (int i = 0; i < dim; i++)
      dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    dist = sqrt(dist);

    CE_t[j * data.num_edges + i] = dist;
    RE_t[j * data.num_edges + i] = - dist;
    FE_t[j * data.num_edges + i] = - dist;
  };
  
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator<Int>(0),
    thrust::make_counting_iterator<Int>(patt.num_edges * data.num_edges),
    cdist_edge
  );

  ac::host::RowSoftmax2(patt.num_nodes, data.num_nodes, CV_t);
  ac::host::RowSoftmax2(patt.num_nodes, data.num_nodes, MU_t);
  ac::host::RowSoftmax2(patt.num_edges, data.num_edges, CE_t);
  ac::host::RowSoftmax2(patt.num_edges, data.num_edges, RE_t);
  ac::host::RowSoftmax2(patt.num_edges, data.num_edges, FE_t);

  auto RepeatColumnsByPatternEdges_op = [=] __device__(Int const& offset) {
      Int i                        = offset / patt.num_edges;
      Int j                        = offset % patt.num_edges;
      VR_t[i + data.num_nodes * j] = MU_t[i + data.num_nodes * patt.srcs[j]];
      VF_t[i + data.num_nodes * j] = MU_t[i + data.num_nodes * patt.dsts[j]];
  };
  thrust::for_each(
    thrust::device,
    thrust::make_counting_iterator<Int>(0),
    thrust::make_counting_iterator<Int>(data.num_nodes * patt.num_edges),
    RepeatColumnsByPatternEdges_op
  );

  ac::host::RowMax2(patt.num_edges, data.num_nodes, VF_t, VFmax);
  ac::host::RowMax2(patt.num_edges, data.num_nodes, VR_t, VRmax);

  ac::host::EdgeMaxReduce2_t(
    data.num_edges, data.num_nodes, patt.num_edges,
    VFmax,
    RE_t,
    RMax_t,
    data.srcs
  );
  
  ac::host::EdgeMaxReduce2_t(
    data.num_edges, data.num_nodes, patt.num_edges,
    VRmax,
    FE_t,
    FMax_t,
    data.dsts
  );
  
  nvtxRangePop();
  
  // --
  // Run
  
  Real* MU_tmp;
  Real* RE_tmp;
  Real* FE_tmp;
  cudaMalloc(&MU_tmp, patt.num_nodes * sizeof(Real));
  cudaMalloc(&RE_tmp, patt.num_edges * sizeof(Real));
  cudaMalloc(&FE_tmp, patt.num_edges * sizeof(Real));

  for (Int i = 0; i < patt.num_nodes; i++) {
    nvtxRangePushA("loop");
    
    nvtxRangePushA("step1");
    auto RepeatColumnsByPatternEdgesSubtract_op = [=] __device__(Int const& offset) {
      Int i = offset % data.num_nodes;
      Int j = offset / data.num_nodes;
      VF_t[offset] = MU_t[i + data.num_nodes * patt.dsts[j]] - FMax_t[offset];
      VR_t[offset] = MU_t[i + data.num_nodes * patt.srcs[j]] - RMax_t[offset];
    };
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      thrust::make_counting_iterator<Int>(data.num_nodes * patt.num_edges),
      RepeatColumnsByPatternEdgesSubtract_op
    );
    nvtxRangePop();

    nvtxRangePushA("step2");
    auto RepeatColumnsByDataEdges_op = [=] __device__(Int const& offset) {
      Int i = offset % data.num_edges;
      Int j = offset / data.num_edges;
      FE_t[offset] = VR_t[data.srcs[i] + data.num_nodes * j] - CE_t[offset];
      RE_t[offset] = VF_t[data.srcs[i] + data.num_nodes * j] - CE_t[offset];
    };
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      thrust::make_counting_iterator<Int>(data.num_edges * patt.num_edges),
      RepeatColumnsByDataEdges_op
    );
    nvtxRangePop();
    
    nvtxRangePushA("step3");
    nvtxRangePushA("step3_a");
    ac::host::RowMax2(patt.num_edges, data.num_nodes, VF_t, VFmax);               // x.max(axis=0)
    nvtxRangePop();
    nvtxRangePushA("step3_b");
    ac::host::RowSoftmax2_prealloc(patt.num_edges, data.num_edges, RE_t, RE_tmp); // x.softmax(axis=0)
    nvtxRangePop();
    nvtxRangePushA("step3_c");
    ac::host::EdgeMaxReduce2_t(
      data.num_edges, data.num_nodes, patt.num_edges,
      VFmax, RE_t, RMax_t,
      data.srcs
    );
    nvtxRangePop();
    nvtxRangePop();
    
    nvtxRangePushA("step4");
    ac::host::RowMax2(patt.num_edges, data.num_nodes, VR_t, VRmax);               // x.max(axis=0)
    ac::host::RowSoftmax2_prealloc(patt.num_edges, data.num_edges, FE_t, FE_tmp); // x.softmax(axis=0)
    ac::host::EdgeMaxReduce2_t(
      data.num_edges, data.num_nodes, patt.num_edges,
      VRmax, FE_t, FMax_t,
      data.dsts
    );
    nvtxRangePop();

    nvtxRangePushA("step5");
    ac::host::ComputeMU2_t(
      data.num_nodes, patt.num_edges,
      data.num_nodes, patt.num_nodes,
      CV_t,
      FMax_t,
      RMax_t,
      patt.srcs,
      patt.dsts,
      MU_t
    );
    
    ac::host::RowSoftmax2_prealloc(patt.num_nodes, data.num_nodes, MU_t, MU_tmp);
    nvtxRangePop();
    
    nvtxRangePop();
  }

  transpose(MU_t, MU, patt.num_nodes, data.num_nodes);
  
  long long elapsed = timer.stop();
  std::cerr << "elapsed=" << elapsed << std::endl;

  // --
  // Copy results to host and print

  Real *h_MU = (Real *) malloc(data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMemcpy(h_MU, MU, data.num_nodes * patt.num_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
  for (Int i = 0; i < data.num_nodes * patt.num_nodes; i ++) printf("%e\n", h_MU[i]);
}

#endif