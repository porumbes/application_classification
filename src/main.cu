#include <iostream>
#include "nvToolsExt.h"
#include "thrust/device_vector.h"

#include "ac.hxx"
#include "helpers.hxx"

typedef struct Graph {
  Int   num_nodes;
  Int   node_feat_dim;
  Real* node_feats;

  Int   num_edges;
  Int   edge_feat_dim;
  Real* edge_feats;

  Int* srcs;
  Int* dsts;
} Graph;

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

  cudaMalloc((void **)&MU,      data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMalloc((void **)&CV_t,    data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMalloc((void **)&MU_t,    data.num_nodes * patt.num_nodes * sizeof(Real));

  cudaMalloc((void **)&VRmax,                  patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VFmax,                  patt.num_edges * sizeof(Real));

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

  ac::cdist(patt.num_nodes, data.num_nodes, patt.node_feat_dim, patt.node_feats, data.node_feats, CV_t);
  ac::cdist(patt.num_edges, data.num_edges, patt.edge_feat_dim, patt.edge_feats, data.edge_feats, CE_t);
  
  thrust::transform(thrust::device, CV_t, CV_t + (patt.num_nodes * data.num_nodes), MU_t, [=] __device__ (Real const& val) {return - val;});
  thrust::transform(thrust::device, CE_t, CE_t + (patt.num_nodes * data.num_nodes), RE_t, [=] __device__ (Real const& val) {return - val;});
  thrust::transform(thrust::device, CE_t, CE_t + (patt.num_nodes * data.num_nodes), FE_t, [=] __device__ (Real const& val) {return - val;});

  ac::RowSoftmax2(patt.num_nodes, data.num_nodes, CV_t);
  ac::RowSoftmax2(patt.num_nodes, data.num_nodes, MU_t);
  ac::RowSoftmax2(patt.num_edges, data.num_edges, CE_t);
  ac::RowSoftmax2(patt.num_edges, data.num_edges, RE_t);
  ac::RowSoftmax2(patt.num_edges, data.num_edges, FE_t);

  auto init_VX = [=] __device__(Int const& offset) {
      Int i        = offset % data.num_nodes;
      Int j        = offset / data.num_nodes;
      VR_t[offset] = MU_t[data.num_nodes * patt.srcs[j] + i];
      VF_t[offset] = MU_t[data.num_nodes * patt.dsts[j] + i];
  };
  thrust::for_each_n(
    thrust::device,
    thrust::make_counting_iterator<Int>(0),
    data.num_nodes * patt.num_edges,
    init_VX
  );

  ac::RowMax2(patt.num_edges, data.num_nodes, VF_t, VFmax);
  ac::RowMax2(patt.num_edges, data.num_nodes, VR_t, VRmax);

  ac::EdgeMaxReduce2_t(
    data.num_edges, data.num_nodes, patt.num_edges,
    VFmax, RE_t, RMax_t, data.srcs
  );
  
  ac::EdgeMaxReduce2_t(
    data.num_edges, data.num_nodes, patt.num_edges,
    VRmax, FE_t, FMax_t, data.dsts
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
    // random row access -- BAD
    auto update_VX = [=] __device__(Int const& offset) {
      Int r        = offset / data.num_nodes;
      Int c        = offset % data.num_nodes;
      VF_t[offset] = MU_t[data.num_nodes * patt.dsts[r] + c] - FMax_t[offset];
      VR_t[offset] = MU_t[data.num_nodes * patt.srcs[r] + c] - RMax_t[offset];
    };
    thrust::for_each_n(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      data.num_nodes * patt.num_edges,
      update_VX
    );
    nvtxRangePop();

    nvtxRangePushA("step2");
    // random column read -- OK
    auto update_XE = [=] __device__(Int const& offset) {
      Int r        = offset / data.num_edges;
      Int c        = offset % data.num_edges;
      FE_t[offset] = VR_t[data.num_nodes * r + data.srcs[c]] - CE_t[offset];
      RE_t[offset] = VF_t[data.num_nodes * r + data.srcs[c]] - CE_t[offset];
    };
    thrust::for_each_n(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      data.num_edges * patt.num_edges,
      update_XE
    );
    nvtxRangePop();
    
    nvtxRangePushA("step3_a"); // independent of step f
    // simple row-wise -- OK
    ac::RowMax2(patt.num_edges, data.num_nodes, VF_t, VFmax);
    nvtxRangePop();
    
    nvtxRangePushA("step3_b");
    // simple row-wise -- OK
    ac::RowSoftmax2_prealloc(patt.num_edges, data.num_edges, RE_t, RE_tmp);
    nvtxRangePop();
    
    nvtxRangePushA("step3_c");
    // random column read -- OK
    ac::EdgeMaxReduce2_t(
      data.num_edges, data.num_nodes, patt.num_edges,
      VFmax, RE_t, RMax_t,
      data.srcs
    );
    nvtxRangePop();
    
    nvtxRangePushA("step4"); // independent of step 3
    // simple row-wise -- OK
    ac::RowMax2(patt.num_edges, data.num_nodes, VR_t, VRmax);
    // simple row-wise -- OK
    ac::RowSoftmax2_prealloc(patt.num_edges, data.num_edges, FE_t, FE_tmp);
    // random column read -- OK
    ac::EdgeMaxReduce2_t(
      data.num_edges, data.num_nodes, patt.num_edges,
      VRmax, FE_t, FMax_t,
      data.dsts
    );
    nvtxRangePop();

    nvtxRangePushA("step5");
    // random row-write -- BAD
    ac::ComputeMU2_t(
      data.num_nodes, patt.num_edges,
      data.num_nodes, patt.num_nodes,
      CV_t,
      FMax_t,
      RMax_t,
      patt.srcs,
      patt.dsts,
      MU_t
    );
    
    // simple row-wise -- OK
    ac::RowSoftmax2_prealloc(patt.num_nodes, data.num_nodes, MU_t, MU_tmp);
    nvtxRangePop();
    
    nvtxRangePop();
  }

  ac::transpose(MU_t, MU, patt.num_nodes, data.num_nodes);
  
  long long elapsed = timer.stop();
  std::cerr << "elapsed=" << elapsed << std::endl;

  // --
  // Copy results to host and print

  Real *h_MU = (Real *) malloc(data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMemcpy(h_MU, MU, data.num_nodes * patt.num_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
  for (Int i = 0; i < data.num_nodes * patt.num_nodes; i ++) printf("%e\n", h_MU[i]);
}