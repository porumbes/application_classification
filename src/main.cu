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
  // Setup GPUs
  
  cudaStream_t master_stream;
  cudaEvent_t master_event;
  cudaStreamCreateWithFlags(&master_stream, cudaStreamNonBlocking);
  cudaEventCreate(&master_event);

  int _n_gpus = 1;
  cudaGetDeviceCount(&_n_gpus);
  Int n_gpus = (Int)(_n_gpus);
  
  for(Int i = 0; i < n_gpus; i++) {
      cudaSetDevice(i);
      for(Int j = 0; j < n_gpus; j++) {
          if(i == j) continue;
          cudaDeviceEnablePeerAccess(j, 0);
      }
  }
  cudaSetDevice(0);
  
  std::vector<gpu_info> infos;
  
  for(Int i = 0 ; i < n_gpus ; i++) {
      gpu_info info;
      cudaSetDevice(i);
      cudaStreamCreateWithFlags(&info.stream, cudaStreamNonBlocking);
      cudaEventCreate(&info.event);
      infos.push_back(info);
  }
  cudaSetDevice(0);

  Int* chunks    = (Int*)malloc(n_gpus * sizeof(Int));
  Int* starts    = (Int*)malloc(n_gpus * sizeof(Int));
  Int* ends      = (Int*)malloc(n_gpus * sizeof(Int));
  
  for(Int i = 0; i < n_gpus; i++)         chunks[i] = 0;
  for(Int i = 0; i < patt.num_edges; i++) chunks[i % n_gpus]++;
  
  starts[0] = 0;
  ends[0]   = chunks[0];
  for(Int i = 1; i < n_gpus; i++) {
      starts[i] = chunks[i] + starts[i - 1];
      ends[i]   = chunks[i] + ends[i - 1];
  }
  ends[n_gpus - 1] = patt.num_edges;
  
  // --
  // Allocate memory

  Real *MU, *VRmax, *VFmax;
  Real *CV_t, *CE_t, *MU_t, *RE_t, *FE_t, *VR_t, *VF_t, *RMax_t, *FMax_t;

  // Real *Cnull; // Ignoring for now

  cudaMalloc((void **)&MU,      data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMalloc((void **)&CV_t,    data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMalloc((void **)&MU_t,    data.num_nodes * patt.num_nodes * sizeof(Real));

  cudaMalloc((void **)&VRmax,                    patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VFmax,                    patt.num_edges * sizeof(Real));

  cudaMalloc((void **)&CE_t,    data.num_edges * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&RE_t,    data.num_edges * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&FE_t,    data.num_edges * patt.num_edges * sizeof(Real));

  cudaMalloc((void **)&VR_t,    data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&VF_t,    data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&RMax_t,  data.num_nodes * patt.num_edges * sizeof(Real));
  cudaMalloc((void **)&FMax_t,  data.num_nodes * patt.num_edges * sizeof(Real));

  Real* MU_tmp;
  Real* RE_tmp;
  Real* FE_tmp;
  cudaMalloc(&MU_tmp, patt.num_nodes * sizeof(Real));
  cudaMalloc(&RE_tmp, patt.num_edges * sizeof(Real));
  cudaMalloc(&FE_tmp, patt.num_edges * sizeof(Real));
  
  // --
  // Initialize algorithm

  cuda_timer_t timer;
  timer.start();
  nvtxRangePushA("start");
  
  nvtxRangePushA("prep");

  ac::cdist(data.num_nodes, patt.num_nodes, patt.node_feat_dim, data.node_feats, patt.node_feats, CV_t);
  ac::cdist(data.num_edges, patt.num_edges, patt.edge_feat_dim, data.edge_feats, patt.edge_feats, CE_t);
  
  thrust::transform(thrust::device, CV_t, CV_t + (patt.num_nodes * data.num_nodes), MU_t, [=] __device__ (Real const& val) {return - val;});
  thrust::transform(thrust::device, CE_t, CE_t + (patt.num_nodes * data.num_nodes), RE_t, [=] __device__ (Real const& val) {return - val;});
  thrust::transform(thrust::device, CE_t, CE_t + (patt.num_nodes * data.num_nodes), FE_t, [=] __device__ (Real const& val) {return - val;});

  ac::RowSoftmax2(patt.num_nodes, data.num_nodes, CV_t, master_stream);
  ac::RowSoftmax2(patt.num_nodes, data.num_nodes, MU_t, master_stream);
  ac::RowSoftmax2(patt.num_edges, data.num_edges, CE_t, master_stream);
  ac::RowSoftmax2(patt.num_edges, data.num_edges, RE_t, master_stream);
  ac::RowSoftmax2(patt.num_edges, data.num_edges, FE_t, master_stream);
  cudaDeviceSynchronize();
  
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
  
  nvtxRangePushA("scatter");
  for(Int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    cudaSetDevice(0);
  }

  Real** all_CE_t = (Real**)malloc(n_gpus * sizeof(Real*));
  shard_n(CE_t, all_CE_t, n_gpus, patt.num_edges, data.num_edges, starts, ends);

  Real** all_FE_t = (Real**)malloc(n_gpus * sizeof(Real*));
  Real** all_RE_t = (Real**)malloc(n_gpus * sizeof(Real*));
  shard_n(FE_t, all_FE_t, n_gpus, patt.num_edges, data.num_edges, starts, ends);
  shard_n(RE_t, all_RE_t, n_gpus, patt.num_edges, data.num_edges, starts, ends);
  
  Real** all_VF_t = (Real**)malloc(n_gpus * sizeof(Real*));
  Real** all_VR_t = (Real**)malloc(n_gpus * sizeof(Real*));
  shard_n(VF_t, all_VF_t, n_gpus, patt.num_edges, data.num_nodes, starts, ends);
  shard_n(VR_t, all_VR_t, n_gpus, patt.num_edges, data.num_nodes, starts, ends);

  Real** all_FMax_t = (Real**)malloc(n_gpus * sizeof(Real*));
  Real** all_RMax_t = (Real**)malloc(n_gpus * sizeof(Real*));
  shard_n(FMax_t, all_FMax_t, n_gpus, patt.num_edges, data.num_nodes, starts, ends);
  shard_n(RMax_t, all_RMax_t, n_gpus, patt.num_edges, data.num_nodes, starts, ends);

  Real** all_VFmax = (Real**)malloc(n_gpus * sizeof(Real*));
  Real** all_VRmax = (Real**)malloc(n_gpus * sizeof(Real*));
  shard_n(VFmax, all_VFmax, n_gpus, patt.num_edges, 1, starts, ends);
  shard_n(VRmax, all_VRmax, n_gpus, patt.num_edges, 1, starts, ends);
  
  Real** all_FE_tmp = (Real**)malloc(n_gpus * sizeof(Real*));
  Real** all_RE_tmp = (Real**)malloc(n_gpus * sizeof(Real*));
  shard_n(FE_tmp, all_FE_tmp, n_gpus, patt.num_edges, 1, starts, ends);
  shard_n(RE_tmp, all_RE_tmp, n_gpus, patt.num_edges, 1, starts, ends);
    
  Int** all_data_srcs = (Int**)malloc(n_gpus * sizeof(Int**));
  Int** all_data_dsts = (Int**)malloc(n_gpus * sizeof(Int**));
  copy_n(data.srcs, all_data_srcs, n_gpus, data.num_edges, 1);
  copy_n(data.dsts, all_data_dsts, n_gpus, data.num_edges, 1);

  Int** all_patt_srcs = (Int**)malloc(n_gpus * sizeof(Int**));
  Int** all_patt_dsts = (Int**)malloc(n_gpus * sizeof(Int**));
  copy_n(patt.srcs, all_patt_srcs, n_gpus, patt.num_edges, 1);
  copy_n(patt.dsts, all_patt_dsts, n_gpus, patt.num_edges, 1);

  for(Int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    cudaSetDevice(0);
  }

  nvtxRangePop();
  
  // --
  // Run

  for (Int i = 0; i < patt.num_nodes; i++) {
    nvtxRangePushA("loop");
    
    nvtxRangePushA("update_VX");
    
    for(Int i = 0; i < n_gpus; i++) {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
      cudaSetDevice(0);
    }
    
    // random row access -- BAD
    #pragma omp parallel for num_threads(n_gpus)
    for(Int gid = 0; gid < n_gpus; gid++) {
      cudaSetDevice(gid);
      
      Int start = starts[gid];
      Int end   = ends[gid];
      Int size  = end - start;
      
      Real* l_VF_t     = all_VF_t[gid];
      Real* l_VR_t     = all_VR_t[gid];
      Real* l_FMax_t   = all_FMax_t[gid];
      Real* l_RMax_t   = all_RMax_t[gid];
      Int* l_patt_srcs = all_patt_srcs[gid];
      Int* l_patt_dsts = all_patt_dsts[gid];
      
      auto update_VX = [=] __device__(Int const& offset) {
        Int r          = offset / data.num_nodes;
        Int c          = offset % data.num_nodes;
        l_VF_t[offset] = MU_t[data.num_nodes * l_patt_dsts[start + r] + c] - l_FMax_t[offset];
        l_VR_t[offset] = MU_t[data.num_nodes * l_patt_srcs[start + r] + c] - l_RMax_t[offset];
      };
      thrust::for_each_n(
        thrust::cuda::par.on(infos[gid].stream),
        thrust::make_counting_iterator<Int>(0),
        size * data.num_nodes,
        update_VX
      );
      cudaEventRecord(infos[gid].event, infos[gid].stream);
    }

    nvtxRangePop();
        
    // shard_n_prealloc(VF_t, all_VF_t, n_gpus, patt.num_edges, data.num_nodes, starts, ends);
    // shard_n_prealloc(VR_t, all_VR_t, n_gpus, patt.num_edges, data.num_nodes, starts, ends);

    
    nvtxRangePushA("updateXMax_t");
    ac::updateXMax_t(
      patt.num_nodes, patt.num_edges, data.num_nodes, data.num_edges,
      all_CE_t,
      all_VF_t,
      all_VFmax,
      all_RE_t,
      all_RE_tmp,
      all_RMax_t,
      all_data_srcs,
      all_data_srcs,
      n_gpus,
      starts,
      ends,
      RMax_t,
      infos
    );
    
    ac::updateXMax_t(
      patt.num_nodes, patt.num_edges, data.num_nodes, data.num_edges,
      all_CE_t,
      all_VR_t,
      all_VRmax,
      all_FE_t,
      all_FE_tmp,
      all_FMax_t,
      all_data_srcs,
      all_data_dsts,
      n_gpus,
      starts,
      ends,
      FMax_t,
      infos
    );
    
    cudaDeviceSynchronize();
    for(Int gid = 0; gid < n_gpus; gid++)
        cudaStreamWaitEvent(master_stream, infos[gid].event, 0);
    cudaStreamSynchronize(master_stream);

    nvtxRangePop();
    
    nvtxRangePushA("ComputeMU2_t");
    // random row-write -- BAD
    ac::ComputeMU2_t(
      data.num_nodes, patt.num_edges, data.num_nodes, patt.num_nodes,
      CV_t,
      FMax_t,
      RMax_t,
      patt.srcs,
      patt.dsts,
      MU_t,
      master_stream
    );

    cudaEventRecord(master_event, master_stream);
    cudaStreamWaitEvent(master_stream, master_event, 0);

    // simple row-wise -- OK
    ac::RowSoftmax2_prealloc(patt.num_nodes, data.num_nodes, MU_t, MU_tmp, master_stream);
    cudaEventRecord(master_event, master_stream);
    cudaStreamWaitEvent(master_stream, master_event, 0);

    nvtxRangePop();
    
    nvtxRangePop();
  }

  nvtxRangePop();
  long long elapsed = timer.stop();
  std::cerr << "elapsed=" << elapsed << std::endl;

  // --
  // Copy results to host and print

  ac::transpose(MU_t, MU, patt.num_nodes, data.num_nodes);
  Real *h_MU = (Real *) malloc(data.num_nodes * patt.num_nodes * sizeof(Real));
  cudaMemcpy(h_MU, MU, data.num_nodes * patt.num_nodes * sizeof(Real), cudaMemcpyDeviceToHost);
  for (Int i = 0; i < data.num_nodes * patt.num_nodes; i ++) printf("%e\n", h_MU[i]);
}