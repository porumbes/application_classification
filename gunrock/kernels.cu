#include <iostream>
#include <assert.h>
#include "main.h"
#include <cub/cub.cuh>

#define THREAD 1024

namespace ac {

// --
// Helpers

void sort_edges(IntT* srcs, IntT* dsts, IntT* srcs_r, IntT* dsts_r, IntT* map_r, IntT num_edges) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Copy edgelist to be sorted by dst
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      dsts, dsts_r, srcs, srcs_r, num_edges);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      dsts, dsts_r, srcs, srcs_r, num_edges);

  // Map between two edgelist orders
  IntT* h_map = (IntT*)malloc(num_edges * sizeof(IntT));
  for(IntT i = 0; i < num_edges; i++) {
    h_map[i] = (IntT)i;
  }

  IntT* map;
  cudaMalloc((void**)&map, num_edges * sizeof(IntT));
  cudaMemcpy(map, h_map, num_edges * sizeof(IntT), cudaMemcpyHostToDevice);

  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      dsts, dsts_r, map, map_r, num_edges);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      dsts, dsts_r, map, map_r, num_edges);

  free(h_map);
  cudaFree(map);
}

// --
// Kernels

__device__ FloatT d_norm_2(FloatT * vec1, FloatT * vec2, IntT n) {
  // Euclidean distance between vectors
  FloatT sum = 0.0;
  for (int i = 0; i < n; i ++) {
    sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  }
  return sqrt(sum);
}

__global__ void __fillLow(FloatT *d_x, IntT n) {
  // Fill array w/ min value
  IntT offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < n) {
    d_x[offset] = -DBL_MAX;
  }
}

__global__ void __transpose(FloatT *d_xt, FloatT *d_x, IntT num_rows, IntT num_cols) {
  // Transpose matrix
  IntT offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    IntT row = offset / num_cols;
    IntT col = offset % num_cols;
    d_xt[col * num_rows + row] = d_x[offset];
  }
}

__global__ void __transposeWithKey(FloatT *d_xt, FloatT *d_x, IntT *d_idx, IntT* num_entries, IntT num_rows, IntT num_cols) {
  // Transpose matrix, when not all values are defined
  IntT offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_entries[0]) {
    IntT offset_t = d_idx[offset];
    IntT row  = offset_t / num_cols;
    IntT col  = offset_t % num_cols;
    d_xt[col * num_rows + row] = d_x[offset];
  }
}

__global__ void __maxMatrixRowVector(FloatT * d_matrix, FloatT * d_vec, IntT num_cols, IntT n) {
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n) {
    d_matrix[i] = max(d_matrix[i], d_vec[i % num_cols]);
  }
}

__global__ void __rowSubExp(FloatT* d_x, IntT num_rows, IntT num_cols, FloatT* c) {
  IntT offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    IntT row = offset / num_cols;
    d_x[offset] = exp(d_x[offset] - c[row]);
  }
}

__global__ void __rowSubLog(FloatT* d_x, IntT num_rows, IntT num_cols, FloatT* c) {
  IntT offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    IntT row = offset / num_cols;
    d_x[offset] = log(d_x[offset]) - log(c[row]);
  }
}

__global__ void __scalarMultiply(FloatT * d_out, FloatT * d_in, FloatT alpha, IntT n) {
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n)
    d_out[i] = alpha * d_in[i];
}

__global__ void __tileVectorWithOffset(IntT * d_out, IntT * d_in, IntT num_in, IntT num_uin, IntT num_out) {
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < num_out)
    d_out[i] = num_uin * (i / num_in) + d_in[i % num_in];
}

__global__ void __vectorScatterAdd(FloatT * d_out, IntT * d_key_in, FloatT * d_value_in, IntT * n) {
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n[0])
    d_out[d_key_in[i]] += d_value_in[i];
}

__global__ void __reorderEdges(FloatT* d_out, FloatT* d_in, IntT* d_map_r, IntT num_in, IntT num_out) {
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < num_out) {
    IntT col = i % num_in;
    IntT row = i / num_in;
    d_out[i] = d_in[row * num_in + d_map_r[col]];
  }
}



void rowmax(FloatT * d_out, FloatT * d_in, IntT num_rows, IntT num_cols) {
  // Max over rows of matrix

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Compute offsets of matrix
  IntT *h_offsets = (IntT*)malloc((num_rows + 1) * sizeof(IntT));
  IntT *d_offsets;
  for(IntT i = 0; i < num_rows + 1; i++) {
    h_offsets[i] = i * num_cols;
  }
  cudaMalloc((void**)&d_offsets, (num_rows + 1) * sizeof(IntT));
  cudaMemcpy(d_offsets, h_offsets, (num_rows + 1) * sizeof(IntT), cudaMemcpyHostToDevice);

  // Max over rows
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_out,
    num_rows,
    d_offsets,
    d_offsets + 1
  );
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_out,
    num_rows,
    d_offsets,
    d_offsets + 1
  );

  cudaFree(d_offsets);
  cudaFree(d_temp_storage);
}

void rowsum(FloatT * d_out, FloatT * d_in, IntT num_rows, IntT num_cols) {
  // Sum over rows of matrix

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Compute offsets of matrix
  IntT *h_offsets = (IntT*)malloc((num_rows + 1) * sizeof(IntT));
  IntT *d_offsets;
  for(IntT i = 0; i < num_rows + 1; i++) {
    h_offsets[i] = i * num_cols;
  }
  cudaMalloc((void**)&d_offsets, (num_rows + 1) * sizeof(IntT));
  cudaMemcpy(d_offsets, h_offsets, (num_rows + 1) * sizeof(IntT), cudaMemcpyHostToDevice);

  // Sum over rows
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_out,
    num_rows,
    d_offsets,
    d_offsets + 1
  );
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_out,
    num_rows,
    d_offsets,
    d_offsets + 1
  );

  cudaFree(d_offsets);
}

// ================ Application Classification Specific ===============

__global__ void __NodePairwiseNorm(
  int DV,
  int PV,
  FloatT* CV,
  FloatT* MU,
  FloatT* data_node_feats,
  FloatT* patt_node_feats,
  int node_feat_dim
) {
  IntT k = threadIdx.x + blockIdx.x * blockDim.x;
  if(k < DV * PV) {
      IntT i = k / PV;
      IntT j = k % PV;
      FloatT dist = d_norm_2(
        patt_node_feats + j * node_feat_dim,
        data_node_feats + i * node_feat_dim,
        node_feat_dim
      );

      CV[k] = dist;
      MU[k] = -dist;
  }
}

void Init_CV_MU(Graph* data, Graph* patt, FloatT* d_CV, FloatT* d_MU) {
  // Compute pairwise distance between `data` and `patt` features

  IntT DV = data->num_nodes;
  IntT PV = patt->num_nodes;

  int block = 1 + (DV * PV) / THREAD;
  assert(block * THREAD > DV * PV);
  __NodePairwiseNorm<<<block, THREAD>>>(
    DV,
    PV,
    d_CV,
    d_MU,
    data->node_feats,
    patt->node_feats,
    data->node_feat_dim
  );
}


void ColumnSoftmax(const IntT num_rows, const IntT num_cols, FloatT *d_x) {
  // Compute softmax over columns

  // --------------------------
  // Prep

  IntT block  = 1 + (num_rows * num_cols) / THREAD;
  assert(THREAD * block > num_rows * num_cols);

  FloatT *d_storage;
  cudaMalloc((void**)&d_storage, num_cols * sizeof(FloatT));

  // ----------------------------
  // Compute column max

  FloatT *d_xt;
  cudaMalloc((void**)&d_xt, num_rows * num_cols * sizeof(FloatT));
  __transpose<<<block, THREAD>>>(d_xt, d_x, num_rows, num_cols);
  rowmax(d_storage, d_xt, num_cols, num_rows);

  // --------------------------------
  // Subtract max from columns

  __rowSubExp<<<block, THREAD>>>(d_xt, num_cols, num_rows, d_storage);

  // --------------------------------
  // Sum columns

  cudaMemset(d_storage, 0, num_cols * sizeof(FloatT));
  rowsum(d_storage, d_xt, num_cols, num_rows);

  // ---------------------------------
  // Subtract log-sum from columns

  __rowSubLog<<<block, THREAD>>>(d_xt, num_cols, num_rows, d_storage);

  // ---------------------------------
  // Transpose back to original shape

  __transpose<<<block, THREAD>>>(d_x, d_xt, num_cols, num_rows);

  // ---------------------------------
  // Free memory

  cudaFree(d_xt);
  cudaFree(d_storage);
}


__global__ void __RepeatColumnsByEdges(
  const IntT DV,
  const IntT PE,
  const IntT PV,
  FloatT * MU,
  FloatT * VR,
  FloatT * VF,
  IntT * patt_srcs,
  IntT * patt_dsts
)
{
  IntT k = threadIdx.x + blockDim.x * blockIdx.x;

  if(k < DV * PE) {
    IntT i = k / PE;
    IntT j = k % PE;
    VR[k] = MU[i * PV + patt_srcs[j]];
    VF[k] = MU[i * PV + patt_dsts[j]];
  }
}

void Init_VR_VF(Graph * patt, IntT DV, FloatT * MU, FloatT * VR, FloatT * VF) {
  // Replicate columns of MU by pattern edges

  const IntT PV = patt->num_nodes;
  const IntT PE = patt->num_edges;

  IntT block_dv_pe = 1 + (DV * PE) / THREAD;
  assert(THREAD * block_dv_pe > DV * PE);
  __RepeatColumnsByEdges<<<block_dv_pe, THREAD>>>(
    DV,
    PE,
    PV,
    MU,
    VR,
    VF,
    patt->srcs,
    patt->dsts
  );
}

__global__ void __EdgePairwiseNorm(
  IntT DE,
  IntT PE,
  FloatT * CE,
  FloatT * RE,
  FloatT * FE,
  FloatT * data_edge_feats,
  FloatT * patt_edge_feats,
  IntT edge_feat_dim
)
{
  IntT k = threadIdx.x + blockDim.x * blockIdx.x;

  if(k < DE * PE) {
    IntT i = k / PE;
    IntT j = k % PE;
    FloatT dist = d_norm_2(
      patt_edge_feats + j * edge_feat_dim,
      data_edge_feats + i * edge_feat_dim,
      edge_feat_dim
    );

    CE[k] = dist;
    RE[k] = - dist;
    FE[k] = - dist;
  }
}

void Init_CE_RE_FE(Graph * data, Graph * patt, FloatT * d_CE, FloatT * d_RE, FloatT * d_FE) {
  // Pairwise distance between edge features

  IntT DE = data->num_edges;
  IntT PE = patt->num_edges;

  IntT block_de_pe  = 1 + (DE * PE) / THREAD;
  assert(THREAD * block_de_pe > DE * PE);
  __EdgePairwiseNorm<<<block_de_pe, THREAD>>>(
    DE,
    PE,
    d_CE,
    d_RE,
    d_FE,
    data->edge_feats,
    patt->edge_feats,
    data->edge_feat_dim
  );
}


void VFmax_VRmax(IntT num_rows, IntT num_cols, FloatT * d_VF, FloatT * d_VR, FloatT * d_VFmax, FloatT * d_VRmax) {
  // Compute maximum over columns of d_V{F,R}

  IntT block = 1 + (num_rows * num_cols) / THREAD;
  assert(THREAD * block > num_rows * num_cols);

  // VFMax
  FloatT *d_VFt;
  cudaMalloc((void**)&d_VFt, num_rows * num_cols * sizeof(FloatT));
  __transpose<<<block, THREAD>>>(d_VFt, d_VF, num_rows, num_cols);
  rowmax(d_VFmax, d_VFt, num_cols, num_rows);
  cudaFree(d_VFt);

  // VRmax
  FloatT *d_VRt;
  cudaMalloc((void**)&d_VRt, num_rows * num_cols * sizeof(FloatT));
  __transpose<<<block, THREAD>>>(d_VRt, d_VR, num_rows, num_cols);
  rowmax(d_VRmax, d_VRt, num_cols, num_rows);
  cudaFree(d_VRt);
}


__global__ void __SubRepeatColumnsByEdges(
  IntT DV,
  IntT PE,
  IntT PV,
  FloatT * MU,
  FloatT * VR,
  FloatT * VF,
  FloatT * FMax,
  FloatT * RMax,
  IntT * srcs,
  IntT * dsts
)
{

  IntT k = threadIdx.x + blockDim.x * blockIdx.x;
  if(k < DV * PE) {
    IntT i = k / PE;
    IntT j = k % PE;
    VF[k] = MU[i * PV + dsts[j]] - FMax[k];
    VR[k] = MU[i * PV + srcs[j]] - RMax[k];
  }
}

void VF_VR(Graph * patt, IntT DV, FloatT * d_MU, FloatT * d_FMax, FloatT * d_RMax, FloatT * d_VF, FloatT * d_VR) {

  IntT PV = patt->num_nodes;
  IntT PE = patt->num_edges;

  IntT block_dv_pe = 1 + (DV * PE) / THREAD;
  assert(THREAD * block_dv_pe > DV * PE);
  __SubRepeatColumnsByEdges<<<block_dv_pe, THREAD>>>(
    DV,
    PE,
    PV,
    d_MU,
    d_VR,
    d_VF,
    d_FMax,
    d_RMax,
    patt->srcs,
    patt->dsts
  );
}

void UpdateMU(Graph * patt, IntT DV, FloatT * d_CV, FloatT * d_FMax, FloatT * d_RMax, FloatT * d_MU) {
  // Replace columns of MU w/ sum over FMax/RMax of adjacent edges + subtract CV

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  IntT PV = patt->num_nodes;
  IntT PE = patt->num_edges;

  // --------------------------------------------
  // MU = -CV

  IntT block_dv_pv = 1 + (DV * PV) / THREAD;
  assert(THREAD * block_dv_pv > DV * PV);
  __scalarMultiply<<<block_dv_pv, THREAD>>>(d_MU, d_CV, (FloatT)-1.0, DV * PV);

  // --------------------------------------------
  // Tile srcs

  IntT *d_tiled_nodes;
  cudaMalloc((void**)&d_tiled_nodes, DV * PE * sizeof(IntT));

  IntT block_dv_pe = 1 + (DV * PE) / THREAD;
  assert(THREAD * block_dv_pe > DV * PE);
  __tileVectorWithOffset<<<block_dv_pe, THREAD>>>(d_tiled_nodes, patt->srcs, PE, PV, DV * PE);

  // --------------------------------------------
  // Sum over rows of matrix

  IntT   *d_keys_out;
  FloatT *d_values_out;
  IntT   *d_num_runs_out;

  cudaMalloc((void**)&d_keys_out,       DV * PV * sizeof(IntT));
  cudaMalloc((void**)&d_values_out,     DV * PV * sizeof(FloatT));
  cudaMalloc((void**)&d_num_runs_out,   1 * sizeof(IntT));

  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_tiled_nodes, d_keys_out, d_RMax, d_values_out, d_num_runs_out, cub::Sum(), DV * PE);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_tiled_nodes, d_keys_out, d_RMax, d_values_out, d_num_runs_out, cub::Sum(), DV * PE);

  // --------------------------------------------
  // (Scatter) Add to MU

  __vectorScatterAdd<<<block_dv_pv, THREAD>>>(d_MU, d_keys_out, d_values_out, d_num_runs_out);

  // --------------------------------------------
  // Tile dsts

  __tileVectorWithOffset<<<block_dv_pe, THREAD>>>(d_tiled_nodes, patt->dsts_r, PE, PV, DV * PE);

  // --------------------------------------
  // Reorder data

  FloatT *d_FMax_r;
  cudaMalloc((void**)&d_FMax_r, DV * PE * sizeof(FloatT));
  __reorderEdges<<<block_dv_pe, THREAD>>>(d_FMax_r, d_FMax, patt->map_r, PE, DV * PE);

  // --------------------------------------
  // Sum reduce

  d_temp_storage = NULL; temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_tiled_nodes, d_keys_out, d_FMax_r, d_values_out, d_num_runs_out, cub::Sum(), DV * PE);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_tiled_nodes, d_keys_out, d_FMax_r, d_values_out, d_num_runs_out, cub::Sum(), DV * PE);

  // (Scatter) add to d_MU
  __vectorScatterAdd<<<block_dv_pv, THREAD>>>(d_MU, d_keys_out, d_values_out, d_num_runs_out);

  // --------------------------------------
  // Free memory

  cudaFree(d_temp_storage);
  cudaFree(d_tiled_nodes);
  cudaFree(d_keys_out);
  cudaFree(d_values_out);
  cudaFree(d_num_runs_out);
  cudaFree(d_FMax_r);
}

__global__ void __FE_RE(
  IntT DE,
  IntT PE,
  FloatT * CE,
  FloatT * VR,
  FloatT * VF,
  FloatT * FE,
  FloatT * RE,
  IntT * srcs
)
{
  IntT k = threadIdx.x + blockDim.x * blockIdx.x;
  if(k < DE * PE) {
    IntT ij  = k / PE;
    IntT km  = k % PE;
    IntT src = srcs[ij];

    FloatT CE_k = CE[k];
    FE[k] = - CE_k + VR[src * PE + km];
    RE[k] = - CE_k + VF[src * PE + km];
  }
}

void FE_RE(Graph * data, IntT PE, FloatT * d_CE, FloatT * d_VF, FloatT * d_VR, FloatT * d_FE, FloatT * d_RE) {

  IntT DE = data->num_edges;

  IntT block = 1 + (DE * PE) / THREAD;
  assert(THREAD * block > DE * PE);
  __FE_RE<<<block, THREAD>>>(
    DE,
    PE,
    d_CE,
    d_VR,
    d_VF,
    d_FE,
    d_RE,
    data->srcs
  );
}

void RMax(Graph * data, IntT PE, FloatT * d_Cnull, FloatT * d_VFmax, FloatT * d_RE, FloatT * d_RMax) {
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  IntT DV = data->num_nodes;
  IntT DE = data->num_edges;

  // --------------------------------------
  // Transpose

  FloatT *d_REt; // PE x DE
  cudaMalloc((void**)&d_REt, DE * PE * sizeof(FloatT));

  IntT block_de_pe = 1 + (DE * PE) / THREAD;
  assert(THREAD * block_de_pe > DE * PE);
  __transpose<<<block_de_pe, THREAD>>>(d_REt, d_RE, DE, PE);

  // --------------------------------------
  // Tile srcs

  IntT *d_tiled_srcs;
  cudaMalloc((void**)&d_tiled_srcs, DE * PE * sizeof(IntT));
  cudaMemset(d_tiled_srcs, 0, DE * PE * sizeof(IntT));
  __tileVectorWithOffset<<<block_de_pe, THREAD>>>(d_tiled_srcs, data->srcs, DE, DV, DE * PE);

  // --------------------------------------
  // Max reduce rows of transposed matrix

  IntT *d_keys_out;
  FloatT *d_values_out;
  IntT *d_num_runs_out;
  cudaMalloc((void**)&d_keys_out,       DV * PE * sizeof(IntT));
  cudaMalloc((void**)&d_values_out,     DV * PE * sizeof(FloatT));
  cudaMalloc((void**)&d_num_runs_out,   1 * sizeof(IntT));
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_tiled_srcs, d_keys_out, d_REt, d_values_out, d_num_runs_out, cub::Max(), DE * PE);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_tiled_srcs, d_keys_out, d_REt, d_values_out, d_num_runs_out, cub::Max(), DE * PE);

  // --------------------------------------
  // Transpose result back to d_RMax

  IntT block_dv_pe = 1 + (DV * PE) / THREAD;
  assert(THREAD * block_dv_pe > DV * PE);
  __fillLow<<<block_dv_pe, THREAD>>>(d_RMax, PE * DV);
  __transposeWithKey<<<block_dv_pe, THREAD>>>(d_RMax, d_values_out, d_keys_out, d_num_runs_out, PE, DV);

  // --------------------------------------
  // Elementwise max w/ V*max
  //   !! CNull hardcoded to 0, so ignore

  __maxMatrixRowVector<<<block_dv_pe, THREAD>>>(d_RMax, d_VFmax, PE, DV * PE);

  // --------------------------------------
  // Free memory

  cudaFree(d_REt);
  cudaFree(d_tiled_srcs);
  cudaFree(d_keys_out);
  cudaFree(d_values_out);
  cudaFree(d_num_runs_out);
  cudaFree(d_temp_storage);
}


void FMax(Graph* data, IntT PE, FloatT* d_Cnull, FloatT* d_VRmax, FloatT* d_FE, FloatT* d_FMax) {
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  IntT DV = data->num_nodes;
  IntT DE = data->num_edges;

  // --------------------------------------
  // Transpose

  FloatT *d_FEt;
  cudaMalloc((void**)&d_FEt, DE * PE * sizeof(FloatT));
  IntT block_de_pe = 1 + (DE * PE) / THREAD;
  assert(THREAD * block_de_pe > DE * PE);
  __transpose<<<block_de_pe, THREAD>>>(d_FEt, d_FE, DE, PE);

  // --------------------------------------
  // Tile dsts (w/ offset)

  IntT *d_tiled_dsts;
  cudaMalloc((void**)&d_tiled_dsts, DE * PE * sizeof(IntT));
  cudaMemset(d_tiled_dsts, 0, DE * PE * sizeof(IntT));
  __tileVectorWithOffset<<<block_de_pe, THREAD>>>(d_tiled_dsts, data->dsts_r, DE, DV, DE * PE);

  // --------------------------------------
  // Reorder data

  FloatT *d_FEt_r;
  cudaMalloc((void**)&d_FEt_r, DE * PE * sizeof(FloatT));
  __reorderEdges<<<block_de_pe, THREAD>>>(d_FEt_r, d_FEt, data->map_r, DE, DE * PE);

  // --------------------------------------
  // Max reduce rows of transposed matrix

  IntT   *d_keys_out;
  FloatT *d_values_out;
  IntT   *d_num_runs_out;
  cudaMalloc((void**)&d_keys_out,       DV * PE * sizeof(IntT));
  cudaMalloc((void**)&d_values_out,     DV * PE * sizeof(FloatT));
  cudaMalloc((void**)&d_num_runs_out,   1 * sizeof(IntT));

  d_temp_storage = NULL; temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_tiled_dsts, d_keys_out, d_FEt_r, d_values_out, d_num_runs_out, cub::Max(), DE * PE);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_tiled_dsts, d_keys_out, d_FEt_r, d_values_out, d_num_runs_out, cub::Max(), DE * PE);

  // --------------------------------------
  // Transpose result back to d_FMax

  IntT block_dv_pe = 1 + (DV * PE) / THREAD;
  assert(THREAD * block_dv_pe > DV * PE);
  __fillLow<<<block_dv_pe, THREAD>>>(d_FMax, PE * DV);
  __transposeWithKey<<<block_dv_pe, THREAD>>>(d_FMax, d_values_out, d_keys_out, d_num_runs_out, PE, DV);

  // --------------------------------------
  // Elementwise max w/ V*max
  //   !! CNull hardcoded to 0, so ignore

  __maxMatrixRowVector<<<block_dv_pe, THREAD>>>(d_FMax, d_VRmax, PE, DV * PE);

  // -------------------------------------
  // Free memory

  cudaFree(d_FEt);
  cudaFree(d_FEt_r);
  cudaFree(d_tiled_dsts);
  cudaFree(d_temp_storage);
  cudaFree(d_keys_out);
  cudaFree(d_values_out);
  cudaFree(d_num_runs_out);
}

}