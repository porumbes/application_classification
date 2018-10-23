#include <iostream>
#include <assert.h>
#include "main.h"
#include <cub/cub.cuh>

#define THREAD 1024

namespace ac {

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

__global__ void __maxMatrixColumnVector(FloatT * d_matrix, FloatT * d_vec, IntT num_rows, IntT n) {
  // Broadast row vector over matrix and take max
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n) {
    d_matrix[i] = max(d_matrix[i], d_vec[i / num_rows]);
  }
}

__global__ void __rowSubExp(FloatT* d_x, IntT num_rows, IntT num_cols, FloatT* c) {
  // Fused row-wise subtract and exp
  IntT offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    IntT row = offset / num_cols;
    d_x[offset] = exp(d_x[offset] - c[row]);
  }
}

__global__ void __rowSubLog(FloatT* d_x, IntT num_rows, IntT num_cols, FloatT* c) {
  // Fused row-wise subtract and log
  IntT offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    IntT row = offset / num_cols;
    d_x[offset] = log(d_x[offset]) - log(c[row]);
  }
}

__global__ void __scalarMultiply(FloatT * d_out, FloatT * d_in, FloatT alpha, IntT n) {
  // Multiply enties of array by scalar
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n)
    d_out[i] = alpha * d_in[i];
}

__global__ void __tileVectorWithOffset(IntT * d_out, IntT * d_in, IntT num_in, IntT num_uin, IntT num_out) {
  // Repeat a vector, adding offset each time to avoid duplicate
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < num_out)
    d_out[i] = num_uin * (i / num_in) + d_in[i % num_in];
}

__global__ void __vectorScatterAdd(FloatT * d_out, IntT * d_key_in, FloatT * d_value_in, IntT * n) {
  // Add vector `in` to vector `out` at specific offsets
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n[0])
    d_out[d_key_in[i]] += d_value_in[i];
}

__global__ void __vectorScatterMax(FloatT * d_out, IntT * d_key_in, FloatT * d_value_in, IntT * n) {
  // Add vector `in` to vector `out` at specific offsets
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n[0])
    d_out[d_key_in[i]] = max(d_value_in[i], d_out[d_key_in[i]]);
}

__global__ void __reorderColumns(FloatT* d_out, FloatT* d_in, IntT* d_map_r, IntT num_in, IntT num_out) {
  // Reorder columns of matrix
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < num_out) {
    IntT col = i % num_in;
    IntT row = i / num_in;
    d_out[i] = d_in[row * num_in + d_map_r[col]];
  }
}

template<typename Op>
void __row_reduce(FloatT * d_out, FloatT * d_in, IntT num_rows, IntT num_cols,
  Op reduce_op, double initial_value, IntT* d_offsets) {

  // Max over rows of matrix

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Max over rows
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_rows, d_offsets, d_offsets + 1, reduce_op, initial_value);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_rows, d_offsets, d_offsets + 1, reduce_op, initial_value);

  cudaFree(d_temp_storage);
}

// ================ Application Classification Specific ===============

namespace device {
  __global__ void NodePairwiseNorm(
    IntT DV,
    IntT PV,
    FloatT* CV,
    FloatT* MU,
    FloatT* data_node_feats,
    FloatT* patt_node_feats,
    IntT node_feat_dim
  ) {
    IntT k = threadIdx.x + blockIdx.x * blockDim.x;
    if(k < DV * PV) {
        IntT i = k / DV;
        IntT j = k % DV;
        FloatT dist = d_norm_2(
          patt_node_feats + i * node_feat_dim,
          data_node_feats + j * node_feat_dim,
          node_feat_dim
        );

        CV[k] = dist;
        MU[k] = -dist;
    }
  }

  __global__ void EdgePairwiseNorm(
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
      IntT i = k / DE;
      IntT j = k % DE;
      FloatT dist = d_norm_2(
        patt_edge_feats + i * edge_feat_dim,
        data_edge_feats + j * edge_feat_dim,
        edge_feat_dim
      );

      CE[k] = dist;
      RE[k] = - dist;
      FE[k] = - dist;
    }
  }

  __global__ void RepeatRowsByPatternEdges(
    IntT DV,
    IntT PE,
    IntT PV,
    FloatT * MU,
    FloatT * VR,
    FloatT * VF,
    IntT * patt_srcs,
    IntT * patt_dsts
  )
  {
    IntT k = threadIdx.x + blockDim.x * blockIdx.x;

    if(k < DV * PE) {
      IntT i = k / DV;
      IntT j = k % DV;
      VR[k] = MU[patt_srcs[i] * DV + j];
      VF[k] = MU[patt_dsts[i] * DV + j];
    }
  }

  __global__ void RepeatRowsByPatternEdgesSubtract(
    IntT DV,
    IntT PE,
    IntT PV,
    FloatT * MU,
    FloatT * VR,
    FloatT * VF,
    FloatT * FMax,
    FloatT * RMax,
    IntT * patt_srcs,
    IntT * patt_dsts
  )
  {

    IntT k = threadIdx.x + blockDim.x * blockIdx.x;
    if(k < DV * PE) {
      IntT i = k / DV;
      IntT j = k % DV;
      VR[k] = MU[patt_srcs[i] * DV + j] - RMax[k];
      VF[k] = MU[patt_dsts[i] * DV + j] - FMax[k];
    }
  }

  __global__ void RepeatRowsByDataEdges(
    IntT DE,
    IntT PE,
    IntT DV,
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
      IntT ij  = k / DE;
      IntT km  = k % DE;
      IntT src = srcs[km];

      FloatT CE_k = CE[k];
      FE[k] = - CE_k + VR[src + DV * ij];
      RE[k] = - CE_k + VF[src + DV * ij];
    }
  }
}

namespace host {

  void SortEdges(IntT* srcs, IntT* dsts, IntT* srcs_r, IntT* dsts_r, IntT* map_r, IntT num_edges) {
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

  void RowMax(IntT num_rows, IntT num_cols, FloatT* d_in, FloatT* d_out, IntT* d_offsets) {
    IntT block = 1 + (num_rows * num_cols) / THREAD;
    assert(THREAD * block > num_rows * num_cols);
    __row_reduce(d_out, d_in, num_rows, num_cols, cub::Max(), -DBL_MAX, d_offsets);
  }

  void RowSoftmax(const IntT num_rows, const IntT num_cols, FloatT *d_x, IntT* d_offsets) {
    // Compute softmax over columns

    // --------------------------
    // Prep

    IntT block  = 1 + (num_rows * num_cols) / THREAD;
    assert(THREAD * block > num_rows * num_cols);

    FloatT *d_storage;
    cudaMalloc((void**)&d_storage, num_cols * sizeof(FloatT));

    // ----------------------------
    // Compute column max

    __row_reduce(d_storage, d_x, num_rows, num_cols, cub::Max(), -DBL_MAX, d_offsets);

    // --------------------------------
    // Subtract max from columns

    __rowSubExp<<<block, THREAD>>>(d_x, num_rows, num_cols, d_storage);

    // --------------------------------
    // Sum columns

    cudaMemset(d_storage, 0, num_cols * sizeof(FloatT));
    __row_reduce(d_storage, d_x, num_rows, num_cols, cub::Sum(), 0, d_offsets);

    // ---------------------------------
    // Subtract log-sum from columns

    __rowSubLog<<<block, THREAD>>>(d_x, num_rows, num_cols, d_storage);

    // ---------------------------------
    // Free memory

    cudaFree(d_storage);
  }

  void EdgeMaxReduce(IntT num_rows_in, IntT num_rows_out, IntT num_cols,
    FloatT* d_VYmax, FloatT* d_XE, FloatT* d_XMax, IntT* nodes, IntT* map=NULL) {

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    IntT block_rowin_col  = 1 + (num_rows_in * num_cols) / THREAD;
    IntT block_rowout_col = 1 + (num_rows_out * num_cols) / THREAD;
    assert(THREAD * block_rowin_col > num_rows_in * num_cols);
    assert(THREAD * block_rowout_col > num_rows_out * num_cols);

    // --------------------------------------
    // Transpose

    FloatT *d_XEt;
    cudaMalloc((void**)&d_XEt, num_rows_in * num_cols * sizeof(FloatT));
    cudaMemcpy(d_XEt, d_XE, num_rows_in * num_cols * sizeof(FloatT), cudaMemcpyDeviceToDevice);

    // --------------------------------------
    // Tile dsts (w/ offset)

    IntT *d_tiled_nodes;
    cudaMalloc((void**)&d_tiled_nodes, num_rows_in * num_cols * sizeof(IntT));
    cudaMemset(d_tiled_nodes, 0, num_rows_in * num_cols * sizeof(IntT));
    __tileVectorWithOffset<<<block_rowin_col, THREAD>>>(d_tiled_nodes, nodes, num_rows_in, num_rows_out, num_rows_in * num_cols);

    // --------------------------------------
    // Reorder data (optional)

    if(map != NULL) {
      FloatT *d_XEr;
      cudaMalloc((void**)&d_XEr, num_rows_in * num_cols * sizeof(FloatT));
      __reorderColumns<<<block_rowin_col, THREAD>>>(d_XEr, d_XEt, map, num_rows_in, num_rows_in * num_cols);
      cudaMemcpy(d_XEt, d_XEr, num_rows_in * num_cols * sizeof(FloatT), cudaMemcpyDeviceToDevice);
      cudaFree(d_XEr);
    }

    // --------------------------------------
    // Max reduce rows of transposed matrix

    IntT   *d_keys_out;
    FloatT *d_values_out;
    IntT   *d_num_runs_out;

    cudaMalloc((void**)&d_keys_out,       num_rows_out * num_cols * sizeof(IntT));
    cudaMalloc((void**)&d_values_out,     num_rows_out * num_cols * sizeof(FloatT));
    cudaMalloc((void**)&d_num_runs_out,   1 * sizeof(IntT));

    d_temp_storage = NULL; temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
      d_tiled_nodes, d_keys_out, d_XEt, d_values_out, d_num_runs_out, cub::Max(), num_rows_in * num_cols);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
      d_tiled_nodes, d_keys_out, d_XEt, d_values_out, d_num_runs_out, cub::Max(), num_rows_in * num_cols);

    // --------------------------------------
    // Transpose result back to d_FMax

    __fillLow<<<block_rowout_col, THREAD>>>(d_XMax, num_cols * num_rows_out);
    __vectorScatterMax<<<block_rowout_col, THREAD>>>(d_XMax, d_keys_out, d_values_out, d_num_runs_out);

    // --------------------------------------
    // Elementwise max w/ V*max
    //   !! CNull hardcoded to 0, so ignore

    __maxMatrixColumnVector<<<block_rowout_col, THREAD>>>(d_XMax, d_VYmax, num_rows_out, num_rows_out * num_cols);

    // -------------------------------------
    // Free memory

    cudaFree(d_XEt);
    cudaFree(d_tiled_nodes);
    cudaFree(d_temp_storage);
    cudaFree(d_keys_out);
    cudaFree(d_values_out);
    cudaFree(d_num_runs_out);
  }


  void ComputeMU(Graph * patt, IntT DV, FloatT * d_CV, FloatT * d_FMax, FloatT * d_RMax, FloatT * d_MU) {
    // Replace columns of MU w/ sum over FMax/RMax of adjacent edges + subtract CV

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    IntT PV = patt->num_nodes;
    IntT PE = patt->num_edges;

    IntT block_dv_pv = 1 + (DV * PV) / THREAD;
    IntT block_dv_pe = 1 + (DV * PE) / THREAD;

    assert(THREAD * block_dv_pv > DV * PV);
    assert(THREAD * block_dv_pe > DV * PE);

    // --------------------------------------------
    // MU = -CV

    __scalarMultiply<<<block_dv_pv, THREAD>>>(d_MU, d_CV, (FloatT)-1.0, DV * PV);

    FloatT *d_MUt;
    cudaMalloc((void**)&d_MUt, PV * DV * sizeof(FloatT));
    __transpose<<<block_dv_pv, THREAD>>>(d_MUt, d_MU, PV, DV);

    // --------------------------------------------
    // Tile srcs

    IntT *d_tiled_nodes;
    cudaMalloc((void**)&d_tiled_nodes, DV * PE * sizeof(IntT));
    __tileVectorWithOffset<<<block_dv_pe, THREAD>>>(d_tiled_nodes, patt->srcs, PE, PV, DV * PE);

    // --------------------------------------------
    // Sum over rows of matrix

    FloatT *d_RMax_t;
    cudaMalloc((void**)&d_RMax_t, PE * DV * sizeof(FloatT));
    __transpose<<<block_dv_pe, THREAD>>>(d_RMax_t, d_RMax, PE, DV);

    IntT   *d_keys_out;
    FloatT *d_values_out;
    IntT   *d_num_runs_out;

    cudaMalloc((void**)&d_keys_out,       DV * PV * sizeof(IntT));
    cudaMalloc((void**)&d_values_out,     DV * PV * sizeof(FloatT));
    cudaMalloc((void**)&d_num_runs_out,   1 * sizeof(IntT));

    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
      d_tiled_nodes, d_keys_out, d_RMax_t, d_values_out, d_num_runs_out, cub::Sum(), DV * PE);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
      d_tiled_nodes, d_keys_out, d_RMax_t, d_values_out, d_num_runs_out, cub::Sum(), DV * PE);
    cudaFree(d_RMax_t);

    // --------------------------------------------
    // (Scatter) Add to MU

    __vectorScatterAdd<<<block_dv_pv, THREAD>>>(d_MUt, d_keys_out, d_values_out, d_num_runs_out);

    // --------------------------------------------
    // Tile dsts

    __tileVectorWithOffset<<<block_dv_pe, THREAD>>>(d_tiled_nodes, patt->dsts_r, PE, PV, DV * PE);

    // --------------------------------------
    // Reorder data

    FloatT *d_FMax_t;
    cudaMalloc((void**)&d_FMax_t, PE * DV * sizeof(FloatT));
    __transpose<<<block_dv_pe, THREAD>>>(d_FMax_t, d_FMax, PE, DV);

    FloatT *d_FMax_r;
    cudaMalloc((void**)&d_FMax_r, DV * PE * sizeof(FloatT));
    __reorderColumns<<<block_dv_pe, THREAD>>>(d_FMax_r, d_FMax_t, patt->map_r, PE, DV * PE);
    cudaFree(d_FMax_t);

    // --------------------------------------
    // Sum reduce

    d_temp_storage = NULL; temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
      d_tiled_nodes, d_keys_out, d_FMax_r, d_values_out, d_num_runs_out, cub::Sum(), DV * PE);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
      d_tiled_nodes, d_keys_out, d_FMax_r, d_values_out, d_num_runs_out, cub::Sum(), DV * PE);

    // (Scatter) add to d_MUt
    __vectorScatterAdd<<<block_dv_pv, THREAD>>>(d_MUt, d_keys_out, d_values_out, d_num_runs_out);

    __transpose<<<block_dv_pv, THREAD>>>(d_MU, d_MUt, DV, PV);

    // --------------------------------------
    // Free memory

    cudaFree(d_temp_storage);
    cudaFree(d_tiled_nodes);
    cudaFree(d_keys_out);
    cudaFree(d_values_out);
    cudaFree(d_num_runs_out);
    cudaFree(d_FMax_r);
    cudaFree(d_MUt);
  }

}

}