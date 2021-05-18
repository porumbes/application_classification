#include <iostream>
#include <assert.h>
#include "main.h"
#include <cub/cub.cuh>
#include "thrust/device_vector.h"

#define THREAD 1024

__device__ static double atomicMax(double* address, double value) {
  unsigned long long* addr_as_longlong =
      reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *addr_as_longlong;
  unsigned long long expected;
  do {
    expected = old;
    old = ::atomicCAS(
        addr_as_longlong, expected,
        __double_as_longlong(::fmax(value, __longlong_as_double(expected))));
  } while (expected != old);
  return __longlong_as_double(old);
}

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

__global__ void __maxMatrixRowVector(FloatT * d_matrix, FloatT * d_vec, IntT num_cols, IntT n) {
  // Broadast row vector over matrix and take max
  IntT i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n) {
    d_matrix[i] = max(d_matrix[i], d_vec[i % num_cols]);
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
  Op reduce_op, double initial_value) {

  // Max over rows of matrix

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Compute offsets of matrix
  IntT *h_offsets = (IntT*)malloc((num_rows + 1) * sizeof(IntT));
  for(IntT i = 0; i < num_rows + 1; i++) {
    h_offsets[i] = i * num_cols;
  }
  IntT *d_offsets;
  cudaMalloc((void**)&d_offsets, (num_rows + 1) * sizeof(IntT));
  cudaMemcpy(d_offsets, h_offsets, (num_rows + 1) * sizeof(IntT), cudaMemcpyHostToDevice);

  // Max over rows
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_rows, d_offsets, d_offsets + 1, reduce_op, initial_value);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_rows, d_offsets, d_offsets + 1, reduce_op, initial_value);

  cudaFree(d_offsets);
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

  __global__ void RepeatColumnsByPatternEdges(
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
      IntT i = k / PE;
      IntT j = k % PE;
      VR[k] = MU[i * PV + patt_srcs[j]];
      VF[k] = MU[i * PV + patt_dsts[j]];
    }
  }

  __global__ void RepeatColumnsByPatternEdgesSubtract(
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
      IntT i = k / PE;
      IntT j = k % PE;
      VF[k] = MU[i * PV + patt_dsts[j]] - FMax[k];
      VR[k] = MU[i * PV + patt_srcs[j]] - RMax[k];
    }
  }

  __global__ void RepeatColumnsByDataEdges(
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
}

namespace host {

  // ------------------------------------------------

  void ColumnMax2(Int num_rows, Int num_cols, Real* d_in, Real* d_out) {
    auto op = [=]__device__(Int const& offset) {
      Int j = offset % num_cols;
      atomicMax(d_out + j, d_in[offset]);
    };
    
    auto it_start = thrust::make_counting_iterator<Int>(0);
    auto it_end   = thrust::make_counting_iterator<Int>(num_rows * num_cols);
    thrust::fill_n(thrust::device, d_out, num_cols, -999999);
    thrust::for_each(thrust::device, it_start, it_end, op);
  }
  
  void ColumnSoftmax2(const Int num_rows, const Int num_cols, Real* d_x) {
    Real* tmp;
    cudaMalloc(&tmp, num_cols * sizeof(Real));
    ColumnSoftmax2_prealloc(num_rows, num_cols, d_x, tmp);
    cudaFree(tmp);
  }

  void ColumnSoftmax2_prealloc(const Int num_rows, const Int num_cols, Real *d_x, Real* tmp) {
    cudaMemset(tmp, 0, num_cols * sizeof(Real));
    
    auto row_exp_sum = [=]__device__(Int const& offset) {
      Int j = offset % num_cols;
      atomicAdd(tmp + j, exp(d_x[offset]));
    };
    
    auto log_op = [=] __device__(Real const& val) -> double {
      return log(val);
    };

    auto sub_row = [=] __device__(Int const& offset) {
      Int j = offset % num_cols;
      d_x[offset] -= tmp[j];
    };
    
    auto it_start = thrust::make_counting_iterator<Int>(0);
    auto it_end   = thrust::make_counting_iterator<Int>(num_rows * num_cols);
    
    thrust::for_each(thrust::device, it_start, it_end, row_exp_sum);
    thrust::transform(thrust::device, tmp, tmp + num_cols, tmp, log_op);
    thrust::for_each(thrust::device, it_start, it_end, sub_row);
  }

  void EdgeMaxReduce2(
    IntT num_rows_in,  // n_edges
    IntT num_rows_out, // n_nodes
    IntT num_cols,
    FloatT* VYMax,
    FloatT* XE,
    FloatT* XMax, // output
    Int* nodes
  ) {
    auto fill = [=] __device__(Int const& offset) {
      XMax[offset] = VYMax[offset % num_cols];
    };
    
    auto op = [=] __device__(Int const& offset) {
      Int edge_idx = offset / num_cols;
      Int col      = offset % num_cols;
      Int src      = nodes[edge_idx];
      atomicMax(XMax + (src * num_cols) + col, XE[offset]);
    };

    auto it_start1 = thrust::make_counting_iterator<Int>(0);
    auto it_end1   = thrust::make_counting_iterator<Int>(num_rows_out * num_cols);    
    thrust::for_each(thrust::device, it_start1, it_end1, fill);
    
    auto it_start2 = thrust::make_counting_iterator<Int>(0);
    auto it_end2   = thrust::make_counting_iterator<Int>(num_rows_in * num_cols);
    thrust::for_each(thrust::device, it_start2, it_end2, op);
  }
  
  void ComputeMU2(
    Int row_in,
    Int col_in,
    Int row_out,
    Int col_out,
    Real* CV,
    Real* FMax,
    Real* RMax,
    Int* srcs,
    Int* dsts,
    Real* MU
  ) {
    
    auto mu_op = [=] __device__(Int const& offset) {
      auto row = offset / col_in;
      auto col = offset % col_in;
      auto src = srcs[col];
      auto dst = dsts[col];
      atomicAdd(MU + (row * col_out + dst), FMax[row * col_in + col]);
      atomicAdd(MU + (row * col_out + src), RMax[row * col_in + col]);
    };
    
    thrust::transform(
      thrust::device,
      CV,
      CV + (row_out * col_out),
      MU,
      [=] __device__(Real const& val) -> Real { return -val; }
    );
    
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      thrust::make_counting_iterator<Int>(row_in * col_in),
      mu_op
    );
  }
  
  // ------------------------------------------------



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

  void ColumnMax(IntT num_rows, IntT num_cols, FloatT* d_in, FloatT* d_out) {
    IntT block = 1 + (num_rows * num_cols) / THREAD;
    assert(THREAD * block > num_rows * num_cols);

    FloatT *d_in_t;
    cudaMalloc((void**)&d_in_t, num_rows * num_cols * sizeof(FloatT));
    __transpose<<<block, THREAD>>>(d_in_t, d_in, num_rows, num_cols);
    __row_reduce(d_out, d_in_t, num_cols, num_rows, cub::Max(), -DBL_MAX);
    cudaFree(d_in_t);
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
    __row_reduce(d_storage, d_xt, num_cols, num_rows, cub::Max(), -DBL_MAX);

    // --------------------------------
    // Subtract max from columns

    __rowSubExp<<<block, THREAD>>>(d_xt, num_cols, num_rows, d_storage);

    // --------------------------------
    // Sum columns

    cudaMemset(d_storage, 0, num_cols * sizeof(FloatT));
    __row_reduce(d_storage, d_xt, num_cols, num_rows, cub::Sum(), 0);

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
    __transpose<<<block_rowin_col, THREAD>>>(d_XEt, d_XE, num_rows_in, num_cols);

    // --------------------------------------
    // Tile dsts (w/ offset)

    IntT *d_tiled_nodes;
    cudaMalloc((void**)&d_tiled_nodes, num_rows_in * num_cols * sizeof(IntT));
    cudaMemset(d_tiled_nodes, 0, num_rows_in * num_cols * sizeof(IntT));
    __tileVectorWithOffset<<<block_rowin_col, THREAD>>>(d_tiled_nodes, nodes, num_rows_in, num_rows_out, num_rows_in * num_cols);

    // --------------------------------------
    // Reorder data (optional)

    if(map != NULL) {
      FloatT *d_XEt_r;
      cudaMalloc((void**)&d_XEt_r, num_rows_in * num_cols * sizeof(FloatT));
      __reorderColumns<<<block_rowin_col, THREAD>>>(d_XEt_r, d_XEt, map, num_rows_in, num_rows_in * num_cols);
      cudaMemcpy(d_XEt, d_XEt_r, num_rows_in * num_cols * sizeof(FloatT), cudaMemcpyDeviceToDevice);
      cudaFree(d_XEt_r);
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
    __transposeWithKey<<<block_rowout_col, THREAD>>>(d_XMax, d_values_out, d_keys_out, d_num_runs_out, num_cols, num_rows_out);

    // --------------------------------------
    // Elementwise max w/ V*max
    //   !! CNull hardcoded to 0, so ignore

    __maxMatrixRowVector<<<block_rowout_col, THREAD>>>(d_XMax, d_VYmax, num_cols, num_rows_out * num_cols);

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
    __reorderColumns<<<block_dv_pe, THREAD>>>(d_FMax_r, d_FMax, patt->map_r, PE, DV * PE);

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

}

}