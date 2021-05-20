#include <iostream>
#include "thrust/device_vector.h"
#include <thrust/iterator/discard_iterator.h>
#include <cub/cub.cuh>

#include "helpers.hxx"

template <typename Op>
struct row_reducer_t {

  Real* d_out;
  Real* d_in;
  Int num_rows;
  Int num_cols;
  Op reduce_op;
  Real initial_value;
  cudaStream_t stream;

  Int* d_offsets;
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  row_reducer_t(
    Real* _d_out,
    Real* _d_in,  
    Int _num_rows, 
    Int _num_cols, 
    Op _reduce_op, 
    Real _initial_value, 
    cudaStream_t _stream
  ) :
    d_out(_d_out),
    d_in(_d_in),
    num_rows(_num_rows),
    num_cols(_num_cols),
    reduce_op(_reduce_op),
    initial_value(_initial_value),
    stream(_stream)
  {

    Int *h_offsets = (Int*)malloc((num_rows + 1) * sizeof(Int));
    for(Int i = 0; i < num_rows + 1; i++) {
      h_offsets[i] = i * num_cols;
    }
    
    cudaMalloc((void**)&d_offsets, (num_rows + 1) * sizeof(Int));
    cudaMemcpy(d_offsets, h_offsets, (num_rows + 1) * sizeof(Int), cudaMemcpyHostToDevice);

    cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
      d_in, d_out, num_rows, d_offsets, d_offsets + 1, reduce_op, initial_value, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  }
  
  void run() {
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
      d_in, d_out, num_rows, d_offsets, d_offsets + 1, reduce_op, initial_value, stream);
  }
};

template<typename Op>
void __row_reduce(Real * d_out, Real * d_in, Int num_rows, Int num_cols, Op reduce_op, Real initial_value, cudaStream_t stream, 
  void* d_temp_storage, size_t temp_storage_bytes, Int* d_offsets) {

//   void *d_temp_storage = NULL;
//   size_t temp_storage_bytes = 0;

//   Int *h_offsets = (Int*)malloc((num_rows + 1) * sizeof(Int));
//   for(Int i = 0; i < num_rows + 1; i++) {
//     h_offsets[i] = i * num_cols;
//   }
//   Int *d_offsets;
//   cudaMalloc((void**)&d_offsets, (num_rows + 1) * sizeof(Int));
//   cudaMemcpy(d_offsets, h_offsets, (num_rows + 1) * sizeof(Int), cudaMemcpyHostToDevice);

//   cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
//     d_in, d_out, num_rows, d_offsets, d_offsets + 1, reduce_op, initial_value, stream);
//   cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
    d_in, d_out, num_rows, d_offsets, d_offsets + 1, reduce_op, initial_value, stream);

//   cudaFree(d_offsets);
//   cudaFree(d_temp_storage);
}

struct gpu_info {
    cudaStream_t stream;
    cudaEvent_t  event;
};

template <typename val_t>
void copy_n(val_t* in, val_t** out, Int n_gpus, Int n_rows, Int n_cols) {
  auto nbytes = n_rows * n_cols * sizeof(val_t);
  
  #pragma omp parallel for num_threads(n_gpus)
  for(Int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    cudaMemcpy(out[i], in, nbytes, cudaMemcpyDeviceToDevice);
  }
  
  cudaSetDevice(0);
}

template <typename val_t>
void shard_n(val_t* in, val_t** out, Int n_gpus, Int n_rows, Int n_cols, Int* starts, Int* ends) {
  
  #pragma omp parallel for num_threads(n_gpus)
  for(Int i = 0; i < n_gpus; i++) {
    
    cudaSetDevice(i);
    
    Int start  = starts[i];
    Int end    = ends[i];
    Int l_rows = end - start;
    
    auto nbytes = l_rows * n_cols * sizeof(val_t);
    
    cudaMemcpy(out[i], in + start * n_cols, nbytes, cudaMemcpyDeviceToDevice);
  }
  
  cudaSetDevice(0);
}

template <typename val_t>
void copy_alloc_n(val_t** out, Int n_gpus, Int n_rows, Int n_cols) {
  auto nbytes = n_rows * n_cols * sizeof(val_t);
  
  #pragma omp parallel for num_threads(n_gpus)
  for(Int i = 0; i < n_gpus; i++) {
    cudaSetDevice(i);
    
    val_t* tmp;
    cudaMalloc(&tmp, nbytes);
    out[i] = tmp;
  }
  
  cudaSetDevice(0);
}

template <typename val_t>
void shard_alloc_n(val_t** out, Int n_gpus, Int n_rows, Int n_cols, Int* starts, Int* ends) {
  
  #pragma omp parallel for num_threads(n_gpus)
  for(Int i = 0; i < n_gpus; i++) {
    
    cudaSetDevice(i);
    
    Int start  = starts[i];
    Int end    = ends[i];
    Int l_rows = end - start;
    
    auto nbytes = l_rows * n_cols * sizeof(val_t);
    
    val_t* tmp;
    cudaMalloc(&tmp, nbytes);
    out[i] = tmp;
  }
  
  cudaSetDevice(0);
}

namespace ac {

  struct floor_functor {
    Int c;
    floor_functor(Int _c) : c(_c) {};
    __host__ __device__ Int operator() (const Int i) {
        return i / c;
    }
  };

  template <typename val>
  void transpose(val* in, val* out, Int num_rows, Int num_cols) {
    auto op = [=]__device__(Int const& offset) {
      Int src_row = offset / num_cols;
      Int src_col = offset % num_cols;
      out[src_col * num_rows + src_row] = in[offset];
    };

    thrust::for_each_n(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      num_rows * num_cols,
      op
    );
  }
  
  void cdist(Int n_a, Int n_b, Int dim, Real* feats_a, Real* feats_b, Real* out) {
      
      auto cdist_op = [=] __device__(Int const& offset) {
        Int i = offset / n_a;
        Int j = offset % n_a;
        
        Real* vec1 = feats_a + (j * dim);
        Real* vec2 = feats_b + (i * dim);

        Real dist = 0.0;
        for (int i = 0; i < dim; i++)
          dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
        dist = sqrt(dist);

        out[offset] = dist;
      };
      
      thrust::for_each_n(
        thrust::device,
        thrust::make_counting_iterator<Int>(0),
        n_a * n_b,
        cdist_op
      );
  }

  void RowMax2(Int n_row, Int n_col, Real* d_in, Real* d_out) {
    auto it_start = thrust::make_counting_iterator<Int>(0);
    auto it_end   = thrust::make_counting_iterator<Int>(n_row * n_col);
    
    thrust::equal_to<Int> binary_pred;
    thrust::maximum<Real> binary_op;
    
    thrust::reduce_by_key(
      thrust::device,
      thrust::make_transform_iterator(it_start, floor_functor(n_col)),
      thrust::make_transform_iterator(it_end,   floor_functor(n_col)),
      d_in,
      thrust::make_discard_iterator(),
      d_out,
      binary_pred,
      binary_op
    );
  }

  void RowSoftmax2_prealloc(const Int n_row, const Int n_col, Real *d_x, Real* tmp, cudaStream_t stream) {
    auto exp_op  = [=] __device__(Real const& val) -> Real { return exp(val); };
    auto log_op  = [=] __device__(Real const& val) -> Real { return log(val); };
    auto sub_row = [=] __device__(Int const& offset) { d_x[offset] -= tmp[offset / n_col]; };
    
    auto it_start = thrust::make_counting_iterator<Int>(0);
    auto it_end   = thrust::make_counting_iterator<Int>(n_row * n_col);
    
    thrust::transform(thrust::device, d_x, d_x + (n_row * n_col), d_x, exp_op);
    thrust::reduce_by_key(
      thrust::cuda::par.on(stream),
      thrust::make_transform_iterator(it_start, floor_functor(n_col)),
      thrust::make_transform_iterator(it_end, floor_functor(n_col)),
      d_x,
      thrust::make_discard_iterator(),
      tmp
    );
    thrust::transform(thrust::cuda::par.on(stream), tmp, tmp + n_row, tmp, log_op);
    thrust::transform(thrust::cuda::par.on(stream), d_x, d_x + (n_row * n_col), d_x, log_op);
    thrust::for_each(thrust::cuda::par.on(stream), it_start, it_end, sub_row);
  }
  
  void RowSoftmax2(const Int n_row, const Int n_col, Real* d_x, cudaStream_t stream) {
    Real* tmp;
    cudaMalloc(&tmp, n_row * sizeof(Real));
    RowSoftmax2_prealloc(n_row, n_col, d_x, tmp, stream);
    cudaFree(tmp);
  }

  void EdgeMaxReduce2_t(
    Int data_num_edges,
    Int data_num_nodes,
    Int patt_num_edges,
    Real* VYmax,
    Real* XE_t,
    Real* XMax_t, // output
    Int* nodes
  ) {
    auto fill = [=] __device__(Int const& offset) {
      XMax_t[offset] = VYmax[offset / data_num_nodes];
    };
    
    auto op = [=] __device__(Int const& offset) {
      Int r = offset / data_num_edges;
      Int c = offset % data_num_edges;
      
      // random column write
      atomicMax(XMax_t + (data_num_nodes * r) + nodes[c], XE_t[offset]);
    };

    thrust::for_each_n(
      thrust::device,
      thrust::make_counting_iterator<Int>(0), 
      data_num_nodes * patt_num_edges, 
      fill
    );
    
    thrust::for_each_n(
      thrust::device,
      thrust::make_counting_iterator<Int>(0), 
      data_num_edges * patt_num_edges, 
      op
    );
  }


  void updateXMax_t(
    Int patt_num_nodes,
    Int patt_num_edges,
    Int data_num_nodes,
    Int data_num_edges,
    Real** all_CE_t,
    Real** all_VY_t,
    Real** all_VYmax,
    Real** all_XE_t,
    Real** all_XE_tmp,
    Real** all_XMax_t,
    Int** all_srcs,
    Int** all_nodes,
    Int n_gpus,
    Int* starts,
    Int* ends,
    Real* XMax_t_out,
    std::vector<gpu_info> infos,
    std::vector<row_reducer_t<decltype(cub::Max())>>& max_reducers,
    std::vector<row_reducer_t<decltype(cub::Sum())>>& sum_reducers
  ) {
    
    // VY_t   : (patt_num_edges, data_num_nodes) // read from subset of rows
    // XE_t   : (patt_num_edges, data_num_edges) // subset of rows from VY_t
    // VYmax  : (patt.num_edges, )               // subset of rows of VY_t
    // XMax_t : (patt.num_edges, data.num_nodes)
    
    // Broadcast CE_t
    
    #pragma omp parallel for num_threads(n_gpus)
    for(Int gid = 0; gid < n_gpus; gid++) {
      cudaSetDevice(gid);
      
      Int start = starts[gid];
      Int end   = ends[gid];
      Int size  = end - start;
      
      Real* CE_t   = all_CE_t[gid];
      Real* VY_t   = all_VY_t[gid];
      Real* VYmax  = all_VYmax[gid];
      Real* XE_t   = all_XE_t[gid];
      Real* XE_tmp = all_XE_tmp[gid];
      Real* XMax_t = all_XMax_t[gid];
      Int* srcs    = all_srcs[gid];
      Int* nodes   = all_nodes[gid];
      
      auto policy = thrust::cuda::par.on(infos[gid].stream);
      
      nvtxRangePushA("op1");
      auto update_XE = [=] __device__(Int const& offset) {
        Int r        = offset / data_num_edges;
        Int c        = offset % data_num_edges;
        XE_t[offset] = VY_t[data_num_nodes * r + srcs[c]] - CE_t[offset];
      };
      thrust::for_each_n(
        policy,
        thrust::make_counting_iterator<Int>(0),
        size * data_num_edges,
        update_XE
      );
      nvtxRangePop();
      
      // --
      
      nvtxRangePushA("op2");
      max_reducers[gid].run();
      nvtxRangePop();
      
      // --
      
      nvtxRangePushA("op3");
      auto exp_op  = [=] __device__(Real const& val) -> Real { return exp(val); };
      auto log_op  = [=] __device__(Real const& val) -> Real { return log(val); };
      auto sub_row = [=] __device__(Int const& idx)  -> void { XE_t[idx] -= XE_tmp[idx / data_num_edges]; };
          
      thrust::transform(policy, XE_t, XE_t + (size * data_num_edges), XE_t, exp_op);
      sum_reducers[gid].run();
      nvtxRangePop();
      
      nvtxRangePushA("op4");
      thrust::transform(policy, XE_tmp, XE_tmp + size, XE_tmp, log_op);
      thrust::transform(policy, XE_t, XE_t + (size * data_num_edges), XE_t, log_op);
      thrust::for_each(
        policy,
        thrust::make_counting_iterator<Int>(0),
        thrust::make_counting_iterator<Int>(size * data_num_edges),
        sub_row
      );
      nvtxRangePop();
      
      // --
      
      auto fill_op = [=] __device__(Int const& offset) {
        XMax_t[offset] = VYmax[offset / data_num_nodes];
      };
      
      auto max_op = [=] __device__(Int const& offset) {
        Int r = offset / data_num_edges;
        Int c = offset % data_num_edges;
        atomicMax(XMax_t + (data_num_nodes * r) + nodes[c], XE_t[offset]);
      };
      
      nvtxRangePushA("op5");
      thrust::for_each_n(
        policy, 
        thrust::make_counting_iterator<Int>(0), 
        size * data_num_nodes,
        fill_op
      );
      nvtxRangePop();
      
      nvtxRangePushA("op6");
      thrust::for_each_n(
        policy, 
        thrust::make_counting_iterator<Int>(0), 
        size * data_num_edges,
        max_op
      );
      nvtxRangePop();
      
      nvtxRangePushA("op7");
      thrust::copy_n( // could possibly update MU_t directly, instead of copying and then doing a single GPU
        policy,
        XMax_t,
        data_num_nodes * size,
        XMax_t_out + (data_num_nodes * start)
      );
      nvtxRangePop();
      
      cudaEventRecord(infos[gid].event, infos[gid].stream);
    }
    
    cudaSetDevice(0);
  }
  
  void ComputeMU2_t(
    Int n_col_in,
    Int n_row_in,
    Int n_col_out,
    Int n_row_out,
    Real* CV_t,
    Real* FMax_t,
    Real* RMax_t,
    Int* srcs,
    Int* dsts,
    Real* MU_t,
    cudaStream_t stream
  ) {
    
    auto mu_op = [=] __device__(Int const& offset) {
      auto r   = offset / n_col_in;
      auto c   = offset % n_col_in;
      // random row write -- bad
      atomicAdd(MU_t + (n_col_out * dsts[r] + c), FMax_t[offset]);
      atomicAdd(MU_t + (n_col_out * srcs[r] + c), RMax_t[offset]);
    };
    
    thrust::transform(
      thrust::cuda::par.on(stream),
      CV_t,
      CV_t + (n_col_out * n_row_out),
      MU_t,
      [=] __device__(Real const& val) -> Real { return -val; }
    );
    
    thrust::for_each_n(
      thrust::cuda::par.on(stream),
      thrust::make_counting_iterator<Int>(0),
      n_col_in * n_row_in,
      mu_op
    );
  }
  
}