#include <iostream>
#include <assert.h>
#include "main.h"
#include <cub/cub.cuh>
#include "thrust/device_vector.h"
#include <thrust/iterator/discard_iterator.h>

#define THREAD 1024

struct floor_functor {
   Int c;
   floor_functor(Int _c) : c(_c) {};
   __host__ __device__ Int operator() (const Int i) {
      return i / c;
      // return 0;
   }
};

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

namespace host {

  void RowMax2(Int num_rows, Int num_cols, Real* d_in, Real* d_out) {
    auto it_start = thrust::make_counting_iterator<Int>(0);
    auto it_end   = thrust::make_counting_iterator<Int>(num_rows * num_cols);
    
    thrust::equal_to<Int> binary_pred;
    thrust::maximum<Real> binary_op;
    
    thrust::reduce_by_key(
      thrust::device,
      thrust::make_transform_iterator(it_start, floor_functor(num_cols)),
      thrust::make_transform_iterator(it_end,   floor_functor(num_cols)),
      d_in,
      thrust::make_discard_iterator(),
      d_out,
      binary_pred,
      binary_op
    );
  }
  
  void RowSoftmax2(const Int num_rows, const Int num_cols, Real* d_x) {
    Real* tmp;
    cudaMalloc(&tmp, num_rows * sizeof(Real));
    RowSoftmax2_prealloc(num_rows, num_cols, d_x, tmp);
    cudaFree(tmp);
  }

  void RowSoftmax2_prealloc(const Int num_rows, const Int num_cols, Real *d_x, Real* tmp) {
    auto exp_op = [=] __device__(Real const& val) -> Real {
      return exp(val);
    };
        
    auto log_op = [=] __device__(Real const& val) -> Real {
      return log(val);
    };

    auto sub_row = [=] __device__(Int const& offset) {
      Int j = offset / num_cols;
      d_x[offset] -= tmp[j];
    };
    
    auto it_start = thrust::make_counting_iterator<Int>(0);
    auto it_end   = thrust::make_counting_iterator<Int>(num_rows * num_cols);
    
    thrust::transform(thrust::device, d_x, d_x + (num_rows * num_cols), d_x, exp_op);
    thrust::reduce_by_key(
      thrust::device,
      thrust::make_transform_iterator(it_start, floor_functor(num_cols)),
      thrust::make_transform_iterator(it_end, floor_functor(num_cols)),
      d_x,
      thrust::make_discard_iterator(),
      tmp
    );
    thrust::transform(thrust::device, tmp, tmp + num_rows, tmp, log_op);
    thrust::transform(thrust::device, d_x, d_x + (num_rows * num_cols), d_x, log_op);
    thrust::for_each(thrust::device, it_start, it_end, sub_row);
  }

  void EdgeMaxReduce2_t(
    IntT num_rows_in,  // n_edges
    IntT num_rows_out, // n_nodes
    IntT num_cols,
    FloatT* VYMax,
    FloatT* XE_t,
    FloatT* XMax_t, // output
    Int* nodes
  ) {
    auto fill = [=] __device__(Int const& offset) {
      Int j = offset / num_rows_out;
      XMax_t[offset] = VYMax[j];
    };
    
    auto op = [=] __device__(Int const& offset) {
      Int edge_idx = offset % num_rows_in;
      Int col      = offset / num_rows_in;
      Int src      = nodes[edge_idx];
      atomicMax(XMax_t + src + num_rows_out * col, XE_t[edge_idx + num_rows_in * col]);
    };

    auto it_start1 = thrust::make_counting_iterator<Int>(0);
    auto it_end1   = thrust::make_counting_iterator<Int>(num_rows_out * num_cols);    
    thrust::for_each(thrust::device, it_start1, it_end1, fill);
    
    auto it_start2 = thrust::make_counting_iterator<Int>(0);
    auto it_end2   = thrust::make_counting_iterator<Int>(num_rows_in * num_cols);
    thrust::for_each(thrust::device, it_start2, it_end2, op);
  }
  
  void ComputeMU2_t(
    Int row_in,
    Int col_in,
    Int row_out,
    Int col_out,
    Real* CV_t,
    Real* FMax_t,
    Real* RMax_t,
    Int* srcs,
    Int* dsts,
    Real* MU_t
  ) {
    
    auto mu_op = [=] __device__(Int const& offset) {
      auto row = offset % row_in;
      auto col = offset / row_in;
      auto src = srcs[col];
      auto dst = dsts[col];
      atomicAdd(MU_t + (row + row_out * dst), FMax_t[row + row_in * col]);
      atomicAdd(MU_t + (row + row_out * src), RMax_t[row + row_in * col]);
    };
    
    thrust::transform(
      thrust::device,
      CV_t,
      CV_t + (row_out * col_out),
      MU_t,
      [=] __device__(Real const& val) -> Real { return -val; }
    );
    
    thrust::for_each(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      thrust::make_counting_iterator<Int>(row_in * col_in),
      mu_op
    );
  }

}

}