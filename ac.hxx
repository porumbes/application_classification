#pragma once

#include <iostream>
#include "thrust/device_vector.h"
#include <thrust/iterator/discard_iterator.h>

typedef uint64_t Int;
typedef double Real;

typedef struct Graph {
  Int    num_nodes;
  Int    node_feat_dim;
  Real* node_feats;

  Int    num_edges;
  Int    edge_feat_dim;
  Real* edge_feats;

  Int* srcs;
  Int* dsts;

  Int* srcs_r;
  Int* dsts_r;
} Graph;

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

  void RowSoftmax2_prealloc(const Int n_row, const Int n_col, Real *d_x, Real* tmp) {
    auto exp_op = [=] __device__(Real const& val) -> Real {
      return exp(val);
    };
        
    auto log_op = [=] __device__(Real const& val) -> Real {
      return log(val);
    };

    auto sub_row = [=] __device__(Int const& offset) {
      Int r = offset / n_col;
      d_x[offset] -= tmp[r];
    };
    
    auto it_start = thrust::make_counting_iterator<Int>(0);
    auto it_end   = thrust::make_counting_iterator<Int>(n_row * n_col);
    
    thrust::transform(thrust::device, d_x, d_x + (n_row * n_col), d_x, exp_op);
    thrust::reduce_by_key(
      thrust::device,
      thrust::make_transform_iterator(it_start, floor_functor(n_col)),
      thrust::make_transform_iterator(it_end, floor_functor(n_col)),
      d_x,
      thrust::make_discard_iterator(),
      tmp
    );
    thrust::transform(thrust::device, tmp, tmp + n_row, tmp, log_op);
    thrust::transform(thrust::device, d_x, d_x + (n_row * n_col), d_x, log_op);
    thrust::for_each(thrust::device, it_start, it_end, sub_row);
  }
  
  void RowSoftmax2(const Int n_row, const Int n_col, Real* d_x) {
    Real* tmp;
    cudaMalloc(&tmp, n_row * sizeof(Real));
    RowSoftmax2_prealloc(n_row, n_col, d_x, tmp);
    cudaFree(tmp);
  }

  void EdgeMaxReduce2_t(
    int_fast8_t n_col_in,
    Int n_col_out,
    Int n_row,
    Real* VYMax,
    Real* XE_t,
    Real* XMax_t, // output
    Int* nodes
  ) {
    auto fill = [=] __device__(Int const& offset) {
      XMax_t[offset] = VYMax[offset / n_col_out];
    };
    
    auto op = [=] __device__(Int const& offset) {
      Int r = offset / n_col_in;
      Int c = offset % n_col_in;
      
      // random column write
      atomicMax(XMax_t + (n_col_out * r) + nodes[c], XE_t[offset]);
    };

    thrust::for_each_n(
      thrust::device, 
      thrust::make_counting_iterator<Int>(0), 
      n_col_out * n_row, 
      fill
    );
    
    thrust::for_each_n(
      thrust::device, 
      thrust::make_counting_iterator<Int>(0), 
      n_col_in * n_row, 
      op
    );
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
    Real* MU_t
  ) {
    
    auto mu_op = [=] __device__(Int const& offset) {
      auto r   = offset / n_col_in;
      auto c   = offset % n_col_in;
      // random row write -- bad
      atomicAdd(MU_t + (n_col_out * dsts[r] + c), FMax_t[offset]);
      atomicAdd(MU_t + (n_col_out * srcs[r] + c), RMax_t[offset]);
    };
    
    thrust::transform(
      thrust::device,
      CV_t,
      CV_t + (n_col_out * n_row_out),
      MU_t,
      [=] __device__(Real const& val) -> Real { return -val; }
    );
    
    thrust::for_each_n(
      thrust::device,
      thrust::make_counting_iterator<Int>(0),
      n_col_in * n_row_in,
      mu_op
    );
  }
  
}
}