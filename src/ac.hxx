#include <iostream>
#include "thrust/device_vector.h"
#include <thrust/iterator/discard_iterator.h>

#include "helpers.hxx"

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
    Real* g_CE_t,
    Real* g_VY_t,
    Real* g_VYmax,
    Real* g_XE_t,
    Real* g_XE_tmp,
    Real* g_XMax_t,
    Int* srcs,
    Int* nodes
  ) {
    
    // VY_t   : (patt_num_edges, data_num_nodes) // read from subset of rows
    // XE_t   : (patt_num_edges, data_num_edges) // subset of rows from VY_t
    // VYmax  : (patt.num_edges, )               // subset of rows of VY_t
    // XMax_t : (patt.num_edges, data.num_nodes)
    
    // Broadcast CE_t
    
    Int n_gpus     = 2;
    Int* starts    = (Int*)malloc(n_gpus * sizeof(Int));
    Int* ends      = (Int*)malloc(n_gpus * sizeof(Int));
    Int chunk_size = (patt_num_edges + n_gpus - 1) / n_gpus;
    for(Int i = 0; i < n_gpus; i++) {
        starts[i] = i       * chunk_size;
        ends[i]   = (i + 1) * chunk_size;
    }
    ends[n_gpus - 1] = patt_num_edges;
    
    for(Int i = 0; i < n_gpus; i++) {
      
      Int start = starts[i];
      Int end   = ends[i];
      Int size  = end - start;
      
      Real* CE_t;
      Real* VY_t;
      Real* VYmax;
      Real* XE_t;
      Real* XE_tmp;
      Real* XMax_t;
      
      cudaMalloc(&VYmax,                    size * sizeof(Real)); // local
      cudaMalloc(&CE_t,    data_num_edges * size * sizeof(Real)); // static
      cudaMalloc(&XE_t,    data_num_edges * size * sizeof(Real)); // local
      cudaMalloc(&VY_t,    data_num_nodes * size * sizeof(Real)); // needs copy to
      cudaMalloc(&XMax_t,  data_num_nodes * size * sizeof(Real)); // needs copy back
      cudaMalloc(&XE_tmp,                   size * sizeof(Real)); // local
      
      // also need static local copies of `data.srcs` and `data.dsts`
      
      cudaMemcpy(VYmax,    g_VYmax + start,                    size * sizeof(Real),  cudaMemcpyDeviceToDevice);
      cudaMemcpy(CE_t,      g_CE_t + start * data_num_edges,   data_num_edges * size * sizeof(Real),  cudaMemcpyDeviceToDevice);
      cudaMemcpy(XE_t,      g_XE_t + start * data_num_edges,   data_num_edges * size * sizeof(Real),  cudaMemcpyDeviceToDevice);
      cudaMemcpy(VY_t,      g_VY_t + start * data_num_nodes,   data_num_nodes * size * sizeof(Real),  cudaMemcpyDeviceToDevice);
      cudaMemcpy(XMax_t,  g_XMax_t + start * data_num_nodes,   data_num_nodes * size * sizeof(Real),  cudaMemcpyDeviceToDevice);
      cudaMemcpy(XE_tmp,  g_XE_tmp + start,                    size * sizeof(Real),  cudaMemcpyDeviceToDevice);

      // --
      
      auto update_XE = [=] __device__(Int const& offset) {
        Int r        = offset / data_num_edges;
        Int c        = offset % data_num_edges;
        XE_t[offset] = VY_t[data_num_nodes * r + srcs[c]] - CE_t[offset];
      };
      thrust::for_each_n(
        thrust::device,
        thrust::make_counting_iterator<Int>(0),
        size * data_num_edges,         // chunk: size
        update_XE
      );
      
      // --
      
      thrust::equal_to<Int> binary_pred;
      thrust::maximum<Real> binary_op;
      
      thrust::reduce_by_key(
        thrust::device,
        thrust::make_transform_iterator(
          thrust::make_counting_iterator<Int>(0), 
          floor_functor(data_num_nodes)
        ),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator<Int>(size * data_num_nodes), // chunk: size
          floor_functor(data_num_nodes)
        ),
        VY_t,
        thrust::make_discard_iterator(),
        VYmax,
        binary_pred,
        binary_op
      );
      
      // --
      
      auto exp_op  = [=] __device__(Real const& val) -> Real { return exp(val); };
      auto log_op  = [=] __device__(Real const& val) -> Real { return log(val); };
      auto sub_row = [=] __device__(Int const& idx)  -> void { XE_t[idx] -= XE_tmp[idx / data_num_edges]; };
          
      thrust::transform(thrust::device, XE_t, XE_t + (size * data_num_edges), XE_t, exp_op);
      thrust::reduce_by_key(
        thrust::device,
        thrust::make_transform_iterator(
          thrust::make_counting_iterator<Int>(0),
          floor_functor(data_num_edges)
        ),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator<Int>(size * data_num_edges), // chunk: size
          floor_functor(data_num_edges)
        ),
        XE_t,
        thrust::make_discard_iterator(),
        XE_tmp
      );
      thrust::transform(thrust::device, XE_tmp, XE_tmp + size, XE_tmp, log_op);
      thrust::transform(thrust::device, XE_t, XE_t + (size * data_num_edges), XE_t, log_op);
      thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<Int>(0),
        thrust::make_counting_iterator<Int>(size * data_num_edges), // chunk: size
        sub_row
      );
      
      // --
      
      auto fill_op = [=] __device__(Int const& offset) {
        XMax_t[offset] = VYmax[offset / data_num_nodes];
      };
      
      auto max_op = [=] __device__(Int const& offset) {
        Int r = offset / data_num_edges;
        Int c = offset % data_num_edges;
        atomicMax(XMax_t + (data_num_nodes * r) + nodes[c], XE_t[offset]);
      };

      thrust::for_each_n(
        thrust::device, 
        thrust::make_counting_iterator<Int>(0), 
        size * data_num_nodes, // chunk: size (?)
        fill_op
      );
      
      thrust::for_each_n(
        thrust::device, 
        thrust::make_counting_iterator<Int>(0), 
        size * data_num_edges, // chunk: size (?)
        max_op
      );      
      
      cudaMemcpy(g_XMax_t + (data_num_nodes * start),  XMax_t,   data_num_nodes * size * sizeof(Real),  cudaMemcpyDeviceToDevice);
      
      cudaFree(VYmax);
      cudaFree(CE_t);
      cudaFree(XE_t);
      cudaFree(VY_t);
      cudaFree(XMax_t);
      cudaFree(XE_tmp);
    }
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