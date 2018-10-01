#ifndef __KERNELS
#define __KERNELS

#include <cub/cub.cuh>

// --
// Helpers

int ceil_pow2(int x) {
  // Round rows up to nearest power of 2
  return (int)pow(2, ceil(log(x)/log(2)));
}

// __global__ double norm_1(int num_attr, double * vec1) {
//   double sum = 0.0;
//   for (int i = 0; i < num_attr; i ++) {
//     sum += (vec1[i] * vec1[i]);
//   }
//   return sqrt(sum);
// }

__device__ void d_norm_2(int num_attr, double * vec1, double * vec2, double *out, int k) {
  double sum = 0.0;
  for (int i = 0; i < num_attr; i ++) {
    sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  }
  out[k] = sqrt(sum);
}

// --
// Kernels

__global__ void __NormProb(double * d_x, int num_cols, int num_rows)
{
    // Apply softmax to columns of `num_rows` x `num_cols` matrix

    extern __shared__ double sdata[];

    int row    = threadIdx.y;
    int col    = blockIdx.x;
    int offset = row * num_cols + col;

    // Compute max per column
    if(row < num_rows) {
      sdata[row] = d_x[offset];
    }
    __syncthreads();

    for(unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
      if((row < stride) && (row + stride < num_rows)) {
        sdata[row] = max(sdata[row], sdata[row + stride]);
      }
      __syncthreads();
    }

    // Subtract max value
    double max_value = sdata[0];
    if(row < num_rows) {
      sdata[row] = exp(d_x[offset] - max_value);
    }
    __syncthreads();

    // Compute sum of exp'd values
    for(unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
      if((row < stride) && (row + stride < num_rows)) {
        sdata[row] += sdata[row + stride];
      }
      __syncthreads();
    }

    // Subtract
    d_x[offset] = d_x[offset] - max_value - log(sdata[0]);
}

__global__ void __transpose(double *d_xt, double *d_x, int num_rows, int num_cols) {
  int offset = threadIdx.x + blockDim.x * blockIdx.x;
  int row = offset / num_cols;
  int col = offset % num_cols;
  d_xt[col * num_rows + row] = d_x[offset];
}

__global__ void __columnSub(double* d_x, int num_rows, int num_cols, double* c) {
  int offset = threadIdx.x + blockDim.x * blockIdx.x;
  int col = offset / num_rows;
  if(offset < num_rows * num_cols) {
    d_x[offset] -= c[col];
  }
}

__global__ void __logColumnSub(double* d_x, int num_rows, int num_cols, double* c) {
  int offset = threadIdx.x + blockDim.x * blockIdx.x;
  int col = offset / num_rows;
  if(offset < num_rows * num_cols) {
    d_x[offset] = log(d_x[offset]) - log(c[col]);
  }
}

__global__ void __vectorExp(double* d_x, int num_rows, int num_cols) {
  int offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    d_x[offset] = exp(d_x[offset]);
  }
}

void d_NormProb(int num_rows, int num_cols, double *d_x) {
  int thread = 1024;
  int block  = (int)ceil((num_rows * num_cols) / thread);

  // ----------------------------
  // Compute max of columns

  // Transpose matrix
  double *d_xt;
  cudaMalloc((void**)&d_xt, num_rows * num_cols * sizeof(double));
  __transpose<<<block, thread>>>(d_xt, d_x, num_rows, num_cols);

  // Compute offsets
  int h_offsets[num_cols + 1];
  int *d_offsets;
  for(int i = 0; i < num_cols + 1; i++) {
    h_offsets[i] = i * num_rows;
  }
  cudaMalloc((void**)&d_offsets, (num_cols + 1) * sizeof(int));
  cudaMemcpy(d_offsets, h_offsets, (num_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);

  // Initialize output
  // double h_colmax[num_cols];
  double *d_col_storage;
  cudaMalloc((void**)&d_col_storage, num_cols * sizeof(double));
  cudaMemset(d_col_storage, 0, num_cols * sizeof(double));

  // Compute column maxes
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage,
    temp_storage_bytes,
    d_xt,
    d_col_storage,
    num_cols,
    d_offsets,
    d_offsets + 1
  );
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage,
    temp_storage_bytes,
    d_xt,
    d_col_storage,
    num_cols,
    d_offsets,
    d_offsets + 1
  );

  // --------------------------------
  // Subtract max from columns

  __columnSub<<<block, thread>>>(d_xt, num_rows, num_cols, d_col_storage);

  // --------------------------------
  // Sum exp'd values

  __vectorExp<<<block, thread>>>(d_xt, num_rows, num_cols);

  cudaMemset(d_col_storage, 0, num_cols * sizeof(double));
  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage,
    temp_storage_bytes,
    d_xt,
    d_col_storage,
    num_cols,
    d_offsets,
    d_offsets + 1
  );
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage,
    temp_storage_bytes,
    d_xt,
    d_col_storage,
    num_cols,
    d_offsets,
    d_offsets + 1
  );

  // ---------------------------------
  // Subtract log-sum from columns

  __logColumnSub<<<block, thread>>>(d_xt, num_rows, num_cols, d_col_storage);

  // ---------------------------------
  // Copy back to original shape

  __transpose<<<block, thread>>>(d_x, d_xt, num_cols, num_rows);

  cudaFree(d_xt);
  cudaFree(d_col_storage);

  cudaDeviceSynchronize();
}

// void d_NormProb(int num_rows, int num_cols, double * d_x) {
//   dim3 block(num_cols, 1, 1);
//   dim3 thread(1, ceil_pow2(num_rows), 1);
//   int shmem_size = num_rows * sizeof(double);
//   __NormProb<<<block, thread, shmem_size>>>(d_x, num_cols, num_rows);
//   cudaDeviceSynchronize();
// }



__global__ void __d_Init_VR_VF(
  int num_AT,
  int num_PE,
  int num_PV,
  double * MU,
  double * VR,
  double * VF,
  int * PAttr
)
{
  int k = threadIdx.x + blockDim.x * blockIdx.x;

  int i = k / num_PE;
  int j = k % num_PE;
  int src = PAttr[j * num_AT];         // src id
  int dst = PAttr[j * num_AT + 1];     // dst id
  VR[k] = MU[i * num_PV + src];
  VF[k] = MU[i * num_PV + dst];

}

void d_Init_VR_VF(Graph * Data_Graph, Graph * Pattern_Graph, double * MU, double * VR, double * VF) {
  int num_DV  = Data_Graph->num_vertices;
  int num_PV  = Pattern_Graph->num_vertices;
  int num_PE  = Pattern_Graph->num_edges;
  int num_AT  = Pattern_Graph->Etable.num_cols;               // number of pattern edge attributes
  int * PAttr = (int *) Pattern_Graph->Etable.table;     // reading only vertex ids, so use uint64_t *

  int thread = 1024;
  int block  = (int)ceil((num_DV * num_PE) / thread);
  __d_Init_VR_VF<<<block, thread>>>(
    num_AT,
    num_PE,
    num_PV,
    MU,
    VR,
    VF,
    PAttr
  );
  cudaDeviceSynchronize();
}

__global__ void __d_Init_CE_RE_FE(
  int num_AT,
  int num_PE,
  double * PAttr,
  double * DAttr,
  double * CE,
  double * RE,
  double * FE
)
{
  int k = threadIdx.x + blockDim.x * blockIdx.x;

  int i = k / num_PE;
  int j = k % num_PE;
  d_norm_2(num_AT - 2, PAttr + j * num_AT + 2, DAttr + i * num_AT + 2, CE, k);
  RE[k] = - CE[k];
  FE[k] = - CE[k];
}

void d_Init_CE_RE_FE(Graph * Data_Graph, Graph * Pattern_Graph,
  double * CE, double * RE, double * FE) {

  uint64_t num_DE = Data_Graph->num_edges;
  uint64_t num_PE = Pattern_Graph->num_edges;
  uint64_t num_AT = Data_Graph->Etable.num_cols;                // number of edge attributes
  double * DAttr  = (double *) Data_Graph->Etable.table;        // reading only attributes, so use double *
  double * PAttr  = (double *) Pattern_Graph->Etable.table;     // reading only attributes, so use double *

  int thread = 1024;
  int block  = (int)ceil((num_DE * num_PE) / thread);
  __d_Init_CE_RE_FE<<<block, thread>>>(
    num_AT,
    num_PE,
    PAttr,
    DAttr,
    CE,
    RE,
    FE
  );
  cudaDeviceSynchronize();
}

#endif