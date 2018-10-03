#include <iostream>
#include <assert.h>
#include "main.h"
#include <cub/cub.cuh>

#define THREAD 1024

// --
// Helpers

void device2host(WorkArrays &h_WA, WorkArrays &d_WA, uint64_t DV, uint64_t DE, uint64_t PV, uint64_t PE) {
  cudaDeviceSynchronize();
  cudaMemcpy(h_WA.CV, d_WA.CV,    DV * PV * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.CE, d_WA.CE,    DE * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.Cnull, d_WA.Cnull, PE *      sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.MU, d_WA.MU,    DV * PV * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.RE, d_WA.RE,    DE * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.FE, d_WA.FE,    DE * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.VR, d_WA.VR,    DV * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.VF, d_WA.VF,    DV * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.VRmax, d_WA.VRmax, PE *      sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.VFmax, d_WA.VFmax, PE *      sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.RMax, d_WA.RMax,  DV * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_WA.FMax, d_WA.FMax,  DV * PE * sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

void host2device(WorkArrays &h_WA, WorkArrays &d_WA, uint64_t DV, uint64_t DE, uint64_t PV, uint64_t PE) {
  cudaDeviceSynchronize();
  cudaMemcpy(d_WA.CV, h_WA.CV,    DV * PV * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.CE, h_WA.CE,    DE * PE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.Cnull, h_WA.Cnull, PE *      sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.MU, h_WA.MU,    DV * PV * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.RE, h_WA.RE,    DE * PE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.FE, h_WA.FE,    DE * PE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.VR, h_WA.VR,    DV * PE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.VF, h_WA.VF,    DV * PE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.VRmax, h_WA.VRmax, PE *      sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.VFmax, h_WA.VFmax, PE *      sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.RMax, h_WA.RMax,  DV * PE * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_WA.FMax, h_WA.FMax,  DV * PE * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

// --
// Kernels

// __global__ double norm_1(int num_attr, double * vec1) {
//   double sum = 0.0;
//   for (int i = 0; i < num_attr; i ++) {
//     sum += (vec1[i] * vec1[i]);
//   }
//   return sqrt(sum);
// }

__device__ double d_norm_2(int num_attr, double * vec1, double * vec2) {
  double sum = 0.0;
  for (int i = 0; i < num_attr; i ++) {
    sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  }
  return sqrt(sum);
}

__global__ void __transpose(double *d_xt, double *d_x, uint64_t num_rows, uint64_t num_cols) {
  uint64_t offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    uint64_t row = offset / num_cols;
    uint64_t col = offset % num_cols;
    d_xt[col * num_rows + row] = d_x[offset];
  }
}

__global__ void __rowSubExp(double* d_x, uint64_t num_rows, uint64_t num_cols, double* c) {
  uint64_t offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    uint64_t row = offset / num_cols;
    d_x[offset] = exp(d_x[offset] - c[row]);
  }
}

__global__ void __rowSubLog(double* d_x, uint64_t num_rows, uint64_t num_cols, double* c) {
  uint64_t offset = threadIdx.x + blockDim.x * blockIdx.x;
  if(offset < num_rows * num_cols) {
    uint64_t row = offset / num_cols;
    d_x[offset] = log(d_x[offset]) - log(c[row]);
  }
}


void d_rowmax(double * d_out, double * d_in, uint64_t num_rows, uint64_t num_cols) {

  // Compute offsets of transpose matrix
  uint64_t h_offsets[num_rows + 1];
  uint64_t *d_offsets;
  for(uint64_t i = 0; i < num_rows + 1; i++) {
    h_offsets[i] = i * num_cols;
  }
  cudaMalloc((void**)&d_offsets, (num_rows + 1) * sizeof(uint64_t));
  cudaMemcpy(d_offsets, h_offsets, (num_rows + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Initialize output storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

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
}

void d_rowsum(double * d_out, double * d_in, uint64_t num_rows, uint64_t num_cols) {

  // Compute offsets of transpose matrix
  uint64_t h_offsets[num_rows + 1];
  uint64_t *d_offsets;
  for(uint64_t i = 0; i < num_rows + 1; i++) {
    h_offsets[i] = i * num_cols;
  }
  cudaMalloc((void**)&d_offsets, (num_rows + 1) * sizeof(uint64_t));
  cudaMemcpy(d_offsets, h_offsets, (num_rows + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // Initialize output storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

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

// ================ Specific ===============

__global__ void __pairwiseNorm(
  int num_DV,
  int num_PV,
  int num_AT,
  double* CV,
  double* MU,
  double* PAttr,
  double* DAttr
) {
  uint64_t k = threadIdx.x + blockIdx.x * blockDim.x;
  if(k < num_DV * num_PV) {
      uint64_t i = k / num_PV;
      uint64_t j = k % num_PV;
      double tmp = d_norm_2(num_AT - 1, PAttr + j * num_AT + 1, DAttr + i * num_AT + 1);
      CV[k] = tmp;
      MU[k] = -tmp;
  }
}

void d_Init_CV_MU(Graph* d_Data_Graph, Graph* d_Pattern_Graph, double* d_CV, double* d_MU) {
  uint64_t num_DV = d_Data_Graph->num_vertices;
  uint64_t num_PV = d_Pattern_Graph->num_vertices;
  uint64_t num_AT = d_Data_Graph->Vtable.num_cols;
  double * d_DAttr  = (double *) d_Data_Graph->Vtable.table;
  double * d_PAttr  = (double *) d_Pattern_Graph->Vtable.table;

  int block = 1 + (num_DV * num_PV) / THREAD;
  __pairwiseNorm<<<block, THREAD>>>(num_DV, num_PV, num_AT, d_CV, d_MU, d_PAttr, d_DAttr);

}

void d_VFmax_VRmax(Graph * d_Data_Graph, Graph * d_Pattern_Graph,
                 double * d_VF, double * d_VR, double * d_VFmax, double * d_VRmax) {

  uint64_t num_rows = d_Data_Graph->num_vertices; // DV
  uint64_t num_cols = d_Pattern_Graph->num_edges; // PE

  uint64_t block  = 1 + (num_rows * num_cols) / THREAD;
  assert(THREAD * block > num_rows * num_cols);

  double *d_VFt;
  cudaMalloc((void**)&d_VFt, num_rows * num_cols * sizeof(double));
  __transpose<<<block, THREAD>>>(d_VFt, d_VF, num_rows, num_cols);
  d_rowmax(d_VFmax, d_VFt, num_cols, num_rows);

  double *d_VRt;
  cudaMalloc((void**)&d_VRt, num_rows * num_cols * sizeof(double));
  __transpose<<<block, THREAD>>>(d_VRt, d_VR, num_rows, num_cols);
  d_rowmax(d_VRmax, d_VRt, num_cols, num_rows);

  cudaFree(d_VFt);
  cudaFree(d_VRt);
  cudaDeviceSynchronize();
}



void d_NormProb(const uint64_t num_rows, const uint64_t num_cols, double *d_x) {

  // --------------------------
  // Prep

  uint64_t block  = 1 + (num_rows * num_cols) / THREAD;
  assert(THREAD * block > num_rows * num_cols);

  double *d_storage;
  double h_storage[num_cols];
  cudaMalloc((void**)&d_storage, num_cols * sizeof(double));

  // ----------------------------
  // Compute column max

  double *d_xt;
  cudaMalloc((void**)&d_xt, num_rows * num_cols * sizeof(double));
  __transpose<<<block, THREAD>>>(d_xt, d_x, num_rows, num_cols);

  d_rowmax(d_storage, d_xt, num_cols, num_rows);
  cudaMemcpy(h_storage, d_storage, num_cols * sizeof(double), cudaMemcpyDeviceToHost);

  // --------------------------------
  // Subtract max from columns

  __rowSubExp<<<block, THREAD>>>(d_xt, num_cols, num_rows, d_storage);

  // --------------------------------
  // Sum columns

  cudaMemset(d_storage, 0, num_cols * sizeof(double));
  d_rowsum(d_storage, d_xt, num_cols, num_rows);

  // ---------------------------------
  // Subtract log-sum from columns

  __rowSubLog<<<block, THREAD>>>(d_xt, num_cols, num_rows, d_storage);

  // ---------------------------------
  // Transpose back to original shape

  __transpose<<<block, THREAD>>>(d_x, d_xt, num_cols, num_rows);

  cudaFree(d_xt);
  cudaFree(d_storage);

  cudaDeviceSynchronize();
}



__global__ void __d_Init_VR_VF(
  const int num_AT,
  const int num_DV,
  const int num_PE,
  const int num_PV,
  double * MU,
  double * VR,
  double * VF,
  const uint64_t * d_PAttr
)
{
  int k = threadIdx.x + blockDim.x * blockIdx.x;

  if(k < num_DV * num_PE) {
    int i   = k / num_PE;
    int j   = k % num_PE;
    int src = (int)d_PAttr[j * num_AT];         // src id
    int dst = (int)d_PAttr[j * num_AT + 1];     // dst id

    VR[k] = MU[i * num_PV + src];
    VF[k] = MU[i * num_PV + dst];
  }
}

void d_Init_VR_VF(Graph * d_Data_Graph, Graph * d_Pattern_Graph, double * MU, double * VR, double * VF) {
  int num_DV  = (int)d_Data_Graph->num_vertices;
  int num_PV  = (int)d_Pattern_Graph->num_vertices;
  int num_PE  = (int)d_Pattern_Graph->num_edges;
  int num_AT  = (int)d_Pattern_Graph->Etable.num_cols;

  uint64_t* d_PAttr = (uint64_t *)d_Pattern_Graph->Etable.table;

  int block  = 1 + (num_DV * num_PE) / THREAD;
  assert(THREAD * block > num_DV * num_PE);

  __d_Init_VR_VF<<<block, THREAD>>>(
    num_AT,
    num_DV,
    num_PE,
    num_PV,
    MU,
    VR,
    VF,
    d_PAttr
  );
  cudaDeviceSynchronize();
}



__global__ void __d_Init_CE_RE_FE(
  uint64_t num_AT,
  uint64_t num_PE,
  double * PAttr,
  double * DAttr,
  double * CE,
  double * RE,
  double * FE
)
{
  uint64_t k = threadIdx.x + blockDim.x * blockIdx.x;

  uint64_t i = k / num_PE;
  uint64_t j = k % num_PE;
  double tmp = d_norm_2(num_AT - 2, PAttr + j * num_AT + 2, DAttr + i * num_AT + 2);
  CE[k] = tmp;
  RE[k] = - CE[k];
  FE[k] = - CE[k];
}

void d_Init_CE_RE_FE(Graph * d_Data_Graph, Graph * d_Pattern_Graph,
  double * d_CE, double * d_RE, double * d_FE) {

  uint64_t num_DE = d_Data_Graph->num_edges;
  uint64_t num_PE = d_Pattern_Graph->num_edges;
  uint64_t num_AT = d_Data_Graph->Etable.num_cols;                // number of edge attributes
  double * d_DAttr  = (double *) d_Data_Graph->Etable.table;        // reading only attributes, so use double *
  double * d_PAttr  = (double *) d_Pattern_Graph->Etable.table;     // reading only attributes, so use double *

  uint64_t block  = 1 + (num_DE * num_PE) / THREAD;
  assert(THREAD * block > num_DE * num_PE);

  __d_Init_CE_RE_FE<<<block, THREAD>>>(
    num_AT,
    num_PE,
    d_PAttr,
    d_DAttr,
    d_CE,
    d_RE,
    d_FE
  );
  cudaDeviceSynchronize();
}



__global__ void __d_VF_VR(
  uint64_t num_AT,
  uint64_t num_DV,
  uint64_t num_PE,
  uint64_t num_PV,
  uint64_t * PAttr,
  double * MU,
  double * VR,
  double * VF,
  double * FMax,
  double * RMax
)
{

  uint64_t k = threadIdx.x + blockDim.x * blockIdx.x;
  if(k < num_DV * num_PE) {
    uint64_t i = k / num_PE;
    uint64_t j = k % num_PE;

    uint64_t src = PAttr[j * num_AT];         // src id
    uint64_t dst = PAttr[j * num_AT + 1];     // dst id
    VF[k] = MU[i * num_PV + dst] - FMax[k];
    VR[k] = MU[i * num_PV + src] - RMax[k];
  }
}

void d_VF_VR(Graph * d_Data_Graph, Graph * d_Pattern_Graph,
           double * d_MU, double * d_FMax, double * d_RMax, double * d_VF, double * d_VR) {

  uint64_t num_DV  = d_Data_Graph->num_vertices;
  uint64_t num_PV  = d_Pattern_Graph->num_vertices;
  uint64_t num_PE  = d_Pattern_Graph->num_edges;
  uint64_t num_AT  = d_Pattern_Graph->Etable.num_cols;
  uint64_t * d_PAttr = (uint64_t *) d_Pattern_Graph->Etable.table;

  uint64_t block  = 1 + (num_DV * num_PE) / THREAD;
  assert(THREAD * block > num_DV * num_PE);

  __d_VF_VR<<<block, THREAD>>>(
    num_AT,
    num_DV,
    num_PE,
    num_PV,
    d_PAttr,
    d_MU,
    d_VR,
    d_VF,
    d_FMax,
    d_RMax
  );
  cudaDeviceSynchronize();
}

__global__ void __copyChangeSign(double * d_out, double * d_in, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n) {
    d_out[i] = -d_in[i];
  }
}

__global__ void __tileVector(uint64_t * d_out, uint64_t * d_in,
  int num_in, int num_out) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < num_out) {
    d_out[i] = d_in[i % num_in];
  }
}

__global__ void __tileVectorOffset(uint64_t * d_out, uint64_t * d_in,
    int num_in, int num_uin, int num_out) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < num_out) {
    d_out[i] = num_uin * (i / num_in) + d_in[i % num_in];
  }
}

__global__ void __vectorAdd(double * d_out, double * d_in, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n) {
    d_out[i] += d_in[i];
  }
}

__global__ void __vectorScatterAdd(double * d_out, uint64_t * d_key_in, double * d_value_in, int * n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n[0]) {
    d_out[d_key_in[i]] += d_value_in[i];
  }
}

__global__ void __tileMax(double * d_out, double * d_in, int num_in, int num_out) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < num_out) {
    d_out[i] = max(d_out[i], d_in[i % num_in]);
  }
}

__global__ void __MU(
  uint64_t num_DV,
  uint64_t num_PV,
  uint64_t num_PE,
  uint64_t num_AT,
  uint64_t * srcs,
  uint64_t * dsts,
  double * FMax,
  double * RMax,
  double * MU
)
{
  // Sum reduce columns of FMax/RMax by key
  // Some of the keys are sequential, some are not
  uint64_t k = threadIdx.x + blockDim.x * blockIdx.x;
  if(k < num_DV * num_PE) {
    uint64_t i = k / num_PE;
    uint64_t j = k % num_PE;
    atomicAdd(&MU[i * num_PV + dsts[j]], FMax[k]);
    atomicAdd(&MU[i * num_PV + srcs[j]], RMax[k]);
  }
}

void d_UpdateMU(Graph * d_Data_Graph, Graph * d_Pattern_Graph, double * d_CV,
  double * d_FMax, double * d_RMax, double * d_MU) {

  uint64_t num_DV = d_Data_Graph->num_vertices;
  uint64_t num_PV = d_Pattern_Graph->num_vertices;
  uint64_t num_PE = d_Pattern_Graph->num_edges;
  uint64_t * d_srcs = d_Pattern_Graph->Etable.srcs;
  uint64_t * d_dsts = d_Pattern_Graph->Etable.dsts;

  // MU = -CV
  uint64_t block_vv = 1 + (num_DV * num_PV) / THREAD;
  assert(THREAD * block_vv > num_DV * num_PV);
  __copyChangeSign<<<block_vv, THREAD>>>(d_MU, d_CV, num_DV * num_PV);

  // Sum reduce
  uint64_t block_ve = 1 + (num_DV * num_PE) / THREAD;
  assert(THREAD * block_ve > num_DV * num_PE);

  // --------------------------------------------
  // Reduce over `src`

  uint64_t *d_tiled_srcs;
  cudaMalloc((void**)&d_tiled_srcs, num_DV * num_PE * sizeof(uint64_t));
  cudaMemset(d_tiled_srcs, 0, num_DV * num_PE * sizeof(uint64_t));
  __tileVector<<<block_ve, THREAD>>>(d_tiled_srcs, d_srcs, num_PE, num_DV * num_PE);

  int num_items = num_DV * num_PE;
  uint64_t *d_keys_in;
  uint64_t *d_keys_out;
  double   *d_values_in;
  double   *d_values_out;
  int *d_num_runs_out;

  cudaMalloc((void**)&d_keys_in,        num_DV * num_PE * sizeof(uint64_t));
  cudaMalloc((void**)&d_keys_out,       num_DV * num_PV * sizeof(uint64_t));
  cudaMalloc((void**)&d_values_in,      num_DV * num_PE * sizeof(double));
  cudaMalloc((void**)&d_values_out,     num_DV * num_PV * sizeof(double));
  cudaMalloc((void**)&d_num_runs_out,   1 * sizeof(int));

  cudaMemcpy(d_keys_in,   d_tiled_srcs, num_DV * num_PE * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_values_in, d_RMax,       num_DV * num_PE * sizeof(double), cudaMemcpyDeviceToDevice);

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs_out, cub::Sum(), num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs_out, cub::Sum(), num_items);

  // !! Probably need scatterAdd here
  __vectorAdd<<<block_vv, THREAD>>>(d_MU, d_values_out, num_DV * num_PV);

  // --------------------------------------------
  // Reduce over `dst`

  // Tile
  uint64_t *d_tiled_dsts;
  cudaMalloc((void**)&d_tiled_dsts, num_DV * num_PE * sizeof(uint64_t));
  cudaMemset(d_tiled_dsts, 0, num_DV * num_PE * sizeof(uint64_t));
  __tileVectorOffset<<<block_ve, THREAD>>>(d_tiled_dsts, d_dsts, num_PE, num_PV, num_DV * num_PE);

  uint64_t *d_keys_tmp;
  double *d_values_tmp;
  cudaMalloc((void**)&d_keys_tmp,   num_DV * num_PE * sizeof(uint64_t));
  cudaMalloc((void**)&d_values_tmp, num_DV * num_PE * sizeof(double));

  cudaMemcpy(d_keys_in,   d_tiled_dsts, num_DV * num_PE * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_values_in, d_FMax,       num_DV * num_PE * sizeof(double), cudaMemcpyDeviceToDevice);

  cudaMemset(d_keys_out,   0, num_DV * num_PV * sizeof(uint64_t));
  cudaMemset(d_values_out, 0, num_DV * num_PV * sizeof(double));

  // Sort by dst
  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_tmp, d_values_in, d_values_tmp, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_tmp, d_values_in, d_values_tmp, num_items);

  // Sum over dst
  d_temp_storage = NULL;
  temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_keys_tmp, d_keys_out, d_values_tmp, d_values_out, d_num_runs_out, cub::Sum(), num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_keys_tmp, d_keys_out, d_values_tmp, d_values_out, d_num_runs_out, cub::Sum(), num_items);

  // Add to d_MU
  __vectorScatterAdd<<<block_vv, THREAD>>>(d_MU, d_keys_out, d_values_out, d_num_runs_out);

  cudaDeviceSynchronize();
}

__global__ void __d_FE_RE(
  int num_DE,
  int num_PE,
  uint64_t * srcs,
  double * CE,
  double * VR,
  double * VF,
  double * FE,
  double * RE
)
{
  uint64_t k = threadIdx.x + blockDim.x * blockIdx.x;
  if(k < num_DE * num_PE) {
    uint64_t ij  = k / num_PE;
    uint64_t km  = k % num_PE;
    uint64_t src = srcs[ij];
    FE[k] = - CE[k] + VR[src * num_PE + km];
    RE[k] = - CE[k] + VF[src * num_PE + km];
  }
}

void d_FE_RE(Graph * d_Data_Graph, Graph * d_Pattern_Graph,
           double * d_CE, double * d_VF, double * d_VR, double * d_FE, double * d_RE) {

  uint64_t num_DE   = d_Data_Graph->num_edges;
  uint64_t num_PE   = d_Pattern_Graph->num_edges;
  uint64_t * d_srcs = d_Data_Graph->Etable.srcs;

  int block = 1 + (num_DE * num_PE) / THREAD;
  assert(THREAD * block > num_DE * num_PE);
  __d_FE_RE<<<block, THREAD>>>(
    num_DE,
    num_PE,
    d_srcs,
    d_CE,
    d_VR,
    d_VF,
    d_FE,
    d_RE
  );
}

void d_RMax(Graph * d_Data_Graph, Graph * d_Pattern_Graph, double * d_Cnull, double * d_VFmax, double * d_RE, double * d_RMax) {

  uint64_t *d_srcs = d_Data_Graph->Etable.srcs;

  // --
  // Max reduction-by-key over rows

  uint64_t num_DV = d_Data_Graph->num_vertices;
  uint64_t num_DE = d_Data_Graph->num_edges;
  uint64_t num_PE = d_Pattern_Graph->num_edges;

  int block_ee = 1 + (num_DE * num_PE) / THREAD;
  assert(THREAD * block_ee > num_DE * num_PE);

  double *d_REt; // num_PE x num_DE
  cudaMalloc((void**)&d_REt, num_DE * num_PE * sizeof(double));
  __transpose<<<block_ee, THREAD>>>(d_REt, d_RE, num_DE, num_PE);

  uint64_t *d_tiled_srcs;
  cudaMalloc((void**)&d_tiled_srcs, num_DE * num_PE * sizeof(uint64_t));
  cudaMemset(d_tiled_srcs, 0, num_DE * num_PE * sizeof(uint64_t));
  __tileVector<<<block_ee, THREAD>>>(d_tiled_srcs, d_srcs, num_DE, num_DE * num_PE);

  uint64_t h_tiled_srcs[num_DE * num_PE];
  cudaMemcpy(h_tiled_srcs, d_tiled_srcs, num_DE * num_PE * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  int num_items = num_DE * num_PE;
  uint64_t *d_keys_in;
  uint64_t *d_keys_out;
  double   *d_values_in;
  double   *d_values_out;
  int *d_num_runs_out;

  cudaMalloc((void**)&d_keys_in,        num_DE * num_PE * sizeof(uint64_t));
  cudaMalloc((void**)&d_keys_out,       num_DV * num_PE * sizeof(uint64_t));
  cudaMalloc((void**)&d_values_in,      num_DE * num_PE * sizeof(double));
  cudaMalloc((void**)&d_values_out,     num_DV * num_PE * sizeof(double));
  cudaMalloc((void**)&d_num_runs_out,   1 * sizeof(int));

  cudaMemcpy(d_keys_in,   d_tiled_srcs, num_DE * num_PE * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_values_in, d_REt,        num_DE * num_PE * sizeof(double), cudaMemcpyDeviceToDevice);

  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs_out, cub::Max(), num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out, d_values_in, d_values_out, d_num_runs_out, cub::Max(), num_items);

  int block_ev = 1 + (num_DV * num_PE) / THREAD;
  assert(THREAD * block_ev > num_DV * num_PE);
  __transpose<<<block_ev, THREAD>>>(d_RMax, d_values_out, num_PE, num_DV);

  // !! CNull hardcoded to 0
  __tileMax<<<block_ev, THREAD>>>(d_RMax, d_VFmax, num_PE, num_DV * num_PE);
}


void d_FMax(Graph* d_Data_Graph, Graph* d_Pattern_Graph, double* d_Cnull, double* d_VRmax, double* d_FE, double* d_FMax) {

  uint64_t *d_dsts = d_Data_Graph->Etable.dsts;

  // --
  // Max reduction-by-key over rows

  uint64_t num_DV = d_Data_Graph->num_vertices;
  uint64_t num_DE = d_Data_Graph->num_edges;
  uint64_t num_PE = d_Pattern_Graph->num_edges;

  int block_ee = 1 + (num_DE * num_PE) / THREAD;
  assert(THREAD * block_ee > num_DE * num_PE);

  double *d_FEt; // num_PE x num_DE
  cudaMalloc((void**)&d_FEt, num_DE * num_PE * sizeof(double));
  __transpose<<<block_ee, THREAD>>>(d_FEt, d_FE, num_DE, num_PE);


  uint64_t *d_tiled_dsts;
  cudaMalloc((void**)&d_tiled_dsts, num_DE * num_PE * sizeof(uint64_t));
  cudaMemset(d_tiled_dsts, 0, num_DE * num_PE * sizeof(uint64_t));
  __tileVectorOffset<<<block_ee, THREAD>>>(d_tiled_dsts, d_dsts, num_DE, num_DV, num_DE * num_PE);

  // uint64_t h_tiled_dsts[num_DE * num_PE];
  // cudaMemcpy(h_tiled_dsts, d_tiled_dsts, num_DE * num_PE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  // for(int i = 0; i < 20; i ++) {
  //   fprintf(stderr, "h_tiled_dsts[%lu]=%d\n", i, h_tiled_dsts[i]);
  // }
  // for(int i = num_DE * num_PE - 10; i < num_DE * num_PE; i ++) {
  //   fprintf(stderr, "h_tiled_dsts[%lu]=%d\n", i, h_tiled_dsts[i]);
  // }

  int num_items = num_DE * num_PE;
  uint64_t *d_keys_in;
  uint64_t *d_keys_out;
  double   *d_values_in;
  double   *d_values_out;
  int *d_num_runs_out;

  cudaMalloc((void**)&d_keys_in,        num_DE * num_PE * sizeof(uint64_t));
  cudaMalloc((void**)&d_keys_out,       num_DV * num_PE * sizeof(uint64_t));
  cudaMalloc((void**)&d_values_in,      num_DE * num_PE * sizeof(double));
  cudaMalloc((void**)&d_values_out,     num_DV * num_PE * sizeof(double));
  cudaMalloc((void**)&d_num_runs_out,   1 * sizeof(int));

  cudaMemcpy(d_keys_in,   d_tiled_dsts, num_DE * num_PE * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_values_in, d_FEt,        num_DE * num_PE * sizeof(double), cudaMemcpyDeviceToDevice);

  uint64_t *d_keys_tmp;
  double *d_values_tmp;
  cudaMalloc((void**)&d_keys_tmp,   num_DE * num_PE * sizeof(uint64_t));
  cudaMalloc((void**)&d_values_tmp, num_DE * num_PE * sizeof(double));

  // Can precompute
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_tmp, d_values_in, d_values_tmp, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_tmp, d_values_in, d_values_tmp, num_items);

  // uint64_t h_tiled_dsts[num_DE * num_PE];
  // cudaMemcpy(h_tiled_dsts, d_keys_tmp, num_DE * num_PE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  // for(int i = 0; i < 20; i ++) {
  //   fprintf(stderr, "h_tiled_dsts[%lu]=%d\n", i, h_tiled_dsts[i]);
  // }
  // for(int i = num_DE * num_PE - 10; i < num_DE * num_PE; i ++) {
  //   fprintf(stderr, "h_tiled_dsts[%lu]=%d\n", i, h_tiled_dsts[i]);
  // }

  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_keys_tmp, d_keys_out, d_values_tmp, d_values_out, d_num_runs_out, cub::Max(), num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
    d_keys_tmp, d_keys_out, d_values_tmp, d_values_out, d_num_runs_out, cub::Max(), num_items);

  int block_ev = 1 + (num_DV * num_PE) / THREAD;
  assert(THREAD * block_ev > num_DV * num_PE);
  __transpose<<<block_ev, THREAD>>>(d_FMax, d_values_out, num_PE, num_DV);

  // // !! CNull hardcoded to 0
  __tileMax<<<block_ev, THREAD>>>(d_FMax, d_VRmax, num_PE, num_DV * num_PE);
}

