#include <iostream>
#include <assert.h>
#include "main.h"
#include <cub/cub.cuh>

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

__device__ void d_norm_2(int num_attr, double * vec1, double * vec2, double *out, int k) {
  double sum = 0.0;
  for (int i = 0; i < num_attr; i ++) {
    sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
  }
  out[k] = sqrt(sum);
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

void d_VFmax_VRmax(Graph * d_Data_Graph, Graph * d_Pattern_Graph,
                 double * d_VF, double * d_VR, double * d_VFmax, double * d_VRmax) {

  uint64_t num_rows = d_Data_Graph->num_vertices; // DV
  uint64_t num_cols = d_Pattern_Graph->num_edges; // PE

  uint64_t thread = 1024;
  uint64_t block  = 1 + (num_rows * num_cols) / thread;
  assert(thread * block > num_rows * num_cols);

  double *d_VFt;
  cudaMalloc((void**)&d_VFt, num_rows * num_cols * sizeof(double));
  __transpose<<<block, thread>>>(d_VFt, d_VF, num_rows, num_cols);
  d_rowmax(d_VFmax, d_VFt, num_cols, num_rows);

  double *d_VRt;
  cudaMalloc((void**)&d_VRt, num_rows * num_cols * sizeof(double));
  __transpose<<<block, thread>>>(d_VRt, d_VR, num_rows, num_cols);
  d_rowmax(d_VRmax, d_VRt, num_cols, num_rows);

  cudaFree(d_VFt);
  cudaFree(d_VRt);
  cudaDeviceSynchronize();
}



void d_NormProb(const uint64_t num_rows, const uint64_t num_cols, double *d_x) {

  // --------------------------
  // Prep

  uint64_t thread = 1024;
  uint64_t block  = 1 + (num_rows * num_cols) / thread;
  assert(thread * block > num_rows * num_cols);

  double *d_storage;
  double h_storage[num_cols];
  cudaMalloc((void**)&d_storage, num_cols * sizeof(double));

  // ----------------------------
  // Compute column max

  double *d_xt;
  cudaMalloc((void**)&d_xt, num_rows * num_cols * sizeof(double));
  __transpose<<<block, thread>>>(d_xt, d_x, num_rows, num_cols);

  d_rowmax(d_storage, d_xt, num_cols, num_rows);
  cudaMemcpy(h_storage, d_storage, num_cols * sizeof(double), cudaMemcpyDeviceToHost);

  // --------------------------------
  // Subtract max from columns

  __rowSubExp<<<block, thread>>>(d_xt, num_cols, num_rows, d_storage);

  // --------------------------------
  // Sum columns

  cudaMemset(d_storage, 0, num_cols * sizeof(double));
  d_rowsum(d_storage, d_xt, num_cols, num_rows);

  // ---------------------------------
  // Subtract log-sum from columns

  __rowSubLog<<<block, thread>>>(d_xt, num_cols, num_rows, d_storage);

  // ---------------------------------
  // Transpose back to original shape

  __transpose<<<block, thread>>>(d_x, d_xt, num_cols, num_rows);

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

  int thread = 1024;
  int block  = 1 + (num_DV * num_PE) / thread;
  assert(thread * block > num_DV * num_PE);

  __d_Init_VR_VF<<<block, thread>>>(
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
  d_norm_2(num_AT - 2, PAttr + j * num_AT + 2, DAttr + i * num_AT + 2, CE, k);
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

  uint64_t thread = 1024;
  uint64_t block  = 1 + (num_DE * num_PE) / thread;
  assert(thread * block > num_DE * num_PE);

  __d_Init_CE_RE_FE<<<block, thread>>>(
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

  uint64_t i = k / num_PE;
  uint64_t j = k % num_PE;
  uint64_t src = PAttr[j * num_AT];         // src id
  uint64_t dst = PAttr[j * num_AT + 1];     // dst id
  VF[k] = MU[i * num_PV + dst] - FMax[k];
  VR[k] = MU[i * num_PV + src] - RMax[k];
}

void d_VF_VR(Graph * Data_Graph, Graph * Pattern_Graph,
           double * MU, double * FMax, double * RMax, double * VF, double * VR) {

  uint64_t num_DV  = Data_Graph->num_vertices;
  uint64_t num_PV  = Pattern_Graph->num_vertices;
  uint64_t num_PE  = Pattern_Graph->num_edges;
  uint64_t num_AT  = Pattern_Graph->Etable.num_cols;
  uint64_t * PAttr = (uint64_t *) Pattern_Graph->Etable.table;

  uint64_t thread = 1024;
  uint64_t block  = 1 + (num_DV * num_PE) / thread;
  assert(thread * block > num_DV * num_PE);

  __d_VF_VR<<<block, thread>>>(
    num_AT,
    num_PE,
    num_PV,
    PAttr,
    MU,
    VR,
    VF,
    FMax,
    RMax
  );
  cudaDeviceSynchronize();
}
