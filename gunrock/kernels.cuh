#ifndef __KERNELS
#define __KERNELS 1

void device2host(WorkArrays&, WorkArrays&, uint64_t, uint64_t, uint64_t, uint64_t);
void host2device(WorkArrays&, WorkArrays&, uint64_t, uint64_t, uint64_t, uint64_t);

void d_rowmax(double*, double*, uint64_t, uint64_t);
void d_rowsum(double*, double*, uint64_t, uint64_t);

void d_VFmax_VRmax(Graph*, Graph*,double*, double*, double*, double*);
void d_NormProb(uint64_t, uint64_t, double*);
void d_Init_VR_VF(Graph*, Graph*, double*, double*, double*);
void d_Init_CE_RE_FE(Graph*, Graph*, double*, double*, double*);
void d_VF_VR(Graph*, Graph*, double*, double*, double*, double*, double*);
void d_UpdateMU(Graph*, Graph*, double*, double*, double*, double*);
void d_FE_RE(Graph *, Graph *, double *, double *, double *, double *, double *);
void d_FMax(Graph *, Graph *, double *, double *, double *, double *);
void d_RMax(Graph *, Graph *, double *, double *, double *, double *);

#endif