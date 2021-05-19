#pragma once

typedef uint64_t Int;
typedef double Real;

struct cuda_timer_t {
  float time;

  cuda_timer_t() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~cuda_timer_t() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_); }
  
  float stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&time, start_, stop_);

    return microseconds();
  }
  
  float microseconds() { return (long long)(1000 * time); }

 private:
  cudaEvent_t start_, stop_;
};

__device__ static double atomicMax(double* address, double value) {
  unsigned long long* addr_as_longlong = reinterpret_cast<unsigned long long*>(address);
  unsigned long long  old              = *addr_as_longlong;
  unsigned long long expected;
  
  do {
    expected = old;
    old = ::atomicCAS(
      addr_as_longlong,
      expected,
      __double_as_longlong(
        ::fmax(
          value, 
          __longlong_as_double(expected)
        )
      )
    );
  } while (expected != old);
  
  return __longlong_as_double(old);
}