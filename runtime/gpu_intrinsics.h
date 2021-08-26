#ifndef _GPU_INTRINSICS_H
#define _GPU_INTRINSICS_H
#include "infra_gpu/graph.h"
#include <sys/time.h>
namespace graphit_runtime {


template <typename T>
static cudaError_t cudaMalloc(T** devPtr, size_t size) {
	return ::cudaMalloc((void**)devPtr, size);
}

template <typename T>
void cudaMemcpyHostToDevice(T* dst, T* src, int size) {
	cudaMemcpy(dst, src, size, ::cudaMemcpyHostToDevice);
}
template <typename T>
void cudaMemcpyDeviceToHost(T* dst, T* src, int size) {
	cudaMemcpy(dst, src, size, ::cudaMemcpyDeviceToHost);
}

template <typename T>
static bool __device__ writeSum(T* dst, T src) {
	atomicAdd(dst, src);
	return true;
}
template <typename T>
static bool __device__ CAS(T* dst, T old_val, const T& new_val) {
	if (*dst != old_val)
		return false;
	return old_val == atomicCAS(dst, old_val, new_val);
}

static void inline __device__ sync_threads(void) {
	__syncthreads();
}


static struct timeval start_time_;
static struct timeval elapsed_time_;

static void start_timer(void) {
	gettimeofday(&start_time_, NULL);
}
static float stop_timer(void) {
	struct timeval stop_time_;
	gettimeofday(&stop_time_, NULL);
	timersub(&stop_time_, &start_time_, &elapsed_time_);
	return elapsed_time_.tv_sec + elapsed_time_.tv_usec/1e6f;
}
static void print_time(float x) {
	std::cout << x << std::endl;
}
}

#endif
