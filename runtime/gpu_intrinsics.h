#ifndef _GPU_INTRINSICS_H
#define _GPU_INTRINSICS_H
#include "infra_gpu/graph.h"
#include "infra_gpu/vertex_frontier.h"
#include "infra_gpu/vertex_representation.h"
#include "infra_gpu/gpu_priority_queue.h"
#include "infra_gpu/list.h"

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
static T __device__ atomicSub(T* dst, T val) {
	return ::atomicSub(dst, val);
}

static void inline __device__ sync_threads(void) {
	__syncthreads();
}
static void inline __device__ sync_grid(void) {
	this_grid().sync();
}
static int inline __device__ shfl_sync(int mask, int val, int src_line, int width) {
	return __shfl_sync((unsigned)mask, val, src_line, width);	
}
static int inline __device__ shfl_up_sync(int mask, int val, int src_line) {
	return __shfl_up_sync((unsigned)mask, val, src_line);	
}
static int inline __device__ shfl_down_sync(int mask, int val, int src_line) {
	return __shfl_down_sync((unsigned)mask, val, src_line);	
}
static int32_t __device__ binary_search_upperbound(int32_t *array, int32_t len, int32_t key){
        int32_t s = 0;
        while(len>0){
                int32_t half = len>>1;
                int32_t mid = s + half;
                if(array[mid] > key){
                        len = half;
                }else{
                        s = mid+1;
                        len = len-half-1;
                }
        }
        return s;
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

static void __device__ enqueue_sparse(VertexFrontier &f, int vertex) {
	enqueueVertexSparseQueue(f.d_sparse_queue_output, f.d_num_elems_output, vertex);
}
static void __device__ enqueue_sparse_no_dupes(VertexFrontier &f, int vertex) {
	enqueueVertexSparseQueueDedupPerfect(f.d_sparse_queue_output, f.d_num_elems_output, vertex, f);
}
static void __device__ enqueue_bitmap(VertexFrontier &f, int vertex) {
	enqueueVertexBitmap(f.d_bit_map_output, f.d_num_elems_output, vertex);
}
static void __device__ enqueue_boolmap(VertexFrontier &f, int vertex) {
	enqueueVertexBytemap(f.d_byte_map_output, f.d_num_elems_output, vertex);
}


static vertexsubset new_vertex_subset(int num) {
	return create_new_vertex_set(num, 0);
}

static void to_sparse_host(vertexsubset& s) {
	vertex_set_prepare_sparse(s);
	s.format_ready = VertexFrontier::SPARSE;
}
static void __device__ to_sparse_device(vertexsubset& s) {
	vertex_set_prepare_sparse_device(s);
	s.format_ready = VertexFrontier::SPARSE;
}
static void to_bitmap(vertexsubset& s) {
	vertex_set_prepare_bitmap(s);
	s.format_ready = VertexFrontier::BITMAP;
}

template <typename...Args>
static void LaunchCooperativeKernel(void* f, int block_size, int cta_size, const Args...args) {
	void* a[] = {((void*)&args)...};
	cudaLaunchCooperativeKernel(f, block_size, cta_size, a);	
}


template <typename T, typename T2>
void cudaMemcpyFromSymbolMagic(T* dst, const T2 & symbol) {
	cudaMemcpyFromSymbol((char*)dst, symbol, sizeof(T), 0);
}
template <typename T>
void __device__ cudaMemcpyToSymbolMagic(char symbol[], const T& src) {
	//cudaMemcpyFromSymbol(dst, symbol, sizeof(T), 0);
	memcpy(symbol, &src, sizeof(T));
}

}

#endif
