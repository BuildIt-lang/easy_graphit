#include "graphit/runtime.h"

namespace graphit {
namespace runtime {

dyn_var<VertexSubset (int, int)>  new_vertex_subset ("graphit_runtime::new_vertex_subset");
dyn_var<GraphT (char*)> load_graph ("graphit_runtime::load_graph<int>");
dyn_var<void (VertexSubset, int)> enqueue_sparse ("graphit_runtime::enqueue_sparse");
dyn_var<void (VertexSubset, int)> enqueue_sparse_no_dupes ("graphit_runtime::enqueue_sparse_no_dupes");
dyn_var<void (VertexSubset, int)> enqueue_bitmap ("graphit_runtime::enqueue_bitmap");
dyn_var<void (VertexSubset, int)> enqueue_boolmap ("graphit_runtime::enqueue_boolmap");

dyn_var<void (void*, void*, int)> copyHostToDevice ("graphit_runtime::cudaMemcpyHostToDevice");
dyn_var<void (void*, void*, int)> copyDeviceToHost ("graphit_runtime::cudaMemcpyDeviceToHost");
dyn_var<int (void*, int)> writeMin ("graphit_runtime::writeMin");
dyn_var<int (void*, int)> writeSum ("graphit_runtime::writeSum");

dyn_var<void (VertexSubset)> to_sparse_host("graphit_runtime::to_sparse_host");
dyn_var<void (VertexSubset)> to_sparse_device("graphit_runtime::to_sparse_device");
dyn_var<void (VertexSubset)> to_bitmap("graphit_runtime::to_bitmap");
dyn_var<void (VertexSubset)> to_boolmap("graphit_runtime::to_boolmap");
dyn_var<int (VertexSubset, int)> checkBit("graphit_runtime::checkBit");

dyn_var<void (void*, int)> cudaMalloc("graphit_runtime::cudaMalloc");
dyn_var<void (void*, void*, int, int)> cudaMemcpyToSymbol("graphit_runtime::cudaMemcpyToSymbol");
dyn_var<void (void)> sync_threads("graphit_runtime::sync_threads");
dyn_var<void (void)> sync_grid("graphit_runtime::sync_grid");
dyn_var<void (void)> atomicAggInc("graphit_runtime::atomicAggInc");
dyn_var<void (void)> atomicSub("graphit_runtime::atomicSub");
dyn_var<void (void)> shfl_sync("graphit_runtime::shfl_sync");
dyn_var<void (void)> shfl_up_sync("graphit_runtime::shfl_up_sync");
dyn_var<void (void)> shfl_down_sync("graphit_runtime::shfl_down_sync");
dyn_var<void (void)> binary_search_upperbound("graphit_runtime::binary_search_upperbound");


dyn_var<void (void)> start_timer("graphit_runtime::start_timer");
dyn_var<float (void)> stop_timer("graphit_runtime::stop_timer");
dyn_var<void (float)> print_time("graphit_runtime::print_time");

dyn_var<void (void)> new_frontier_list("graphit_runtime::new_frontier_list");
dyn_var<void (void)> dedup_frontier_perfect("graphit_runtime::dedup_frontier_perfect");
}
}
