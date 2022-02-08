#ifndef GRAPHIT_RUNTIME_H
#define GRAPHIT_RUNTIME_H
#include "graphit/graphit_types.h"
namespace graphit {
namespace runtime {

extern dyn_var<VertexSubset::super_name (int, int)>  new_vertex_subset;
extern dyn_var<GraphT::super_name (char*)> load_graph;
extern dyn_var<void (VertexSubset::super_name, int)> enqueue_sparse;
extern dyn_var<void (VertexSubset::super_name, int)> enqueue_sparse_no_dupes;
extern dyn_var<void (VertexSubset::super_name, int)> enqueue_bitmap;
extern dyn_var<void (VertexSubset::super_name, int)> enqueue_boomap;

extern dyn_var<void (VertexSubset::super_name)> to_sparse_host;
extern dyn_var<void (VertexSubset::super_name)> to_sparse_device;

extern dyn_var<void (VertexSubset::super_name)> to_bitmap;
extern dyn_var<void (VertexSubset::super_name)> to_boolmap;
extern dyn_var<int (VertexSubset::super_name, int)> checkBit;
extern dyn_var<void (void*, int)> cudaMalloc;
extern dyn_var<void (void)> sync_threads;
extern dyn_var<void (void)> sync_grid;
extern dyn_var<void (void)> atomicAggInc;
extern dyn_var<void (void)> atomicSub;
extern dyn_var<void (void)> shfl_sync;
extern dyn_var<void (void)> shfl_up_sync;
extern dyn_var<void (void)> shfl_down_sync;
extern dyn_var<void (void)> binary_search_upperbound;


extern dyn_var<void (void)> start_timer;
extern dyn_var<float (void)> stop_timer;
extern dyn_var<void (float)> print_time;


extern dyn_var<void (void)> new_frontier_list;
extern dyn_var<void (void)> dedup_frontier_perfect;
}
}

#endif
