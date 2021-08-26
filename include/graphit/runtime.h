#ifndef GRAPHIT_RUNTIME_H
#define GRAPHIT_RUNTIME_H
#include "graphit/graphit_types.h"
namespace graphit {
namespace runtime {

extern dyn_var<VertexSubset (int, int)>  new_vertex_subset;
extern dyn_var<GraphT (char*)> load_graph;
extern dyn_var<void (VertexSubset, int)> enqueue_sparse;
extern dyn_var<void (VertexSubset, int)> enqueue_bitmap;
extern dyn_var<void (VertexSubset, int)> enqueue_boomap;
extern dyn_var<void (VertexSubset)> to_sparse;
extern dyn_var<void (VertexSubset)> to_bitmap;
extern dyn_var<void (VertexSubset)> to_boolmap;
extern dyn_var<int (VertexSubset, int)> checkBit;
extern dyn_var<void (void*, int)> cudaMalloc;
extern dyn_var<void (void)> sync_threads;


extern dyn_var<void (void)> start_timer;
extern dyn_var<float (void)> stop_timer;
extern dyn_var<void (float)> print_time;

}
}

#endif
