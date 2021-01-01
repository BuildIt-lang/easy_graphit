#ifndef GRAPHIT_RUNTIME_H
#define GRAPHIT_RUNTIME_H
#include "graphit/graphit_types.h"
namespace graphit {
namespace runtime {

extern dyn_var<VertexSubset (int, int)>  new_vertex_subset;
extern dyn_var<GraphT (char*)> load_graph;
extern dyn_var<void (VertexSubset, int)> enqueue_sparse;


}
}

#endif
