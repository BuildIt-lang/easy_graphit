#include "graphit/runtime.h"

namespace graphit {
namespace runtime {

dyn_var<VertexSubset (int, int)>  new_vertex_subset ("graphit_runtime::new_vertex_subset");
dyn_var<GraphT (char*)> load_graph ("graphit_runtime::load_graph");
dyn_var<void (VertexSubset, int)> enqueue_sparse ("graphit_runtime::enqueue_sparse");

dyn_var<void (void*, void*, int)> copyHostToDevice ("graphit_runtime::copyHostToDevice");
dyn_var<void (void*, void*, int)> copyDeviceToHost ("graphit_runtime::copyDeviceToHost");
dyn_var<int (void*, int)> writeMin ("graphit_runtime::writeMin");

}
}
