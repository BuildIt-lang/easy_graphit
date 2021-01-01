#include "graphit/operators.h"
#include "pipeline/extract_cuda.h"

namespace graphit {
void vertexset_apply(dyn_var<VertexSubset> &set, vertexset_apply_udf_t udf) {
	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < 60; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < 512; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * 512 + thread_id;
			for (dyn_var<int> vid = tid; vid < set.num_elems; vid = vid + 60 * 512) {
				dyn_var<int*> d_queue = set.d_sparse_queue;
				Vertex var(d_queue[vid]);
				var.current_access = Vertex::access_type::INDEPENDENT;
				udf(var);	
			}
		}
	}
	current_context = context_type::HOST;
}

template <typename FT>
void vertex_based_load_balance(dyn_var<GraphT> &graph, dyn_var<VertexSubset> &set, FT f) {
	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < 60; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < 512; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * 512 + thread_id;
			for (dyn_var<int> vid = tid; vid < set.num_elems; vid = vid + 60 * 512) {
				dyn_var<int> src = set.d_sparse_queue[vid];
				Vertex src_var(src);
				src_var.current_access = Vertex::access_type::INDEPENDENT;
				for (dyn_var<int> eid = graph.d_row_offsets[src]; eid < graph.d_row_offsets[src+1]; eid = eid + 1) {
					Vertex dst_var(graph.d_edges_dst[eid]);
					dst_var.current_access = Vertex::access_type::SHARED;
					f(src_var, dst_var);
				}
			}
		}
	}
	current_context = context_type::HOST;
	
}

void edgeset_apply_from(dyn_var<GraphT> &graph, dyn_var<VertexSubset> &set, edgeset_apply_udf_t udf) {
	vertex_based_load_balance(graph, set, [=] (Vertex src_var, Vertex dst_var) {udf(src_var, dst_var);});
}


}
