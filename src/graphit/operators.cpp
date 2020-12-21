#include "graphit/operators.h"
#include "pipeline/extract_cuda.h"

namespace graphit {
void vertexset_apply(dyn_var<VertexSubset> set, vertexset_apply_udf_t udf) {

	dyn_var<int> total = set.num_elems;	
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < 60; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < 512; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * 512 + thread_id;
			for (dyn_var<int> vid = tid; vid < total; vid = vid + 60 * 512) {
				dyn_var<int*> d_queue = set.d_sparse_queue;
				udf(d_queue[vid]);	
			}
		}
	}
}

}
