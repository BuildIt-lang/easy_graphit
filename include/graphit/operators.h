#ifndef GRAPHIT_OPERATORS_H

#define GRAPHIT_OPERATORS_H

#include "graphit/graphit_types.h"

namespace graphit {

// Constructors for types


// The basic vertices.apply operator
typedef void (*vertexset_apply_udf_t) (Vertex);
void vertexset_apply(dyn_var<VertexSubset> &set, vertexset_apply_udf_t);


// Basic edgeset.apply operator
typedef void (*edgeset_apply_udf_t) (Vertex, Vertex);

void edgeset_apply_from(dyn_var<GraphT> &graph, dyn_var<VertexSubset> &set, edgeset_apply_udf_t udf);

template <typename T>
void edgeset_apply_from_modified(dyn_var<GraphT> &graph, dyn_var<VertexSubset> &set, edgeset_apply_udf_t udf, dyn_var<VertexSubset> &to, VertexData<T> &tracking_var) {

	tracking_var.is_tracked = true;
	tracking_var.output_queue = to.addr();

	edgeset_apply_from(graph, set, udf);

	tracking_var.is_tracked = false;
}

}

#endif
