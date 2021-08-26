#ifndef GRAPHIT_OPERATORS_H

#define GRAPHIT_OPERATORS_H

#include "graphit/graphit_types.h"
#include "graphit/schedule.h"


namespace graphit {

// Constructors for types


// The basic vertices.apply operator
typedef void (*vertexset_apply_udf_t) (Vertex);
void vertexset_apply(dyn_var<VertexSubset> &set, vertexset_apply_udf_t);
void vertexset_apply(dyn_var<GraphT> &edges, vertexset_apply_udf_t);




// Basic edgeset.apply operator
typedef void (*edgeset_apply_udf_t) (Vertex, Vertex);

struct edgeset_apply {
	// Members
	SimpleGPUSchedule _default_schedule;
	Schedule* current_schedule;
	dyn_var<VertexSubset> * from_set;
	std::function<dyn_var<int>(Vertex)> to_filter;	

	// Constructors with and without user specified schedule
	edgeset_apply() {
		current_schedule = &_default_schedule;	
		from_set = nullptr;
	}
	edgeset_apply(Schedule& s) {
		current_schedule = &s;
		from_set = nullptr;
	}

	// Chaining functions
	edgeset_apply& from(dyn_var<VertexSubset> &f) {
		from_set = f.addr();
		return *this;
	}
	edgeset_apply& to(std::function<dyn_var<int>(Vertex)> f) {
		to_filter = f;
		return *this;
	}

	// Apply functions
	void apply(dyn_var<GraphT> &graph, edgeset_apply_udf_t udf);

	template <typename T>
	void apply_modified(dyn_var<GraphT> &graph, dyn_var<VertexSubset> &to, VertexData<T> &tracking_var, 
			edgeset_apply_udf_t udf) {
		tracking_var.is_tracked = true;
		tracking_var.output_queue = to.addr();
		apply(graph, udf);
		tracking_var.is_tracked = false;
	}
	
};




}

#endif
