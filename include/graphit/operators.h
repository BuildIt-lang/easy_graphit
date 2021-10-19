#ifndef GRAPHIT_OPERATORS_H
#define GRAPHIT_OPERATORS_H

#include "graphit/graphit_types.h"
#include "graphit/schedule.h"
#include <vector>
#include <string>


namespace graphit {

extern dyn_var<int>* cta_id_ptr;
extern dyn_var<int>* thread_id_ptr;


// Constructors for types


// The basic vertices.apply operator
typedef void (*vertexset_apply_udf_t) (Vertex);
void vertexset_apply(dyn_var<VertexSubset> &set, vertexset_apply_udf_t);
void vertexset_apply(dyn_var<GraphT> &edges, vertexset_apply_udf_t);




// Basic edgeset.apply operator
//typedef void (*edgeset_apply_udf_t) (Vertex, Vertex);
//typedef void (*edgeset_apply_udf_w_t) (Vertex, Vertex, dyn_var<int> w);
typedef std::function<void(Vertex, Vertex)> edgeset_apply_udf_t;
typedef std::function<void(Vertex, Vertex, dyn_var<int>)> edgeset_apply_udf_w_t;

template <typename T>
static void set_shared(const dyn_var<T>& v) {
	v.block_var->template setMetadata<std::vector<std::string>>("attributes", {"__shared__"});
}

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
	void apply(dyn_var<GraphT> &graph, edgeset_apply_udf_w_t udf);
	void apply(dyn_var<GraphT> &graph, edgeset_apply_udf_t udf) {
		apply(graph, [=](Vertex src, Vertex dst, dyn_var<int> w){udf(src, dst);});
	}


	void apply_priority(dyn_var<GraphT> &graph, edgeset_apply_udf_w_t udf, PriorityQueue &pq) {
		// We don't need hybrid schedules for now
		assert(dynamic_cast<SimpleGPUSchedule*>(current_schedule) != nullptr && "At this point only simple schedules are "
			"supported");

		SimpleGPUSchedule* simple_schedule = s_to<SimpleGPUSchedule>(current_schedule);
		pq.current_schedule = simple_schedule;
		apply(graph, udf);	

		int on_device = current_context == context_type::DEVICE;	
		// After all tracking is done, swap the buffers
		dyn_var<VertexSubset> to = pq.pq.frontier_;
		if (simple_schedule->frontier_creation == SimpleGPUSchedule::frontier_creation_type::SPARSE) {
			if (on_device)
				to.swap_queues_device();	
			else
				to.swap_queues_host();
		} else if (simple_schedule->frontier_creation == SimpleGPUSchedule::frontier_creation_type::BITMAP) {
			if (on_device)
				to.swap_bitmaps_device();	
			else
				to.swap_bitmaps_host();
		} else {	
			if(on_device)
				to.swap_boolmaps_device();
			else
				to.swap_boolmaps_host();
		}
		pq.pq.frontier_ = to;
		
	}

	template <typename T>
	void apply_modified(dyn_var<GraphT> &graph, dyn_var<VertexSubset> &to, VertexData<T> &tracking_var, 
			edgeset_apply_udf_t udf, bool allow_dupes = true) {
		apply_modified(graph, to, tracking_var, [=](Vertex src, Vertex dst, dyn_var<int> w) {udf(src, dst);}, 
		allow_dupes);
	}
	template <typename T>
	void apply_modified(dyn_var<GraphT> &graph, dyn_var<VertexSubset> &to, VertexData<T> &tracking_var, 
			edgeset_apply_udf_w_t udf, bool allow_dupes = true) {
		
		// If we have a hybrid schedule on our hand, we need to recusrively call two separate versions
		if (s_isa<HybridGPUSchedule>(current_schedule)) {
			// We will create two edgeset_apply objects that are identical
			edgeset_apply e1, e2;
			e1.current_schedule = s_to<HybridGPUSchedule>(current_schedule)->s1;
			e2.current_schedule = s_to<HybridGPUSchedule>(current_schedule)->s2;
			
			e1.from_set = e2.from_set = from_set;
			e1.to_filter = e2.to_filter = to_filter;
			
			// The actual dynamic condition
			if ((*from_set).size() <= *(s_to<HybridGPUSchedule>(current_schedule)->threshold) * graph.num_vertices) {
				e1.apply_modified(graph, to, tracking_var, udf, allow_dupes);
			} else {
				e2.apply_modified(graph, to, tracking_var, udf, allow_dupes);
			}
			return;	
		}

		assert(dynamic_cast<SimpleGPUSchedule*>(current_schedule) != nullptr && "At this point only simple schedules are "
			"supported");

		SimpleGPUSchedule* simple_schedule = s_to<SimpleGPUSchedule>(current_schedule);

		tracking_var.is_tracked = true;
		tracking_var.output_queue = to.addr();
		tracking_var.frontier_creation= simple_schedule->frontier_creation;
		if (current_context == context_type::DEVICE)
			tracking_var.allow_dupes = allow_dupes;
		else
			tracking_var.allow_dupes = true;

		if (!tracking_var.allow_dupes) {
			to.curr_dedup_counter = to.curr_dedup_counter + 1;
		}
		apply(graph, udf);
		tracking_var.is_tracked = false;
	
		int on_device = current_context == context_type::DEVICE;	
		// After all tracking is done, swap the buffers
		if (simple_schedule->frontier_creation == SimpleGPUSchedule::frontier_creation_type::SPARSE) {
			if (on_device)
				to.swap_queues_device();	
			else
				to.swap_queues_host();
		} else if (simple_schedule->frontier_creation == SimpleGPUSchedule::frontier_creation_type::BITMAP) {
			if (on_device)
				to.swap_bitmaps_device();	
			else
				to.swap_bitmaps_host();
		} else {	
			if(on_device)
				to.swap_boolmaps_device();
			else
				to.swap_boolmaps_host();
		}
		if (!on_device && !allow_dupes && simple_schedule->frontier_creation == SimpleGPUSchedule::frontier_creation_type::SPARSE)
			runtime::dedup_frontier_perfect(to);
	}
	
};

// Kernel Fusion related implementations
void fuse_kernel(bool to_fuse, std::function<void (void)> body);


}

#endif
