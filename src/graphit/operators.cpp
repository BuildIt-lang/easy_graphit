#include "graphit/operators.h"
#include "pipeline/extract_cuda.h"
#include "graphit/runtime.h"

namespace graphit {
void vertexset_apply(dyn_var<VertexSubset> &set, vertexset_apply_udf_t udf) {
	int CTA_SIZE = SimpleGPUSchedule::default_cta_size;
	int MAX_CTA = SimpleGPUSchedule::default_max_cta;

	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
			for (dyn_var<int> vid = tid; vid < set.num_elems; vid = vid + CTA_SIZE * MAX_CTA) {
				dyn_var<int*> d_queue = set.d_sparse_queue;
				Vertex var(d_queue[vid]);
				var.current_access = Vertex::access_type::INDEPENDENT;
				udf(var);	
			}
		}
	}
	current_context = context_type::HOST;
}
void vertexset_apply(dyn_var<GraphT> &edges, vertexset_apply_udf_t udf) {
	int CTA_SIZE = SimpleGPUSchedule::default_cta_size;
	int MAX_CTA = SimpleGPUSchedule::default_max_cta;

	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
			for (dyn_var<int> vid = tid; vid < edges.num_vertices; vid = vid + CTA_SIZE * MAX_CTA) {
				Vertex var(vid);
				var.current_access = Vertex::access_type::INDEPENDENT;
				udf(var);
			}
		}
	}
	current_context = context_type::HOST;
}

static bool true_func(dyn_var<int> vid) {
	return true;
}

template <typename FT, typename ET, typename SFT, typename DFT>
void vertex_based_load_balance(dyn_var<GraphT> graph, FT f, dyn_var<int> count,
		ET enumerator, SFT src_filter, DFT dst_filter, SimpleGPUSchedule* schedule) {
	
	int CTA_SIZE = schedule->cta_size;
	int MAX_CTA = schedule->max_cta;


	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
			for (dyn_var<int> vid = tid; vid < count; vid = vid + MAX_CTA * CTA_SIZE) {
				dyn_var<int> src = enumerator(vid);
				if (!src_filter(src))
					continue;
				Vertex src_var(src);
				src_var.current_access = Vertex::access_type::INDEPENDENT;
				for (dyn_var<int> eid = graph.d_src_offsets[src]; eid < graph.d_src_offsets[src+1]; 
						eid = eid + 1) {
					dyn_var<int> dst = graph.d_edge_dst[eid];
					if (!dst_filter(dst))
						continue;
					Vertex dst_var(dst);
					dst_var.current_access = Vertex::access_type::SHARED;
					f(src_var, dst_var);
				}
			}
		}
	}
	current_context = context_type::HOST;

}

template <typename FT, typename ET, typename SFT, typename DFT>
void edge_only_load_balance_blocked(dyn_var<GraphT> graph, FT f, dyn_var<int> count,
		ET enumerator, SFT src_filter, DFT dst_filter, SimpleGPUSchedule* schedule) {
	
	int CTA_SIZE = schedule->cta_size;
	int MAX_CTA = schedule->max_cta;


	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
			
			for (dyn_var<int> index = 0; index < graph.num_buckets; index = index + 1) {
				dyn_var<int> starting_edge;
				if (index == 0) 
					starting_edge = 0;
				else
					starting_edge = graph.d_bucket_sizes[index - 1];
				dyn_var<int> ending_edge = graph.d_bucket_sizes[index];

				for (dyn_var<int> eid = tid + starting_edge; eid < ending_edge; eid = eid + MAX_CTA * CTA_SIZE) {
					dyn_var<int> src = graph.d_edge_src[eid];
					dyn_var<int> dst = graph.d_edge_dst[eid];
					if (src_filter(src) && dst_filter(dst)) {
						Vertex src_var(src);
						Vertex dst_var(dst);
						src_var.current_access = Vertex::access_type::SHARED;
						dst_var.current_access = Vertex::access_type::INDEPENDENT;
						f(src_var, dst_var);	
					}
				}
				
				runtime::sync_threads();
			}

		}
	}
	current_context = context_type::HOST;

}


template <typename FT, typename ET, typename SFT, typename DFT>
void edge_only_load_balance(dyn_var<GraphT> graph, FT f, dyn_var<int> count,
		ET enumerator, SFT src_filter, DFT dst_filter, SimpleGPUSchedule* schedule) {

	int CTA_SIZE = schedule->cta_size;
	int MAX_CTA = schedule->max_cta;
	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
			for (dyn_var<int> eid = tid; eid < graph.num_edges; eid = eid + MAX_CTA * CTA_SIZE) {
				dyn_var<int> src = graph.d_edge_src[eid];
				dyn_var<int> dst = graph.d_edge_dst[eid];
				if (src_filter(src) && dst_filter(dst)) {
					Vertex src_var(src);
					Vertex dst_var(dst);
					src_var.current_access = Vertex::access_type::SHARED;
					dst_var.current_access = Vertex::access_type::INDEPENDENT;
					f(src_var, dst_var);	
				}
			}
		}
	}	
	current_context = context_type::HOST;
}


void edgeset_apply::apply(dyn_var<GraphT> &graph, edgeset_apply_udf_t udf) {
	dyn_var<int> count;	
	std::function<dyn_var<int>(dyn_var<int>)> enumerator;
	std::function<bool(dyn_var<int>)> src_filter;
	std::function<bool(dyn_var<int>)> dst_filter;
	dyn_var<GraphT*> graph_to_use = &graph;
	
	std::function<void(Vertex, Vertex)> udf_apply;
	
	// Assume that the schedule is simple schedule
	assert(dynamic_cast<SimpleGPUSchedule*>(current_schedule) != nullptr && "Currently only simple schedules are "
		"supported");

	SimpleGPUSchedule* simple_schedule = dynamic_cast<SimpleGPUSchedule*>(current_schedule);
	
	
	
	if (simple_schedule->direction == SimpleGPUSchedule::direction_type::PUSH) {
		if (from_set == nullptr) {
			count = graph.num_vertices;	
			enumerator = [] (dyn_var<int> a) {
				return a;
			};	
			
		} else {
			count = (*from_set).num_elems;
			enumerator = [=] (dyn_var<int> a) {
				return (*from_set).d_sparse_queue[a];
			};
			graphit::runtime::to_sparse(*from_set);
		}
		src_filter = true_func;
		if (to_filter) {
			dst_filter = [=] (dyn_var<int> v) {
				return (bool) to_filter((Vertex)v);
			};
		} else 
			dst_filter = true_func;
		
		udf_apply = udf;
	} else {
		if (from_set == nullptr) {
			count = graph.num_vertices;
			enumerator = [] (dyn_var<int> a) {
				return a;
			};
			dst_filter = [=] (dyn_var<int> a) -> bool {
				return true;
			};
		} else {
			if (simple_schedule->pull_frontier_rep == SimpleGPUSchedule::pull_frontier_rep_type::BITMAP) {
				graphit::runtime::to_bitmap(*from_set);
				dst_filter = [=] (dyn_var<int> a) -> bool {
					return (bool) graphit::runtime::checkBit(from_set, a);
				};
			} else {
				graphit::runtime::to_boolmap(*from_set);
				dst_filter = [=] (dyn_var<int> a) -> bool {
					return (bool) (*from_set).d_boolmap[a];
				};
			}

			count = graph.num_vertices;	
			enumerator = [] (dyn_var<int> a) {
				return a;
			};
		}
		src_filter = true_func;
		graph_to_use = graph_to_use[0].get_transposed_graph();
		udf_apply = [=] (Vertex src, Vertex dst) {
			udf(dst, src);
		};
	}

	if (simple_schedule->load_balancing == SimpleGPUSchedule::load_balancing_type::EDGE_ONLY 
		&& simple_schedule->edge_blocking == SimpleGPUSchedule::edge_blocking_type::BLOCKED) {
		assert(simple_schedule->block_size != -1 && "Invalid block size value");	
		graph_to_use = graph_to_use[0].get_blocked_graph(simple_schedule->block_size);
	}
	
	

	switch(simple_schedule->load_balancing) {	
		case SimpleGPUSchedule::load_balancing_type::VERTEX_BASED: vertex_based_load_balance(graph_to_use[0], udf_apply, count, enumerator, src_filter, dst_filter, simple_schedule); break;
		
		case SimpleGPUSchedule::load_balancing_type::EDGE_ONLY: 
			if (simple_schedule->edge_blocking == SimpleGPUSchedule::edge_blocking_type::UNBLOCKED)
				edge_only_load_balance(graph_to_use[0], udf_apply, count, enumerator, src_filter, dst_filter, simple_schedule); 
			else 
				edge_only_load_balance_blocked(graph_to_use[0], udf_apply, count, enumerator, src_filter, dst_filter, simple_schedule); 
			break;
				
		
		default:
			assert(false && "Load balancing strategy currently not supported");
	}

}


}
