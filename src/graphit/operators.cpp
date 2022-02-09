#include "graphit/operators.h"
#include "pipeline/extract_cuda.h"
#include "graphit/runtime.h"

namespace graphit {

dyn_var<int>* cta_id_ptr = nullptr;
dyn_var<int>* thread_id_ptr = nullptr;

void vertexset_apply(VertexSubset &set, vertexset_apply_udf_t udf) {
	if (current_context == context_type::DEVICE) {	
		graphit::runtime::to_sparse_device(set);
		int CTA_SIZE = SimpleGPUSchedule::default_cta_size;
		int MAX_CTA = SimpleGPUSchedule::default_max_cta;
		graphit::runtime::sync_grid();
		dyn_var<int> tid = *cta_id_ptr * CTA_SIZE + *thread_id_ptr;
		for (dyn_var<int> vid = tid; vid < set.size(); vid = vid + CTA_SIZE * MAX_CTA) {
			dyn_var<int*> d_queue = set.d_sparse_queue_input;
			Vertex var(d_queue[vid]);
			var.current_access = Vertex::access_type::INDEPENDENT;
			udf(var);	
		}
			
		graphit::runtime::sync_grid();
		return;
	}
	assert(current_context == context_type::HOST && "Vertex set apply to be only called from host");
	
	graphit::runtime::to_sparse_host(set);
	int CTA_SIZE = SimpleGPUSchedule::default_cta_size;
	int MAX_CTA = SimpleGPUSchedule::default_max_cta;

	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
			for (dyn_var<int> vid = tid; vid < set.size(); vid = vid + CTA_SIZE * MAX_CTA) {
				dyn_var<int*> d_queue = set.d_sparse_queue_input;
				Vertex var(d_queue[vid]);
				var.current_access = Vertex::access_type::INDEPENDENT;
				udf(var);	
			}
		}
	}
	current_context = context_type::HOST;
}
void vertexset_apply(GraphT &edges, vertexset_apply_udf_t udf) {
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
void vertex_based_load_balance_impl(GraphT graph, FT f, dyn_var<int> count,
		ET enumerator, SFT src_filter, DFT dst_filter, SimpleGPUSchedule* schedule, dyn_var<int> &cta_id, 
		dyn_var<int> &thread_id, dyn_var<int> MAX_CTA, int CTA_SIZE) {

	dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
	dyn_var<int> vid = tid;
	dyn_var<int> src = enumerator(vid);

	Vertex src_var(src);
	src_var.current_access = Vertex::access_type::INDEPENDENT;
	for (dyn_var<int> eid = graph.d_src_offsets[src]; eid < graph.d_src_offsets[src+1]; 
			eid = eid + 1) {
		if (!src_filter(src)) { // We check the source filter afterwards just for the early break optimization
			break;
		}
		dyn_var<int> dst = graph.d_edge_dst[eid];
		if (!dst_filter(dst))
			continue;
		Vertex dst_var(dst);
		dst_var.current_access = Vertex::access_type::SHARED;
		f(src_var, dst_var, graph.d_edge_weight[eid]);
	}
}

template <typename FT, typename ET, typename SFT, typename DFT>
void vertex_based_load_balance(GraphT graph, FT f, dyn_var<int> count,
		ET enumerator, SFT src_filter, DFT dst_filter, SimpleGPUSchedule* schedule) {


	int CTA_SIZE = schedule->cta_size;
	dyn_var<int> num_threads = count;
	dyn_var<int> MAX_CTA = (num_threads + CTA_SIZE - 1)/CTA_SIZE;
	

	if (current_context == context_type::DEVICE) {
		for (dyn_var<int> vcta_id = *cta_id_ptr; vcta_id < MAX_CTA; 
			vcta_id = vcta_id + SimpleGPUSchedule::default_max_cta) {
			vertex_based_load_balance_impl(graph, f, count, enumerator, src_filter, dst_filter, schedule,
				vcta_id, *thread_id_ptr, MAX_CTA, CTA_SIZE);
			graphit::runtime::sync_threads();		
		}
		graphit::runtime::sync_grid();
		return;
	}


	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			vertex_based_load_balance_impl(graph, f, count, enumerator, src_filter, dst_filter, schedule, 
			cta_id, thread_id, MAX_CTA, CTA_SIZE);
		}
	}
	current_context = context_type::HOST;

}

template <typename FT, typename ET, typename SFT, typename DFT>
void edge_only_load_balance_blocked(GraphT graph, FT f, dyn_var<int> count,
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
						dst_var.current_access = Vertex::access_type::SHARED;
						f(src_var, dst_var, graph.d_edge_weight[eid]);
					}
				}
				
				runtime::sync_threads();
			}

		}
	}
	current_context = context_type::HOST;

}


template <typename FT, typename ET, typename SFT, typename DFT>
void edge_only_load_balance(GraphT graph, FT f, dyn_var<int> count,
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
					dst_var.current_access = Vertex::access_type::SHARED;
					f(src_var, dst_var, graph.d_edge_weight[eid]);
				}
			}
		}
	}	
	current_context = context_type::HOST;
}
#define STAGE_1_SIZE (8)
#define WARP_SIZE (32)
template <typename FT, typename ET, typename SFT, typename DFT>
void TWCE_load_balance_impl(GraphT graph, FT f, dyn_var<int> count, ET enumerator, SFT src_filter, 
		DFT dst_filter, SimpleGPUSchedule* schedule, dyn_var<int> &cta_id, 
		dyn_var<int> &thread_id, dyn_var<int> &MAX_CTA, int CTA_SIZE) {

	dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
	dyn_var<int> lane_id = tid % 32;
	
	dyn_var<int32_t[1024]> stage2_queue; set_shared(stage2_queue);
	dyn_var<int32_t[1024]> stage3_queue; set_shared(stage3_queue);
	dyn_var<int32_t[3]> stage_queue_sizes; set_shared(stage_queue_sizes);

	if (thread_id < 3) {
		stage_queue_sizes[thread_id] = 0;
	}	
	runtime::sync_threads();
	dyn_var<int32_t[1024]> stage2_offset; set_shared(stage2_offset);
	dyn_var<int32_t[1024]> stage3_offset; set_shared(stage3_offset);
	dyn_var<int32_t[1024]> stage2_size; set_shared(stage2_size);
	dyn_var<int32_t[1024]> stage3_size; set_shared(stage3_size);
	
	dyn_var<int32_t> total_vertices = count;
	dyn_var<int32_t> local_vertex_idx = tid / (STAGE_1_SIZE);
	dyn_var<int32_t> degree;
	dyn_var<int32_t> s1_offset;
	dyn_var<int32_t> local_vertex;
	dyn_var<int32_t> src_offset;
	if (local_vertex_idx < total_vertices) {
		local_vertex = enumerator(local_vertex_idx);
		// Step 1 aggregate vertices into shared buffers
		degree = graph.out_degrees[local_vertex];
		src_offset = graph.d_src_offsets[local_vertex];
		dyn_var<int32_t> s3_size = degree/CTA_SIZE;
		degree = degree - s3_size * CTA_SIZE;
		if (s3_size > 0) {
			if (thread_id % STAGE_1_SIZE == 0) {
				dyn_var<int32_t> pos = runtime::atomicAggInc(stage_queue_sizes + 2);
				stage3_queue[pos] = local_vertex;
				stage3_size[pos] = s3_size * CTA_SIZE;
				stage3_offset[pos] = src_offset;
			}
		}
		dyn_var<int32_t> s2_size = degree / WARP_SIZE;
		degree = degree - WARP_SIZE * s2_size;
		if (s2_size > 0) {
			if (thread_id % (STAGE_1_SIZE) == 0) {
				dyn_var<int32_t> pos = runtime::atomicAggInc(stage_queue_sizes + 1);
				stage2_queue[pos] = local_vertex;
				stage2_offset[pos] = s3_size * CTA_SIZE + src_offset;
				stage2_size[pos] = s2_size * WARP_SIZE;
				
			}
		}
		s1_offset = s3_size * CTA_SIZE + s2_size * WARP_SIZE + src_offset;
	} else
		local_vertex = -1;
	runtime::sync_threads();	
	if (local_vertex_idx < total_vertices) {
		// STAGE 1
		Vertex src_var(local_vertex);
		src_var.current_access = Vertex::access_type::SHARED;
		for (dyn_var<int32_t> neigh_id = s1_offset + (lane_id % STAGE_1_SIZE); neigh_id < degree 
			+ s1_offset; neigh_id = neigh_id + STAGE_1_SIZE) {
			if (!src_filter(local_vertex)) 
				break;
			dyn_var<int32_t> dst = graph.d_edge_dst[neigh_id];
			if (!dst_filter(dst))
				continue;
			Vertex dst_var(dst);
			dst_var.current_access = Vertex::access_type::SHARED;
			f(src_var, dst_var, graph.d_edge_weight[neigh_id]);
			//f(src_var, dst_var, 0);
		}
	}
	runtime::sync_threads();
	while(1) {
		dyn_var<int32_t> to_process;
		if (lane_id == 0) {
			to_process = runtime::atomicSub(stage_queue_sizes + 1, 1) - 1;	
		}
		to_process = runtime::shfl_sync(-1, to_process, 0, 32);
		if (to_process < 0)
			break;
		local_vertex = stage2_queue[to_process];
		degree = stage2_size[to_process];
		dyn_var<int32_t> s2_offset = stage2_offset[to_process];
		Vertex src_var(local_vertex);
		src_var.current_access = Vertex::access_type::SHARED;
		for (dyn_var<int32_t> neigh_id = s2_offset + lane_id; neigh_id < degree + s2_offset; neigh_id = 
			neigh_id + WARP_SIZE) {
			if (!src_filter(local_vertex)) 
				break;
			dyn_var<int32_t> dst = graph.d_edge_dst[neigh_id];
			if (!dst_filter(dst))
				continue;
			Vertex dst_var(dst);
			dst_var.current_access = Vertex::access_type::SHARED;
			f(src_var, dst_var, graph.d_edge_weight[neigh_id]);			
			//f(src_var, dst_var, 0);
		}
	}
	for (dyn_var<int32_t> wid = 0; wid < stage_queue_sizes[2]; wid = wid + 1) {
		local_vertex = stage3_queue[wid];
		degree = stage3_size[wid];
		dyn_var<int32_t> s3_offset = stage3_offset[wid];
		Vertex src_var(local_vertex);
		src_var.current_access = Vertex::access_type::SHARED;
		for (dyn_var<int32_t> neigh_id = s3_offset + thread_id; neigh_id < degree + s3_offset;
			neigh_id = neigh_id + CTA_SIZE) {
			if (!src_filter(local_vertex)) 
				break;
			dyn_var<int32_t> dst = graph.d_edge_dst[neigh_id];
			if (!dst_filter(dst))
				continue;
			Vertex dst_var(dst);
			dst_var.current_access = Vertex::access_type::SHARED;
			f(src_var, dst_var, graph.d_edge_weight[neigh_id]);
			//f(src_var, dst_var, 0);
		}
	}
}

template <typename FT, typename ET, typename SFT, typename DFT>
void TWCE_load_balance(GraphT graph, FT f, dyn_var<int> count, ET enumerator, SFT src_filter, 
		DFT dst_filter, SimpleGPUSchedule* schedule) {
	int CTA_SIZE = schedule->cta_size;
	dyn_var<int> num_threads = count * STAGE_1_SIZE;
	dyn_var<int> MAX_CTA = (num_threads + CTA_SIZE - 1)/CTA_SIZE;
	
	if (current_context == context_type::DEVICE) {
		assert(CTA_SIZE == SimpleGPUSchedule::default_cta_size && "Currently cannot generate fused kernels with different CTA sizes");
		graphit::runtime::sync_grid();
		// We have to simulate MAX_CTA ctas with SimpleGPUSchedule::default_max_ctas
		// and multiplex the cta blocks some how
		for (dyn_var<int> vcta_id = *cta_id_ptr; vcta_id < MAX_CTA; 
			vcta_id = vcta_id + SimpleGPUSchedule::default_max_cta) {
			TWCE_load_balance_impl(graph, f, count, enumerator, src_filter, dst_filter, schedule,
				vcta_id, *thread_id_ptr, MAX_CTA, CTA_SIZE);
			graphit::runtime::sync_threads();		
		}
		graphit::runtime::sync_grid();
		return;
	}
	

	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);		
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			TWCE_load_balance_impl(graph, f, count, enumerator, src_filter, dst_filter, schedule,
				cta_id, thread_id, MAX_CTA, CTA_SIZE);
		}
	}
	current_context = context_type::HOST;
}

template <typename FT, typename ET, typename SFT, typename DFT>
void CM_load_balance_impl(GraphT graph, FT f, dyn_var<int> count, ET enumerator, SFT src_filter, 
		DFT dst_filter, SimpleGPUSchedule* schedule, dyn_var<int> &cta_id, 
		dyn_var<int> &thread_id, dyn_var<int> &MAX_CTA, int CTA_SIZE) {	
	dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;

	dyn_var<int[1024]> sm_idx; set_shared(sm_idx);
        dyn_var<int[1024]> sm_deg; set_shared(sm_deg);
	dyn_var<int[1024]> sm_loc; set_shared(sm_loc);
	
	dyn_var<int32_t> tot_size = count;
	dyn_var<int> deg, index, src_idx;
	if (tid < tot_size) {
		index = enumerator(tid);
		deg = graph.out_degrees[index];
		
		sm_idx[thread_id] = index;
		sm_deg[thread_id] = deg;
		sm_loc[thread_id] = graph.d_src_offsets[index];
		
	} else {
		deg = 0;
		sm_deg[thread_id] = deg;
	}
	
	dyn_var<int> lane = thread_id % 32;
	dyn_var<int> offset = 0;	

	dyn_var<int> cosize = CTA_SIZE;
	dyn_var<int> tot_deg;
	dyn_var<int> phase = thread_id;
	dyn_var<int> off = 32;
	
	for (dyn_var<int> d = 2; d <= 32; d = d * 2) {
		dyn_var<int> temp = runtime::shfl_up_sync(-1, deg, d / 2);
		if (lane % d == d - 1) deg = deg + temp;
	}
	sm_deg[thread_id] = deg;
	
	for (dyn_var<int> d = cosize / 64; d > 0; d = d / 2) {
		runtime::sync_threads();
		if (phase < d) {
			dyn_var<int> ai = off * (2 * phase + 1) - 1;
			dyn_var<int> bi = off * (2 * phase + 2) - 1;
			sm_deg[bi] = sm_deg[bi] + sm_deg[ai];
		}
		off = off / 2;
	}
	runtime::sync_threads();
	tot_deg = sm_deg[cosize - 1];
	runtime::sync_threads();
	if (!phase) sm_deg[cosize - 1] = 0;
	runtime::sync_threads();
	for (dyn_var<int32_t> d = 1; d < (cosize / 32); d = d * 2) {
		off = off / 2;
		runtime::sync_threads();
		if (phase < d) {
			dyn_var<int> ai = off * (2 * phase + 1) - 1;
			dyn_var<int> bi = off * (2 * phase + 2) - 1;
			dyn_var<int> t = sm_deg[ai];
			sm_deg[ai] = sm_deg[bi];
			sm_deg[bi] = sm_deg[bi] + t;		
		}
	}
	runtime::sync_threads();
	deg = sm_deg[thread_id];
	runtime::sync_threads();
	for (dyn_var<int> d = 32; d > 1; d = d / 2) {
		dyn_var<int> temp_big = runtime::shfl_down_sync(-1, deg, d/2);
		dyn_var<int> temp_small = runtime::shfl_up_sync(-1, deg, d/2);
		if (lane % d == (d/2 - 1)) deg = temp_big;
		else if (lane % d == (d - 1)) deg = deg + temp_small;
	}
	sm_deg[thread_id] = deg;
	runtime::sync_threads();

	dyn_var<int> width = tid - thread_id + CTA_SIZE;
	if (tot_size < width) width = tot_size;
	width = width - (tid - thread_id);
	for (dyn_var<int32_t> i = thread_id; i < tot_deg; i = i + CTA_SIZE) {
		dyn_var<int32_t> id = runtime::binary_search_upperbound(&sm_deg[offset], width, i)-1;
		if (id >= width) continue;
		src_idx = sm_idx[offset + id];
		if (src_filter(src_idx) == false)
			continue;
		dyn_var<int> ei = sm_loc[offset + id] + i - sm_deg[offset + id];
		dyn_var<int> dst_idx = graph.d_edge_dst[ei];
		if (dst_filter(dst_idx)) {
			Vertex src(src_idx);
			Vertex dst(dst_idx);
			src.current_access = Vertex::access_type::SHARED;	
			dst.current_access = Vertex::access_type::SHARED;	
			//f(src, dst, graph.d_edge_weight[ei]);
			f(src, dst, 0);
		}

	}
		
}

template <typename FT, typename ET, typename SFT, typename DFT>
void CM_load_balance(GraphT graph, FT f, dyn_var<int> count, ET enumerator, SFT src_filter, 
		DFT dst_filter, SimpleGPUSchedule* schedule) {
	
	int CTA_SIZE = schedule->cta_size;
	dyn_var<int> num_threads = count;
	dyn_var<int> MAX_CTA = (num_threads + CTA_SIZE - 1)/CTA_SIZE;
	
	if (current_context == context_type::DEVICE) {
		assert(CTA_SIZE == SimpleGPUSchedule::default_cta_size && "Currently cannot generate fused kernels with different CTA sizes");
		graphit::runtime::sync_grid();
		// We have to simulate MAX_CTA ctas with SimpleGPUSchedule::default_max_ctas
		// and multiplex the cta blocks some how
		for (dyn_var<int> vcta_id = *cta_id_ptr; vcta_id < MAX_CTA; 
			vcta_id = vcta_id + SimpleGPUSchedule::default_max_cta) {
			CM_load_balance_impl(graph, f, count, enumerator, src_filter, dst_filter, schedule,
				vcta_id, *thread_id_ptr, MAX_CTA, CTA_SIZE);
			graphit::runtime::sync_threads();		
		}
		graphit::runtime::sync_grid();
		return;
	}
	

	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);		
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			CM_load_balance_impl(graph, f, count, enumerator, src_filter, dst_filter, schedule,
				cta_id, thread_id, MAX_CTA, CTA_SIZE);
		}
	}
	current_context = context_type::HOST;
}
// Host only implementation for now
template <typename FT>
void create_reverse_frontier(dyn_var<int> count, VertexSubset* e, FT f) {
	int CTA_SIZE = SimpleGPUSchedule::default_cta_size;
	int MAX_CTA = SimpleGPUSchedule::default_max_cta;

	current_context = context_type::DEVICE;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			dyn_var<int> tid = cta_id * CTA_SIZE + thread_id;
			for (dyn_var<int> vid = tid; vid < count; vid = vid + CTA_SIZE * MAX_CTA) {
				if (f(vid)) {
					runtime::enqueue_sparse(*e, vid);
				}
			}
		}
	}
	current_context = context_type::HOST;
}
void edgeset_apply::apply(GraphT &graph, edgeset_apply_udf_w_t udf) {
	
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
			e1.apply(graph, udf);
		} else {
			e2.apply(graph, udf);
		}
		return;	
	}

	dyn_var<int> count;	
	std::function<dyn_var<int>(dyn_var<int>)> enumerator;
	std::function<bool(dyn_var<int>)> src_filter;
	std::function<bool(dyn_var<int>)> dst_filter;
	dyn_var<GraphT::super_name*> graph_to_use = &graph;
	
	std::function<void(Vertex, Vertex, dyn_var<int>)> udf_apply;
	
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
			count = (*from_set).size();
			enumerator = [=] (dyn_var<int> a) {
				return (*from_set).d_sparse_queue_input[a];
			};
			if (current_context == context_type::HOST)
				graphit::runtime::to_sparse_host(*from_set);
			else
				graphit::runtime::to_sparse_device(*from_set);
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
		if (to_filter) {
			src_filter = [=] (dyn_var<int> v) {
				return (bool) to_filter((Vertex)v);
			};
		} else 
			src_filter = true_func;
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
					return (bool) graphit::runtime::checkBit((*from_set).d_bit_map_input, a);
				};
			} else {
				graphit::runtime::to_boolmap(*from_set);
				dst_filter = [=] (dyn_var<int> a) -> bool {
					return (bool) (*from_set).d_boolmap[a];
				};
			}
			// Create a reverse frontier first
			create_reverse_frontier(graph.num_vertices, from_set, src_filter);	
			if (current_context == context_type::DEVICE)
				(*from_set).swap_queues_device();	
			else
				(*from_set).swap_queues_host();

			count = (*from_set).size();
			enumerator = [=] (dyn_var<int> a) {
				return (*from_set).d_sparse_queue_input[a];
			};
		}
		//GraphT g = graph_to_use[0];
		graph_to_use = ((GraphT)(builder::cast)graph_to_use[0]).get_transposed_graph();
		udf_apply = [=] (Vertex src, Vertex dst, dyn_var<int> w) {
			udf(dst, src, w);
		};
	}

	if (simple_schedule->load_balancing == SimpleGPUSchedule::load_balancing_type::EDGE_ONLY 
		&& simple_schedule->edge_blocking == SimpleGPUSchedule::edge_blocking_type::BLOCKED) {
		assert(simple_schedule->block_size != -1 && "Invalid block size value");	
		//GraphT g = graph_to_use[0];
		graph_to_use = ((GraphT)(builder::cast)graph_to_use[0]).get_blocked_graph(simple_schedule->block_size);
	}
	
	

	switch(simple_schedule->load_balancing) {	
		case SimpleGPUSchedule::load_balancing_type::VERTEX_BASED: vertex_based_load_balance(graph_to_use[0], udf_apply, count, enumerator, src_filter, dst_filter, simple_schedule); break;
		
		case SimpleGPUSchedule::load_balancing_type::EDGE_ONLY: 
			if (simple_schedule->edge_blocking == SimpleGPUSchedule::edge_blocking_type::UNBLOCKED)
				edge_only_load_balance(graph_to_use[0], udf_apply, count, enumerator, src_filter, dst_filter, simple_schedule); 
			else 
				edge_only_load_balance_blocked(graph_to_use[0], udf_apply, count, enumerator, src_filter, dst_filter, simple_schedule); 
			break;
		case SimpleGPUSchedule::load_balancing_type::TWCE:
			TWCE_load_balance(graph_to_use[0], udf_apply, count, enumerator, src_filter, dst_filter, simple_schedule); break;	
		
		case SimpleGPUSchedule::load_balancing_type::CM:
			CM_load_balance(graph_to_use[0], udf_apply, count, enumerator, src_filter, dst_filter, simple_schedule); break;		
		default:
			assert(false && "Load balancing strategy currently not supported");
	}

	
}
// Kernel fusion related implementations
void fuse_kernel(bool to_fuse, std::function<void (void)> body) {
	if (!to_fuse) {
		body();
		return;
	}	
	assert(current_context == context_type::HOST && "Fuse Kernel cannot be called when on Device");
	
	int MAX_CTA = SimpleGPUSchedule::default_max_cta;
	int CTA_SIZE = SimpleGPUSchedule::default_cta_size;	
	
	current_context = context_type::DEVICE;
	builder::annotate(CUDA_COOP_ANNOTATION_STRING);
	for (dyn_var<int> cta_id = 0; cta_id < MAX_CTA; cta_id = cta_id + 1) {
		for (dyn_var<int> thread_id = 0; thread_id < CTA_SIZE; thread_id = thread_id + 1) {
			cta_id_ptr = cta_id.addr();
			thread_id_ptr = thread_id.addr();
			body();	
		}
	}
	current_context = context_type::HOST;
}

}
