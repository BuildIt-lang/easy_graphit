#ifndef GRAPHIT_TYPES_H
#define GRAPHIT_TYPES_H
#include "graphit/dyn_core.h"
#include "builder/member_base.h"
#include "graphit/schedule.h"
#include "pipeline/extract_cuda.h"

namespace graphit {

enum class context_type {
	HOST = 0,
	DEVICE = 1
};

extern enum context_type current_context;
struct Vertex;

template <typename T>
struct VertexData;

template <typename T>
struct VertexDataIndex;

// Commonly used named types in GraphIt
#define GRAPH_T_NAME "graphit_runtime::GraphT<int>"
extern const char graph_t_name[sizeof(GRAPH_T_NAME)];
using GraphT = typename builder::name<graph_t_name>;

#define VERTEXSUBSET_T_NAME "graphit_runtime::vertexsubset"
extern const char vertexsubset_t_name[sizeof(VERTEXSUBSET_T_NAME)];
using VertexSubset = typename builder::name<vertexsubset_t_name>;

#define PRIOQUEUE_T_NAME "graphit_runtime::PrioQueue<int>"
extern const char prioqueue_t_name[sizeof(PRIOQUEUE_T_NAME)];
using PrioQueue = typename builder::name<prioqueue_t_name>;

#define FRONTIERLIST_T_NAME "graphit_runtime::FrontierList"
extern const char frontierlist_t_name[sizeof(FRONTIERLIST_T_NAME)];
using FrontierList = typename builder::name<frontierlist_t_name>;

template <>
struct member<0>: public builder::member_base_impl<gbuilder> {
	using builder::member_base_impl<gbuilder>::member_base_impl;

	typedef gbuilder member_associated_BT;	
	gbuilder operator[](const gbuilder& t) const;
	gbuilder operator=(const gbuilder& t) const;
	template <typename...Types>
	gbuilder operator()(Types...args);

};

template <int recur>
struct member: public builder::member_base_impl<gbuilder> {
	using builder::member_base_impl<gbuilder>::member_base_impl;
	typedef member<recur-1> member_t;

	// Operator =
	gbuilder operator=(const gbuilder& t) const {
		//return ((gbuilder)*this) = t;
		gbuilder b(*this);
		return (b = t);
	}
	gbuilder operator[](const gbuilder& t) const {
		gbuilder b(*this);
		return (b[t]);
	}
	template <typename...Types>
	gbuilder operator()(Types...args) {
		gbuilder b(*this);
		return b(args...);
	}

	// All the members graphit types can have
	
	// GraphT related types
	member_t num_vertices = member_t(this, "num_vertices");
	member_t num_edges = member_t(this, "num_edges");
	member_t d_src_offsets = member_t(this, "d_src_offsets");
	member_t d_edge_dst = member_t(this, "d_edge_dst");
	member_t d_edge_weight = member_t(this, "d_edge_weight");
	member_t d_edge_src = member_t(this, "d_edge_src");
	member_t out_degrees = member_t(this, "d_out_degrees");
	
	member_t get_blocked_graph = member_t(this, "get_blocked_graph");
	member_t get_transposed_graph = member_t(this, "get_transposed_graph");
	member_t num_buckets = member_t(this, "num_buckets");
	member_t d_bucket_sizes = member_t(this, "d_bucket_sizes");

	// VertexSubset related types
	member_t max_elems = member_t(this, "max_elems");
	member_t num_elems = member_t(this, "num_elems");
	member_t d_sparse_queue_input = member_t(this, "d_sparse_queue_input");
	member_t d_bit_map_input = member_t(this, "d_bit_map_input");
	member_t d_boolmap = member_t(this, "d_boolmap");
	member_t addVertex = member_t(this, "addVertex");

	member_t swap_queues_host = member_t(this, "swap_queues_host");
	member_t swap_bitmaps_host = member_t(this, "swap_bitmaps_host");
	member_t swap_boolmaps_host = member_t(this, "swap_boolmaps_host");
	member_t swap_queues_device = member_t(this, "swap_queues_device");
	member_t swap_bitmaps_device = member_t(this, "swap_bitmaps_device");
	member_t swap_boolmaps_device = member_t(this, "swap_boolmaps_device");

	member_t size_host = member_t(this, "size_host");
	member_t size_device = member_t(this, "size_device");

	member_t curr_dedup_counter = member_t(this, "curr_dedup_counter");

	// Priority Queue related members
	member_t init = member_t(this, "init");
	member_t dequeue = member_t(this, "dequeue");
	member_t dequeue_device = member_t(this, "dequeue_device");
	member_t finished = member_t(this, "finished");
	member_t finished_device = member_t(this, "finished_device");
	member_t frontier_ = member_t(this, "frontier_");
	member_t prio_cutoff = member_t(this, "prio_cutoff");

	// FrontierList related members
	member_t insert_host = member_t(this, "insert_host");
	member_t retrieve_host = member_t(this, "retrieve_host");
	member_t insert_device = member_t(this, "insert_device");
	member_t retrieve_device = member_t(this, "retrieve_device");

	// Type specilization for host vs device
	gbuilder size(void) {
		if (current_context == context_type::HOST)
			return size_host();	
		else
			return size_device();	
	}
	void insert(dyn_var<VertexSubset> &v) {
		if (current_context == context_type::HOST)
			insert_host(v);
		else
			insert_device(v);
	}
	void retrieve(dyn_var<VertexSubset> &v) {
		if (current_context == context_type::HOST)
			retrieve_host(v);
		else
			retrieve_device(v);
	}

};

template <typename...Types>
gbuilder member<0>::operator()(Types...args) {
	gbuilder b(*this);
	return b(args...);
}


struct Vertex {
// Primary members to hold the vertex id
	dyn_var<int> vid;
// Analysis related fields
	enum class access_type {
		INDEPENDENT = 0,
		SHARED = 1,
		CONSTANT = 2
	};
	access_type current_access;	
// Constructors

	Vertex(const dyn_var<int>& vid_expr): vid (vid_expr) {
		// Initialize the analysis bits appropriately
		current_access = access_type::SHARED;	
	}
	// Implicit constructor from a constant
	Vertex(const int &x) {
		vid = x;
		// Initialize the analysis bits appropriately
		current_access = access_type::CONSTANT;	
	}	

	operator dyn_var<int> () {
		return vid;	
	}
/*
	operator gbuilder() {
		return (gbuilder) vid;
	}
*/
};
namespace runtime {
extern dyn_var<void (VertexSubset, int)> enqueue_bitmap;
extern dyn_var<void (VertexSubset, int)> enqueue_boolmap;
extern dyn_var<void (VertexSubset, int)> enqueue_sparse;
extern dyn_var<void (VertexSubset, int)> enqueue_sparse_no_dupes;
extern dyn_var<void (void)> dedup_frontier_perfect;
extern dyn_var<void (void*, void*, int)> copyHostToDevice;
extern dyn_var<void (void*, void*, int)> copyDeviceToHost;
extern dyn_var<int (void*, int)> writeMin;
extern dyn_var<int (void*, int)> writeSum;
extern dyn_var<void (void*, int)> cudaMalloc;
extern dyn_var<void (void*, void*, int, int)> cudaMemcpyToSymbol;
}
template <typename T>
struct VertexDataIndex {
// Primary members to hold the vertex index data
	dyn_var<T*> &variable;
	dyn_var<int> index;
// Analysis related fields
	bool is_tracked;	
	bool allow_dupes;	
	dyn_var<VertexSubset> *output_queue;
	Vertex::access_type current_access;
	graphit::SimpleGPUSchedule::frontier_creation_type frontier_creation;
// Constructors
	VertexDataIndex(dyn_var<T*>& variable_expr, dyn_var<int> index_expr): variable(variable_expr), index(index_expr) {
		// Initialize the analysis related bits appropriately
		is_tracked = false;
		allow_dupes = false;
		frontier_creation = graphit::SimpleGPUSchedule::frontier_creation_type::SPARSE;
	}
	VertexDataIndex(const VertexDataIndex &other): variable(other.variable), index(other.index) {
		// Copy over the analysis bits as expected
		is_tracked = other.is_tracked;
		allow_dupes = other.allow_dupes;
		output_queue = other.output_queue;
		current_access = other.current_access;
		frontier_creation = other.frontier_creation;
	}
	operator dyn_var<T>() const {
		if (current_context == context_type::DEVICE) {
			dyn_var<T> ret = variable[index];
			return ret;	
		} else {
			dyn_var<T> temp;
			runtime::copyDeviceToHost(&temp, &(variable[index]), (int) sizeof(T));
			return temp;
		}
	}

	void enqueue_appropriate() {
		if (frontier_creation == graphit::SimpleGPUSchedule::frontier_creation_type::SPARSE) {
			if (allow_dupes)
				runtime::enqueue_sparse(*output_queue, index);
			else 
				runtime::enqueue_sparse_no_dupes(*output_queue, index);
		}
		else if(frontier_creation == graphit::SimpleGPUSchedule::frontier_creation_type::BITMAP)
			runtime::enqueue_bitmap(*output_queue, index);
		else 
			runtime::enqueue_boolmap(*output_queue, index);
	}
	void operator=(const dyn_var<T> &expr) {
		// Default implementation
		// Change this based on analysis
		// And schedules
		if (current_context == context_type::DEVICE) {
			if (current_access != Vertex::access_type::INDEPENDENT) {
				builder::annotate(NEEDS_ATOMICS_ANNOTATION);
			}
			variable[index] = expr; 
			if (is_tracked == true) {
				enqueue_appropriate();	
			}
		} else {
			dyn_var<T> temp = expr;
			runtime::copyHostToDevice(&(variable[index]), &temp, (int)sizeof(T));
		}
	}
	void operator=(const VertexDataIndex& expr) {
		(*this) = (dyn_var<T>) expr;
	}
	void min(const dyn_var<T> &expr) {
		if (current_access == Vertex::access_type::INDEPENDENT) {
			if (variable[index] > expr) {
				variable[index] = expr;
				if (is_tracked == true) {
					enqueue_appropriate();	
				}
			}
		} else {
			dyn_var<int> res = runtime::writeMin(&(variable[index]), expr);
			if (is_tracked == true)
				if (res)
					enqueue_appropriate();
		}
	}
	void operator+=(const dyn_var<T> &expr) {
		if (current_access == Vertex::access_type::INDEPENDENT) {
			variable[index] = variable[index] + expr;
			if (is_tracked == true) {
				enqueue_appropriate();
			}
		} else {
			runtime::writeSum(&(variable[index]), expr);
			if (is_tracked == true) {
				enqueue_appropriate();
			}
		}
	}
};
template <typename T, typename OT>
dyn_var<T> operator+ (const VertexDataIndex<T>& vdi, const OT& rhs) {
	return (dyn_var<T>)vdi + rhs;
}
template <typename T, typename OT>
dyn_var<T> operator/ (const VertexDataIndex<T>& vdi, const VertexDataIndex<OT>& rhs) {
	return (dyn_var<T>)vdi / (dyn_var<OT>)rhs;
}
template <typename T, typename OT>
dyn_var<T> operator/ (const VertexDataIndex<T>& vdi, const OT& rhs) {
	return (dyn_var<T>)vdi / rhs;
}
template <typename T, typename OT>
dyn_var<T> operator/ (const OT& lhs, const VertexDataIndex<T>& vdi) {
	return lhs / (dyn_var<T>)vdi;
}
template <typename T, typename OT>
dyn_var<T> operator* (const VertexDataIndex<T>& vdi, const OT& rhs) {
	return (dyn_var<T>)vdi * rhs;
}
template <typename T, typename OT>
dyn_var<T> operator* (const OT& lhs, const VertexDataIndex<T>& vdi) {
	return lhs * (dyn_var<T>)vdi;
}
template <typename T, typename OT>
dyn_var<T> operator == (const VertexDataIndex<T>& vdi, const OT& rhs) {
	return (dyn_var<T>)vdi == rhs;
}
template <typename T, typename OT>
dyn_var<T> operator == (const OT&lhs, const VertexDataIndex<T>& vdi) {
	return lhs == (dyn_var<T>)vdi;
}
template <typename T, typename OT>
dyn_var<T> operator- (const VertexDataIndex<T>& vdi, const OT& rhs) {
	return (dyn_var<T>)vdi - rhs;
}
template <typename T, typename OT>
dyn_var<T> operator- (const OT& lhs, const VertexDataIndex<T>& vdi) {
	return lhs - (dyn_var<T>)vdi;
}

// Vertex Data related types
template <typename T>
struct VertexData {
// Primary member to hold the vertex data
	dyn_var<T*> data;

// Analysis related fields
	bool is_tracked;	
	bool allow_dupes;
	dyn_var<VertexSubset> *output_queue;
	graphit::SimpleGPUSchedule::frontier_creation_type frontier_creation;
// Constructors
	VertexData(std::string name): data(name.c_str()) {
		//std::vector<std::string> attrs;
		//attrs.push_back("__device__");
		//data.block_var->setMetadata("attributes", attrs);	
		is_tracked = false;
		allow_dupes = false;
		output_queue = nullptr;
		frontier_creation = graphit::SimpleGPUSchedule::frontier_creation_type::SPARSE;
	}
	VertexData(const VertexData& other): data(other.data) {
		// Copy over analysis related bits appropriately	
		is_tracked = other.is_tracked;
		allow_dupes = other.allow_dupes;
		output_queue = other.output_queue;
		frontier_creation = other.frontier_creation;
	}
// Operations
	VertexDataIndex<T> operator[] (dyn_var<int> index) {
		VertexDataIndex<T> ret (data, index);
		// Set the analysis bits accordingly		
		ret.is_tracked = is_tracked;
		ret.output_queue = output_queue;
		ret.frontier_creation = frontier_creation;
		ret.allow_dupes = allow_dupes;
		return ret;
	
	}
	VertexDataIndex<T> operator[] (const Vertex &index) {
		VertexDataIndex<T> ret (data, index.vid);
		// Set the analysis bits accordingly	
		ret.is_tracked = is_tracked;
		ret.output_queue = output_queue;
		ret.current_access = index.current_access;
		ret.frontier_creation = frontier_creation;
		ret.allow_dupes = allow_dupes;
		return ret;
	}
	VertexDataIndex<T> operator[] (const int& index) {
		VertexDataIndex<T> ret(data, index);
		ret.is_tracked = is_tracked;
		ret.output_queue = output_queue;
		ret.current_access = Vertex::access_type::CONSTANT;
		ret.frontier_creation = frontier_creation;
		ret.allow_dupes = allow_dupes;
		return ret;
	}
	void allocate(dyn_var<int> size) {
		runtime::cudaMalloc(&data, size * (int)sizeof(T));
	}
};

struct PriorityQueue {
	dyn_var<PrioQueue> pq;
	VertexData<int> *p;

	SimpleGPUSchedule* current_schedule;
	PriorityQueue(std::string name): pq(name.c_str()) {}
	void init(dyn_var<GraphT> &edges, VertexData<int>& p_, dyn_var<int> init_p, dyn_var<int> delta, dyn_var<int> src) {
		p = &p_;
		pq.init(edges, p->data, init_p, delta, src);
	}
	void enqueue_appropriate(dyn_var<int> index) {
		if (current_schedule->frontier_creation == graphit::SimpleGPUSchedule::frontier_creation_type::SPARSE)
			runtime::enqueue_sparse(pq.frontier_, index);
		else if(current_schedule->frontier_creation == graphit::SimpleGPUSchedule::frontier_creation_type::BITMAP)
			runtime::enqueue_bitmap(pq.frontier_, index);
		else 
			runtime::enqueue_boolmap(pq.frontier_, index);
	}
	void updatePriorityMin(Vertex v, dyn_var<int> np) {
		if (runtime::writeMin(&(p->data[(dyn_var<int>)v]), np) && np < pq.prio_cutoff()) {
			enqueue_appropriate(v);
		}	
	}
	dyn_var<int> finished(void) {
		if (current_context == context_type::DEVICE)
			return pq.finished_device();
		else
			return pq.finished();
	}
	dyn_var<VertexSubset> dequeue(void) {
		if (current_context == context_type::DEVICE)
			return pq.dequeue_device();
		else
			return pq.dequeue();

	}
};


}

#endif
