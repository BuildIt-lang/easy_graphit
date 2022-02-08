#ifndef GRAPHIT_TYPES_H
#define GRAPHIT_TYPES_H
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

#define VERTEXSUBSET_T_NAME "graphit_runtime::vertexsubset"
extern const char vertexsubset_t_name[sizeof(VERTEXSUBSET_T_NAME)];

#define PRIOQUEUE_T_NAME "graphit_runtime::PrioQueue<int>"
extern const char prioqueue_t_name[sizeof(PRIOQUEUE_T_NAME)];

#define FRONTIERLIST_T_NAME "graphit_runtime::FrontierList"
extern const char frontierlist_t_name[sizeof(FRONTIERLIST_T_NAME)];



struct GraphT: public dyn_var<builder::name<graph_t_name>> {
	typedef dyn_var<builder::name<graph_t_name>> super;
	using super_name = builder::name<graph_t_name>;
	using super::dyn_var;
	using super::operator=;
	GraphT(const GraphT &t): dyn_var<super_name>((builder::builder)t) {}
	builder::builder operator= (const GraphT &t) {
		return (*this) = (builder::builder)t;
	}
	GraphT* addr(void) {
		return this;
	}
	// GraphT specific members
	dyn_var<int> num_vertices = as_member_of(this, "num_vertices");
	dyn_var<int> num_edges = as_member_of(this, "num_edges");
	dyn_var<int*> d_src_offsets = as_member_of(this, "d_src_offsets");
	dyn_var<int*> d_edge_dst = as_member_of(this, "d_edge_dst");
	dyn_var<int*> d_edge_weight = as_member_of(this, "d_edge_weight");
	dyn_var<int*> d_edge_src = as_member_of(this, "d_edge_src");
	dyn_var<int*> out_degrees = as_member_of(this, "d_out_degrees");
	
	// GraphT blocking and transpose specific members
	dyn_var<GraphT::super_name(int)> get_blocked_graph = as_member_of(this, "get_blocked_graph");
	dyn_var<GraphT::super_name(void)> get_transposed_graph = as_member_of(this, "get_transposed_graph");
	dyn_var<int> num_buckets = as_member_of(this, "num_buckets");
	dyn_var<int*> d_bucket_sizes = as_member_of(this, "d_bucket_sizes");	
};

struct VertexSubset: public dyn_var<builder::name<vertexsubset_t_name>> {
	typedef dyn_var<builder::name<vertexsubset_t_name>> super;
	using super_name = builder::name<vertexsubset_t_name>;
	using super::dyn_var;
	using super::operator=;

	VertexSubset(const VertexSubset &t): dyn_var<super_name>((builder::builder)t) {}
	builder::builder operator= (const VertexSubset &t) {
		return (*this) = (builder::builder)t;
	}
	VertexSubset* addr(void) {
		return this;
	}

	// VertexSubset related members
	dyn_var<int> max_elems = as_member_of(this, "max_elems");
	dyn_var<int> num_elems = as_member_of(this, "num_elems");
	dyn_var<int*> d_sparse_queue_input = as_member_of(this, "d_sparse_queue_input");
	dyn_var<char*> d_bit_map_input = as_member_of(this, "d_bit_map_input");
	dyn_var<char*> d_boolmap = as_member_of(this, "d_boolmap");
	dyn_var<void(int)> addVertex = as_member_of(this, "addVertex");

	dyn_var<void(void)> swap_queues_host = as_member_of(this, "swap_queues_host");
	dyn_var<void(void)> swap_bitmaps_host = as_member_of(this, "swap_bitmaps_host");
	dyn_var<void(void)> swap_boolmaps_host = as_member_of(this, "swap_boolmaps_host");
	dyn_var<void(void)> swap_queues_device = as_member_of(this, "swap_queues_device");
	dyn_var<void(void)> swap_bitmaps_device = as_member_of(this, "swap_bitmaps_device");
	dyn_var<void(void)> swap_boolmaps_device = as_member_of(this, "swap_boolmaps_device");

	dyn_var<int(void)> size_host = as_member_of(this, "size_host");
	dyn_var<int(void)> size_device = as_member_of(this, "size_device");

	dyn_var<int> curr_dedup_counter = as_member_of(this, "curr_dedup_counter");

	builder::builder size(void) {
		if (current_context == context_type::HOST)
			return size_host();	
		else
			return size_device();	
	}
	
};

struct PrioQueue: public dyn_var<builder::name<prioqueue_t_name>> {
	typedef dyn_var<builder::name<prioqueue_t_name>> super;
	using super_name = builder::name<prioqueue_t_name>;
	using super::dyn_var;
	using super::operator=;
	PrioQueue(const PrioQueue &t): dyn_var<super_name>((builder::builder)t) {}
	builder::builder operator= (const PrioQueue &t) {
		return (*this) = (builder::builder)t;
	}	
	PrioQueue* addr(void) {
		return this;
	}

	//PrioQueue specific members
	dyn_var<void(void)> init = as_member_of(this, "init");
	dyn_var<VertexSubset::super_name(void)> dequeue = as_member_of(this, "dequeue");
	dyn_var<VertexSubset::super_name(void)> dequeue_device = as_member_of(this, "dequeue_device");
	dyn_var<int(void)> finished = as_member_of(this, "finished");
	dyn_var<int(void)> finished_device = as_member_of(this, "finished_device");
	VertexSubset frontier_ = as_member_of(this, "frontier_");
	dyn_var<int> prio_cutoff = as_member_of(this, "prio_cutoff");
	
};

struct FrontierList: public dyn_var<builder::name<frontierlist_t_name>> {
	typedef dyn_var<builder::name<frontierlist_t_name>> super;
	using super_name = builder::name<frontierlist_t_name>;
	using super::dyn_var;
	using super::operator=;
	FrontierList(const FrontierList &t): dyn_var<super_name>((builder::builder)t) {}
	builder::builder operator= (const FrontierList &t) {
		return (*this) = (builder::builder)t;
	}
	FrontierList* addr(void) {
		return this;
	}

	// FrontierList specific members
	dyn_var<void(VertexSubset::super_name)> insert_host = as_member_of(this, "insert_host");
	dyn_var<VertexSubset::super_name(void)> retrieve_host = as_member_of(this, "retrieve_host");
	dyn_var<void(VertexSubset::super_name)> insert_device = as_member_of(this, "insert_device");
	dyn_var<VertexSubset::super_name(void)> retrieve_device = as_member_of(this, "retrieve_device");

	void insert(VertexSubset &v) {
		if (current_context == context_type::HOST)
			insert_host(v);
		else
			insert_device(v);
	}
	void retrieve(VertexSubset &v) {
		if (current_context == context_type::HOST)
			retrieve_host(v);
		else
			retrieve_device(v);
	}
	
};

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
};
namespace runtime {
extern dyn_var<void (VertexSubset::super_name, int)> enqueue_bitmap;
extern dyn_var<void (VertexSubset::super_name, int)> enqueue_boolmap;
extern dyn_var<void (VertexSubset::super_name, int)> enqueue_sparse;
extern dyn_var<void (VertexSubset::super_name, int)> enqueue_sparse_no_dupes;
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
	VertexSubset *output_queue;
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
	VertexSubset *output_queue;
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
	PrioQueue pq;
	VertexData<int> *p;

	SimpleGPUSchedule* current_schedule;
	PriorityQueue(std::string name): pq(name.c_str()) {}
	void init(GraphT &edges, VertexData<int>& p_, dyn_var<int> init_p, dyn_var<int> delta, dyn_var<int> src) {
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
	VertexSubset dequeue(void) {
		if (current_context == context_type::DEVICE)
			return pq.dequeue_device();
		else
			return pq.dequeue();

	}
};


}

#endif
