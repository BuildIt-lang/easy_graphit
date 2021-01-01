#ifndef GRAPHIT_TYPES_H
#define GRAPHIT_TYPES_H

#include "builder/dyn_var.h"
#include "builder/member_base.h"
#include "builder/builder.h"

namespace graphit {

enum class context_type {
	HOST = 0,
	DEVICE = 1
};

extern enum context_type current_context;

template <int recur>
struct member;

#define MAX_MEMBER_DEPTH (3)
using gbuilder = builder::builder_base<member<MAX_MEMBER_DEPTH>>;

template <typename T>
using dyn_var = builder::dyn_var_base<T, member<MAX_MEMBER_DEPTH>, gbuilder>;

template <>
struct member<0>: public builder::member_base {
	using builder::member_base::member_base;
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

	// All the members graphit types can have
	
	// GraphT related types
	member_t num_vertices = member_t(this, "num_vertices");
	member_t num_edges = member_t(this, "num_edges");
	member_t d_row_offsets = member_t(this, "d_row_offsets");
	member_t d_edges_dst = member_t(this, "d_edges_dst");


	// VertexSubset related types
	member_t max_elems = member_t(this, "max_elems");
	member_t num_elems = member_t(this, "num_elems");
	member_t d_sparse_queue = member_t(this, "d_sparse_queue");

};






// Commonly used named types in GraphIt
#define GRAPH_T_NAME "graphit::GraphT"
extern const char graph_t_name[sizeof(GRAPH_T_NAME)];
using GraphT = typename builder::name<graph_t_name>;

#define VERTEXSUBSET_T_NAME "graphit::vertexsubset"
extern const char vertexsubset_t_name[sizeof(VERTEXSUBSET_T_NAME)];
using VertexSubset = typename builder::name<vertexsubset_t_name>;


template <typename T>
struct VertexData;

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
};
namespace runtime {
extern dyn_var<void (VertexSubset, int)> enqueue_sparse;
extern dyn_var<void (void*, void*, int)> copyHostToDevice;
extern dyn_var<void (void*, void*, int)> copyDeviceToHost;
extern dyn_var<int (void*, int)> writeMin;
}
template <typename T>
struct VertexDataIndex {
// Primary members to hold the vertex index data
	dyn_var<T*> &variable;
	dyn_var<int> index;
// Analysis related fields
	bool is_tracked;	
	dyn_var<VertexSubset> *output_queue;
	Vertex::access_type current_access;
// Constructors
	VertexDataIndex(dyn_var<T*>& variable_expr, dyn_var<int> index_expr): variable(variable_expr), index(index_expr) {
		// Initialize the analysis related bits appropriately
		is_tracked = false;
	}
	VertexDataIndex(const VertexDataIndex &other): variable(other.variable), index(other.index) {
		// Copy over the analysis bits as expected
		is_tracked = other.is_tracked;
		output_queue = other.output_queue;
		current_access = other.current_access;
	}
	operator dyn_var<T>() const {
		dyn_var<T> ret = variable[index];
		return ret;	
	}
	void operator=(const dyn_var<T>& expr) {
		// Default implementation
		// Change this based on analysis
		// And schedules
		if (current_context == context_type::DEVICE) {
			if (current_access != Vertex::access_type::INDEPENDENT)
				assert(false && "Direct assignment to shared variable");
			variable[index] = expr; 
			if (is_tracked == true) {
				runtime::enqueue_sparse(*output_queue, index);	
			}
		} else {
			dyn_var<T> temp = expr;
			runtime::copyHostToDevice(&(variable[index]), &temp, (int)sizeof(T));
		}
	}
	void min(const dyn_var<T> &expr) {
		if (current_access == Vertex::access_type::INDEPENDENT) {
			if (variable[index] > expr) {
				variable[index] = expr;
				if (is_tracked == true) {
					runtime::enqueue_sparse(*output_queue, index);	
				}
			}
		} else {
			dyn_var<int> res = runtime::writeMin(&(variable[index]), expr);
			if (is_tracked == true)
				if (res)
					runtime::enqueue_sparse(*output_queue, index);
		}
	}
};
template <typename T, typename OT>
dyn_var<T> operator+ (const VertexDataIndex<T>& vdi, const OT& rhs) {
	return (dyn_var<T>)vdi + rhs;
}

// Vertex Data related types
template <typename T>
struct VertexData {
// Primary member to hold the vertex data
	dyn_var<T*> data;

// Analysis related fields
	bool is_tracked;	
	dyn_var<VertexSubset> *output_queue;
// Constructors
	VertexData(std::string name): data(name.c_str()) {
		is_tracked = false;
		output_queue = nullptr;
	}
	VertexData(const VertexData& other): data(other.data) {
		// Copy over analysis related bits appropriately	
		is_tracked = other.is_tracked;
		output_queue = other.output_queue;
	}
// Operations
	VertexDataIndex<T> operator[] (dyn_var<int> index) {
		VertexDataIndex<T> ret (data, index);
		// Set the analysis bits accordingly		
		ret.is_tracked = is_tracked;
		ret.output_queue = output_queue;
		return ret;
	
	}
	VertexDataIndex<T> operator[] (const Vertex &index) {
		VertexDataIndex<T> ret (data, index.vid);
		// Set the analysis bits accordingly	
		ret.is_tracked = is_tracked;
		ret.output_queue = output_queue;
		ret.current_access = index.current_access;
		return ret;
	}
	VertexDataIndex<T> operator[] (const int& index) {
		VertexDataIndex<T> ret(data, index);
		ret.is_tracked = is_tracked;
		ret.output_queue = output_queue;
		ret.current_access = Vertex::access_type::CONSTANT;
		return ret;
	}
};



}

#endif
