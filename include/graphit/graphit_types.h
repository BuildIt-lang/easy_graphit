#ifndef GRAPHIT_TYPES_H
#define GRAPHIT_TYPES_H

#include "builder/dyn_var.h"
#include "builder/member_base.h"

namespace graphit {

class gbuilder;
template <typename T>
class dyn_var;

template <int recur>
struct member: public builder::member_base_impl<gbuilder> {
	using builder::member_base_impl<gbuilder>::member_base_impl;
	typedef member<recur-1> member_t;

	// All the members graphit types can have
	
	// GraphT related types
	member_t num_vertices = member_t(this, "num_vertices");
	member_t num_edges = member_t(this, "num_edges");


	// VertexSubset related types
	member_t num_elems = member_t(this, "num_elems");
	member_t d_sparse_queue = member_t(this, "d_sparse_queue");
		
};

template <>
struct member<0>: public builder::member_base {
	using builder::member_base::member_base;
};

#define MAX_MEMBER_DEPTH (3)
class gbuilder: public builder::builder_base<gbuilder>, public member<MAX_MEMBER_DEPTH> {
public:
	using builder::builder_base<gbuilder>::builder_base;
	gbuilder operator=(const gbuilder &a) {
		return assign(a);
	}
	using builder::builder_base<gbuilder>::operator[];
	
	virtual block::expr::Ptr get_parent() const {
		return block_expr;
	}
	
	gbuilder(const gbuilder &a): builder_base<gbuilder>(a), member<MAX_MEMBER_DEPTH>() {
		block_expr = a.block_expr;
	}
	
};

template <typename T>
class dyn_var: public builder::dyn_var_base<T, dyn_var<T>, gbuilder>, public member<MAX_MEMBER_DEPTH> {
public:
	virtual ~dyn_var() = default;
	using builder::dyn_var_base<T, dyn_var<T>, gbuilder>::dyn_var_base;
	using builder::dyn_var_base<T, dyn_var<T>, gbuilder>::operator[];
	using builder::dyn_var_base<T, dyn_var<T>, gbuilder>::operator=;
	
	gbuilder operator=(const dyn_var<T> &a) {
		return (*this = (gbuilder)a);
	}
	virtual block::expr::Ptr get_parent() const {
		return ((gbuilder) (*this)).get_parent();
	}
};


// Commonly used named types in GraphIt
#define GRAPH_T_NAME "graphit::GraphT"
extern const char graph_t_name[sizeof(GRAPH_T_NAME)];
using GraphT = typename builder::name<graph_t_name>;

#define VERTEXSUBSET_T_NAME "graphit::vertexsubset"
extern const char vertexsubset_t_name[sizeof(VERTEXSUBSET_T_NAME)];
using VertexSubset = typename builder::name<vertexsubset_t_name>;


}

#endif
