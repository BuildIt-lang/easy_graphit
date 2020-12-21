#ifndef GRAPHIT_TYPES_H
#define GRAPHIT_TYPES_H

#include "builder/dyn_var.h"
#include "builder/member_base.h"
#include "builder/builder.h"

namespace graphit {


template <int recur>
struct member;

#define MAX_MEMBER_DEPTH (3)
using gbuilder = builder::builder_final<member<MAX_MEMBER_DEPTH>>;

template <typename T>
using dyn_var = builder::dyn_var_final<T, member<MAX_MEMBER_DEPTH>, gbuilder>;

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





// Commonly used named types in GraphIt
#define GRAPH_T_NAME "graphit::GraphT"
extern const char graph_t_name[sizeof(GRAPH_T_NAME)];
using GraphT = typename builder::name<graph_t_name>;

#define VERTEXSUBSET_T_NAME "graphit::vertexsubset"
extern const char vertexsubset_t_name[sizeof(VERTEXSUBSET_T_NAME)];
using VertexSubset = typename builder::name<vertexsubset_t_name>;


}

#endif
