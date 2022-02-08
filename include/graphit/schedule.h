#ifndef GRAPHIT_SCHEDULE_H
#define GRAPHIT_SCHEDULE_H
#include "builder/dyn_var.h"
#include "builder/static_var.h"

using builder::dyn_var;
using builder::as_member_of;

namespace graphit {

// Abstract schedule class
struct Schedule {
	virtual ~Schedule() = default;
};

template <typename T>
bool s_isa(Schedule* s) {
	return dynamic_cast<T*>(s) != nullptr;
}
template <typename T>
T* s_to(Schedule *s) {
	return dynamic_cast<T*>(s);
}

// Simple GPU Schedule class
struct SimpleGPUSchedule: public Schedule{
	static int default_max_cta;
	static int default_cta_size;

	enum class direction_type {
		PUSH, 
		PULL
	};
	enum class pull_frontier_rep_type {
		BITMAP, 
		BOOLMAP
	};
	enum class frontier_creation_type {
		SPARSE,
		BITMAP,
		BOOLMAP
	};

	enum class deduplication_type {
		DISABLED,
		ENABLED
	};
	enum class deduplication_strategy_type {
		FUSED,
		UNFUSED
	};

	enum class load_balancing_type {
		VERTEX_BASED,	
		TWC, 
		TWCE, 
		WM, 
		CM, 
		STRICT,
		EDGE_ONLY
	};

	enum class edge_blocking_type {
		BLOCKED,
		UNBLOCKED
	};

	enum class kernel_fusion_type {
		DISABLED,
		ENABLED
	};

	
	direction_type direction;
	pull_frontier_rep_type pull_frontier_rep;
	frontier_creation_type frontier_creation;
	deduplication_type deduplication;
	deduplication_strategy_type deduplication_stategy;
	load_balancing_type load_balancing;
	edge_blocking_type edge_blocking;
	int block_size;
	kernel_fusion_type kernel_fusion;
	
	int max_cta;
	int cta_size;
	
	SimpleGPUSchedule() {
		direction = direction_type::PUSH;
		load_balancing = load_balancing_type::VERTEX_BASED;
		edge_blocking = edge_blocking_type::UNBLOCKED;
		frontier_creation = frontier_creation_type::SPARSE;
		pull_frontier_rep = pull_frontier_rep_type::BOOLMAP;	

		block_size = -1;
		max_cta = default_max_cta;
		cta_size = default_cta_size;
	}

	void configDirection(direction_type dir, pull_frontier_rep_type pf = pull_frontier_rep_type::BOOLMAP) {
		direction = dir;
		if (direction == direction_type::PULL) {
			pull_frontier_rep = pf;
		}
	}
	void configFrontierCreation(frontier_creation_type fc) {
		frontier_creation = fc;
	}
	void configDeduplication(deduplication_type dedup) {
		deduplication = dedup;
	}
	void configLoadBalancing(load_balancing_type lb, edge_blocking_type eb = edge_blocking_type::UNBLOCKED, int bs = -1) {
		load_balancing = lb;
		if (lb == load_balancing_type::EDGE_ONLY) {
			edge_blocking = eb;
			block_size = bs;	
		}
	}
};


// Simple GPU Schedule class
struct HybridGPUSchedule: public Schedule{
public:
	Schedule* s1;
	Schedule* s2;

	// The threshold is a dynamic paramter to allow for runtime scheduling
	dyn_var<float> *threshold = nullptr;
	float static_threshold;
	HybridGPUSchedule(Schedule &_s1, Schedule &_s2) {
		s1 = &_s1;
		s2 = &_s2;
	}
	void configThreshold(float t) {
		threshold = nullptr;
		static_threshold = t;
	}
	void bindThreshold(dyn_var<float>&);
};

}
#endif
