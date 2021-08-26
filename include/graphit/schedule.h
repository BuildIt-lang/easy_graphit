#ifndef GRAPHIT_SCHEDULE_H
#define GRAPHIT_SCHEDULE_H

namespace graphit {

// Abstract schedule class
struct Schedule {
	virtual ~Schedule() = default;
};

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


}
#endif
