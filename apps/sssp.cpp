#include "pipeline/extract_cuda.h"
#include "builder/dyn_var.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

#include "graphit/operators.h"
#include "graphit/runtime.h"
#include "blocks/rce.h"
#include "pipeline/graphit.h"
#include "graphit/schedule.h"
#include <climits>

using graphit::Vertex;
using graphit::VertexData;
using graphit::VertexSubset;
using graphit::GraphT;
using graphit::PriorityQueue;


VertexData<int> SP("SP");
GraphT edges("edges");
PriorityQueue pq("pq");

static void updateEdge(Vertex src, Vertex dst, dyn_var<int> weight) {
	dyn_var<int> new_dst = SP[src] + weight;	
	pq.updatePriorityMin((dyn_var<int>)dst, new_dst);
}

static void reset(Vertex v) {
	SP[v] = INT_MAX;
}


static void testcase(dyn_var<char*> graph_name, dyn_var<int> src, dyn_var<int> delta, graphit::Schedule &s1,
	const bool to_fuse) {
	graphit::current_context = graphit::context_type::HOST;
	edges = graphit::runtime::load_graph(graph_name);

	SP.allocate(edges.num_vertices);	
	
	for (dyn_var<int> trial = 0; trial < 5; trial = trial + 1) {
		pq.init(edges, SP, 0, delta, src);
		graphit::runtime::start_timer();
		graphit::vertexset_apply(edges, reset);
		SP[src] = 0;

		graphit::fuse_kernel(to_fuse, [&]() {
			while(pq.finished() == 0) {
				VertexSubset frontier = pq.dequeue();
				graphit::edgeset_apply(s1).from(frontier).apply_priority(edges, updateEdge, pq);
			}	
		});
		dyn_var<float> t = graphit::runtime::stop_timer();
		graphit::runtime::print_time(t);
	}
			
}



int main(int argc, char * argv[]) {

	graphit::SimpleGPUSchedule::default_max_cta = atoi(argv[1]);
	graphit::SimpleGPUSchedule::default_cta_size = atoi(argv[2]);
	if (argc > 3 && std::string(argv[3]) == "power") {
		
		graphit::SimpleGPUSchedule s1;
		s1.configDirection(graphit::SimpleGPUSchedule::direction_type::PUSH);
		s1.configLoadBalancing(graphit::SimpleGPUSchedule::load_balancing_type::TWCE);
		s1.configFrontierCreation(graphit::SimpleGPUSchedule::frontier_creation_type::BOOLMAP);


		auto ast = builder::builder_context().extract_function_ast(testcase, "SSSP", s1, false);
		pipeline::run_graphit_pipeline(ast, std::cout);	
	} else {

		graphit::SimpleGPUSchedule s1;
		s1.configDirection(graphit::SimpleGPUSchedule::direction_type::PUSH);
		s1.configFrontierCreation(graphit::SimpleGPUSchedule::frontier_creation_type::SPARSE);
		auto ast = builder::builder_context().extract_function_ast(testcase, "SSSP", s1, true);
		pipeline::run_graphit_pipeline(ast, std::cout);	
			
	}
	return 0;
}
