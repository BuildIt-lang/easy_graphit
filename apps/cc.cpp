#include "pipeline/extract_cuda.h"
#include "builder/dyn_var.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

#include "graphit/operators.h"
#include "graphit/runtime.h"
#include "blocks/rce.h"
#include "pipeline/graphit.h"
#include "graphit/schedule.h"


using graphit::Vertex;
using graphit::VertexData;
using graphit::VertexSubset;
using graphit::GraphT;

VertexData<int> ID("ID");
GraphT edges("edges");

VertexData<int> updates("update");

static void updateEdge(Vertex src, Vertex dst) {
	dyn_var<int> src_id = ID[src];
	dyn_var<int> dst_id = ID[dst];
	
	dyn_var<int> p_src_id = ID[src_id];
	dyn_var<int> p_dst_id = ID[dst_id];
	
	ID[src_id].min(p_dst_id);	
	ID[dst_id].min(p_src_id);	

	if (!(p_dst_id == ID[dst_id]) || !(p_src_id == ID[src_id]))
		if (updates[0] == 0)
			updates[0] = 1;
}
static void init(Vertex v) {
	ID[v] = v;
}

static void pjump(Vertex v) {
	dyn_var<int> y = ID[v];
	dyn_var<int> x = ID[y];
	if (x != y) {
		ID[v] = x;
		if (updates[1] == 0)
			updates[1] = 1;
	}
}



static void testcase(dyn_var<char*> graph_name, graphit::Schedule &s1) {	
	graphit::current_context = graphit::context_type::HOST;
	edges = graphit::runtime::load_graph(graph_name);

	ID.allocate(edges.num_vertices);	
	updates.allocate(2);

	for (dyn_var<int> trial = 0; trial < 10; trial = trial + 1) {
		graphit::runtime::start_timer();
		graphit::vertexset_apply(edges, init);
		updates[0] = 1;
		while (dyn_var<int>(updates[0])) {
			updates[0] = 0;
			graphit::edgeset_apply(s1).apply(edges, updateEdge);
			updates[1] = 1;
			while (dyn_var<int>(updates[1])) {
				updates[1] = 0;
				graphit::vertexset_apply(edges, pjump);
			}
		}	
		dyn_var<float> t = graphit::runtime::stop_timer();
		graphit::runtime::print_time(t);
	}
			
}



int main(int argc, char * argv[]) {
	
	graphit::SimpleGPUSchedule::default_max_cta = atoi(argv[1]);
	graphit::SimpleGPUSchedule::default_cta_size = atoi(argv[2]);
	
	graphit::SimpleGPUSchedule s1;
	s1.configDirection(graphit::SimpleGPUSchedule::direction_type::PUSH);
	s1.configLoadBalancing(graphit::SimpleGPUSchedule::load_balancing_type::EDGE_ONLY);
	//s1.configLoadBalancing(graphit::SimpleGPUSchedule::load_balancing_type::CM);
		
	auto ast = builder::builder_context().extract_function_ast(testcase, "CC", s1);	
	pipeline::run_graphit_pipeline(ast, std::cout);	
	return 0;
}
