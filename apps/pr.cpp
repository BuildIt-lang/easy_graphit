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
using graphit::dyn_var;
using graphit::VertexSubset;
using graphit::GraphT;


VertexData<float> old_rank("old_rank");
VertexData<float> new_rank("new_rank");
VertexData<float> contrib("contrib");

dyn_var<int> update_outer;
dyn_var<int> update_inner;

const float damp = 0.85;


dyn_var<GraphT> edges("edges");

static void computeContrib(Vertex v) {
	contrib[v] = old_rank[v] / edges.out_degrees[(dyn_var<int>)v];
}

static void updateEdge(Vertex src, Vertex dst) {
	new_rank[dst] += contrib[src];
}

static void updateVertex(Vertex v) {
	dyn_var<float> beta_score = (1.0 - damp) / edges.num_vertices;
	new_rank[v] = beta_score + damp * (new_rank[v]);
	old_rank[v] = new_rank[v];
	new_rank[v] = 0.0;	
}

static void reset(Vertex v) {
	old_rank[v] = 1.0 / edges.num_vertices;
	new_rank[v] = 0.0;
}


static void testcase(dyn_var<char*> graph_name, graphit::Schedule &s1) {	
	graphit::current_context = graphit::context_type::HOST;
	edges = graphit::runtime::load_graph(graph_name);

	old_rank.allocate(edges.num_vertices);	
	new_rank.allocate(edges.num_vertices);	
	contrib.allocate(edges.num_vertices);	

	for (dyn_var<int> trial = 0; trial < 10; trial = trial + 1) {
		graphit::runtime::start_timer();
		graphit::vertexset_apply(edges, reset);
		for (dyn_var<int> i = 0; i < 20; i = i + 1) {
			graphit::vertexset_apply(edges, computeContrib);
			graphit::edgeset_apply(s1).apply(edges, updateEdge);
			graphit::vertexset_apply(edges, updateVertex);
		}
		dyn_var<float> t = graphit::runtime::stop_timer();
		graphit::runtime::print_time(t/20);
	}
			
}



int main(int argc, char * argv[]) {
	graphit::SimpleGPUSchedule::default_max_cta = 160;
	
	graphit::SimpleGPUSchedule s1;
	s1.configDirection(graphit::SimpleGPUSchedule::direction_type::PULL);
	s1.configLoadBalancing(graphit::SimpleGPUSchedule::load_balancing_type::EDGE_ONLY, 
		graphit::SimpleGPUSchedule::edge_blocking_type::BLOCKED, 0x42000);
		
	auto ast = builder::builder_context().extract_function_ast(testcase, "PR", s1);	
	pipeline::run_graphit_pipeline(ast, std::cout);	
	return 0;
}
