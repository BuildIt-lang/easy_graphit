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


VertexData<int> parent("parent");

dyn_var<GraphT> edges("edges");


static void updateEdge(Vertex src, Vertex dst) {
	parent[dst] = src;
}

static dyn_var<int> toFilter(Vertex v) {
	return parent[v] == -1;
}

static void reset(Vertex v) {
	parent[v] = -1;
}


static void testcase(dyn_var<char*> graph_name, dyn_var<int> src, graphit::Schedule &s1) {	

	graphit::current_context = graphit::context_type::HOST;
	edges = graphit::runtime::load_graph(graph_name);

	parent.allocate(edges.num_vertices);	
	
	for (dyn_var<int> trial = 0; trial < 10; trial = trial + 1) {
		graphit::runtime::start_timer();
		graphit::vertexset_apply(edges, reset);
		parent[src] = src;
		dyn_var<VertexSubset> frontier = graphit::runtime::new_vertex_subset(edges.num_vertices);
		frontier.addVertex(src);
		while(frontier.size() != 0) {
			graphit::edgeset_apply(s1).from(frontier).to(toFilter).apply_modified(edges, frontier, parent, updateEdge);
		}	
		dyn_var<float> t = graphit::runtime::stop_timer();
		graphit::runtime::print_time(t);
	}
			
}



int main(int argc, char * argv[]) {
	graphit::SimpleGPUSchedule::default_max_cta = 160;
	
	graphit::SimpleGPUSchedule s1;
	s1.configDirection(graphit::SimpleGPUSchedule::direction_type::PUSH);
	s1.configLoadBalancing(graphit::SimpleGPUSchedule::load_balancing_type::VERTEX_BASED);
		
	auto ast = builder::builder_context().extract_function_ast(testcase, "BFS", s1);	
	pipeline::run_graphit_pipeline(ast, std::cout);	
	return 0;
}
