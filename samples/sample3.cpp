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


VertexData<int> SP("SP");
GraphT edges("edges");

static void reset(Vertex v) {
	SP[v] = 0;
}

static void update_func(Vertex src, Vertex dst) {
	SP[src] = SP[dst] + 1;
}

static void update_func_min(Vertex src, Vertex dst) {
	SP[dst].min(SP[src] + 1);
}
static void update_func_min_src(Vertex src, Vertex dst) {
	SP[src].min(SP[dst] + 1);
}

static void testcase(dyn_var<char*> graph_name, graphit::Schedule &s1) {	
	graphit::current_context = graphit::context_type::HOST;

	VertexSubset frontier = graphit::runtime::new_vertex_subset(edges.num_vertices);
	VertexSubset output = graphit::runtime::new_vertex_subset(edges.num_vertices);

	edges = graphit::runtime::load_graph(graph_name);

	SP[0] = 0;

	// Simple test for vertex set apply
	graphit::vertexset_apply(frontier, reset);

	// Simple test for edgset apply no output
	graphit::edgeset_apply().from(frontier).apply(edges, update_func);

	// Simple test for edgset apply output

	graphit::edgeset_apply().from(frontier).apply_modified(edges, output, SP, update_func_min);

	graphit::edgeset_apply().from(frontier).apply_modified(edges, output, SP, update_func_min_src);

	graphit::edgeset_apply(s1).from(frontier).apply_modified(edges, output, SP, update_func_min);		
}



int main(int argc, char * argv[]) {

	graphit::SimpleGPUSchedule s1;
	s1.configDirection(graphit::SimpleGPUSchedule::direction_type::PULL);
	
	auto ast = builder::builder_context().extract_function_ast(testcase, "testcase", s1);	
	pipeline::run_graphit_pipeline(ast, std::cout);	
	return 0;
}
