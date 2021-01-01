#include "pipeline/extract_cuda.h"
#include "builder/dyn_var.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

#include "graphit/operators.h"
#include "graphit/runtime.h"
#include "blocks/rce.h"
#include "pipeline/graphit.h"


using graphit::Vertex;
using graphit::VertexData;
using graphit::dyn_var;
using graphit::VertexSubset;
using graphit::GraphT;


VertexData<int> SP("SP");
dyn_var<GraphT> edges("edges");

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

static void testcase(dyn_var<char*> graph_name) {	
	graphit::current_context = graphit::context_type::HOST;
	edges = graphit::runtime::load_graph(graph_name);

	SP[0] = 0;

	dyn_var<VertexSubset> frontier = graphit::runtime::new_vertex_subset(edges.num_vertices);
	// Simple test for vertex set apply
	graphit::vertexset_apply(frontier, reset);

	// Simple test for edgset apply no output
	graphit::edgeset_apply_from(edges, frontier, update_func);

	dyn_var<VertexSubset> output = graphit::runtime::new_vertex_subset(edges.num_vertices);
	// Simple test for edgset apply output
	graphit::edgeset_apply_from_modified(edges, frontier, update_func_min, output, SP);

	graphit::edgeset_apply_from_modified(edges, frontier, update_func_min_src, output, SP);
		
}



int main(int argc, char * argv[]) {
	auto ast = builder::builder_context().extract_function_ast(testcase, "testcase");	
	pipeline::run_graphit_pipeline(ast, std::cout);	
	return 0;
}
