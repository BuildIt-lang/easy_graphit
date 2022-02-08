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
using graphit::FrontierList;


VertexData<double> num_paths("num_paths");
VertexData<float> dependences("dependences");
VertexData<unsigned char> visited("visited");


GraphT edges("edges");


static void reset(Vertex v) {
	dependences[v] = 0.0;
	num_paths[v] = 0.0;
	visited[v] = 0;
}

static void forward_update(Vertex src, Vertex dst) {
	num_paths[dst] += num_paths[src];
}

static dyn_var<int> visited_filter(Vertex v) {
	return visited[v] == false;
}

static void mark_visited(Vertex v) {
	visited[v] = true;
}
static void mark_unvisited(Vertex v) {
	visited[v] = false;
}

static void backward_vertex_f(Vertex v) {
	visited[v] = true;
	dependences[v] += 1.0 / num_paths[v];
}

static void backward_update(Vertex src, Vertex dst) {
	dependences[dst] += dependences[src];
}

static void final_vertex_f(Vertex v) {
	if (!(num_paths[v] == 0)) 
		dependences[v] = (dependences[v] - 1.0 / num_paths[v]) * num_paths[v];
	else
		dependences[v] = 0;
}
static void testcase(dyn_var<char*> graph_name, dyn_var<int> src, dyn_var<float> t, graphit::Schedule &s1,
	const bool to_fuse) {

	if (graphit::s_isa<graphit::HybridGPUSchedule>(&s1)) 
		graphit::s_to<graphit::HybridGPUSchedule>(&s1)->bindThreshold(t);

	graphit::current_context = graphit::context_type::HOST;
	edges = graphit::runtime::load_graph(graph_name);
	GraphT t_edges = edges.get_transposed_graph()[0];
	
	num_paths.allocate(edges.num_vertices);
	dependences.allocate(edges.num_vertices);
	visited.allocate(edges.num_vertices);
	
	for (dyn_var<int> trial = 0; trial < 3; trial = trial + 1) {
		VertexSubset frontier = graphit::runtime::new_vertex_subset(edges.num_vertices);

		graphit::runtime::start_timer();
		graphit::vertexset_apply(edges, reset);

		num_paths[src] = 1;
		visited[src] = 1;
		frontier.addVertex(src);

		dyn_var<int> round = 0;
		FrontierList frontier_list = graphit::runtime::new_frontier_list(edges.num_vertices);	
		frontier_list.insert(frontier);
		
		graphit::fuse_kernel(to_fuse, [&]() {
			while(frontier.size() != 0) {
				round = round + 1;
				graphit::edgeset_apply(s1).from(frontier).to(visited_filter).apply_modified(edges, frontier, num_paths, forward_update, false);
				graphit::vertexset_apply(frontier, mark_visited);
				frontier_list.insert(frontier);
			}	
		});

		graphit::vertexset_apply(edges, mark_unvisited);	
		frontier_list.retrieve(frontier);
		frontier_list.retrieve(frontier);
		graphit::vertexset_apply(frontier, backward_vertex_f);
		round = round - 1;
		
		graphit::fuse_kernel(to_fuse, [&]() {
			while(round > 0) {
				graphit::edgeset_apply(s1).from(frontier).to(visited_filter).apply(edges, backward_update);
				frontier_list.retrieve(frontier);	
				graphit::vertexset_apply(frontier, backward_vertex_f);
				round = round - 1;
			}
		});


		graphit::vertexset_apply(edges, final_vertex_f);	

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

		graphit::SimpleGPUSchedule s2;
		s2.configDirection(graphit::SimpleGPUSchedule::direction_type::PULL, graphit::SimpleGPUSchedule::pull_frontier_rep_type::BITMAP);
		//s2.configLoadBalancing(graphit::SimpleGPUSchedule::load_balancing_type::VERTEX_BASED);
		s2.configLoadBalancing(graphit::SimpleGPUSchedule::load_balancing_type::TWCE);
		s2.configFrontierCreation(graphit::SimpleGPUSchedule::frontier_creation_type::BITMAP);
			
		graphit::HybridGPUSchedule h1 (s1, s2);

		auto ast = builder::builder_context().extract_function_ast(testcase, "BC", h1, false);
		pipeline::run_graphit_pipeline(ast, std::cout);	
	} else {

		graphit::SimpleGPUSchedule s1;
		s1.configDirection(graphit::SimpleGPUSchedule::direction_type::PUSH);
		s1.configLoadBalancing(graphit::SimpleGPUSchedule::load_balancing_type::TWCE);

		auto ast = builder::builder_context().extract_function_ast(testcase, "BC", s1, true);
		pipeline::run_graphit_pipeline(ast, std::cout);	
		
	}
	return 0;
}
