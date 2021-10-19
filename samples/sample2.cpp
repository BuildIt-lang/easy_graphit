#include "pipeline/extract_cuda.h"
#include "builder/dyn_var.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

#include "graphit/operators.h"



static void reset(graphit::Vertex vert) {
}

static void foo(void) {	
	graphit::dyn_var<graphit::VertexSubset> frontier;
	graphit::vertexset_apply(frontier, reset);
}


int main(int argc, char * argv[]) {
	builder::builder_context context;
	auto ast = context.extract_function_ast(foo, "foo");
	assert(block::isa<block::func_decl>(ast));
	
	block::block::Ptr kernel;
	std::vector<block::decl_stmt::Ptr> new_decls;
	while (kernel = pipeline::extract_single_cuda(block::to<block::func_decl>(ast)->body, new_decls)) {
		block::c_code_generator::generate_code(kernel, std::cout);
		new_decls.clear();
	}

	block::c_code_generator::generate_code(ast, std::cout);
	return 0;
}
