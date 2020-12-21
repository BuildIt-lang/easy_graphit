#include "pipeline/extract_cuda.h"
#include "builder/dyn_var.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

#include "graphit/operators.h"



static void reset(graphit::dyn_var<int> vert) {
	vert = vert + 1;
}

static void foo(void) {	
	graphit::vertexset_apply(0, reset);
}


int main(int argc, char * argv[]) {
	builder::builder_context context;
	auto ast = context.extract_function_ast(foo, "foo");
	assert(block::isa<block::func_decl>(ast));
	
	block::block::Ptr kernel;
	while (kernel = pipeline::extract_single_cuda(block::to<block::func_decl>(ast)->body)) {
		block::c_code_generator::generate_code(kernel, std::cout);
	}

	block::c_code_generator::generate_code(ast, std::cout);
	return 0;
}
