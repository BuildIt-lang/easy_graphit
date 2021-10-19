#include "pipeline/extract_cuda.h"
#include "builder/dyn_var.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

using builder::dyn_var;



static void foo(void) {	
	dyn_var<int> total = 1024;
	builder::annotate(CUDA_ANNOTATION_STRING);
	for(dyn_var<int> cta = 0; cta < 25; cta = cta+1) {
		for (dyn_var<int> thread = 0; thread < 512; thread = thread+1) {
			dyn_var<int> thread_id = total - (cta * 512 + thread);
		}
	}
}


int main(int argc, char * argv[]) {
	builder::builder_context context;
	auto ast = context.extract_function_ast(foo, "foo");
	assert(block::isa<block::func_decl>(ast));
	std::vector<block::decl_stmt::Ptr> new_decls;
	auto ast2 = pipeline::extract_single_cuda(block::to<block::func_decl>(ast)->body, new_decls);
	ast->dump(std::cout, 0);
	if (ast2 != nullptr) {
		ast2->dump(std::cout, 0);
		block::c_code_generator::generate_code(ast2, std::cout);
	}	
	block::c_code_generator::generate_code(ast, std::cout);
	return 0;
}
