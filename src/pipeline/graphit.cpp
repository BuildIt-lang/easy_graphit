#include "pipeline/graphit.h"

namespace pipeline {

std::vector<block::decl_stmt::Ptr> gather_global_decls(block::block::Ptr ast) {	
	gather_declared_vars func_decl;
	ast->accept(&func_decl);
	
	gather_global_vars exts;
	exts.declared = func_decl.declared;
	ast->accept(&exts);
	
	std::vector<block::decl_stmt::Ptr> decls;
	
	for (auto var: exts.gathered) {
		if (var->var_name.find("graphit_runtime::") == 0)
			continue;
		block::decl_stmt::Ptr new_decl = std::make_shared<block::decl_stmt>();
		new_decl->decl_var = var;		
		new_decl->init_expr = nullptr;
		decls.push_back(new_decl);
	}
	return decls;
		
}
void gather_global_vars::visit(block::var_expr::Ptr expr) {
	block::var::Ptr var1 = expr->var1;
	if (std::find(gathered.begin(), gathered.end(), var1) == gathered.end() && std::find(declared.begin(), declared.end(), var1) == declared.end())
		gathered.push_back(var1);
}
	
void run_graphit_pipeline(block::block::Ptr ast, std::ostream& oss) {
	
	block::eliminate_redundant_vars(ast);
	std::vector<block::decl_stmt::Ptr> global_vars = gather_global_decls(ast);
			
	for (auto decl: global_vars) 
		block::c_code_generator::generate_code(decl, oss);
	block::block::Ptr kernel;
	while (kernel = pipeline::extract_single_cuda(block::to<block::func_decl>(ast)->body)) {
		block::c_code_generator::generate_code(kernel, oss);
	}
	block::c_code_generator::generate_code(ast, oss);
}
}
