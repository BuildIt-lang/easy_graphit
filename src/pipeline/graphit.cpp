#include "pipeline/graphit.h"
#include <sstream>

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
struct parent_if_finder: public block::block_visitor {
public:
	using block::block_visitor::visit;
	block::stmt::Ptr to_find;
	block::if_stmt::Ptr parent_found = nullptr;
	virtual void visit(block::if_stmt::Ptr if_s) {
		if (block::isa<block::stmt_block>(if_s->then_stmt)) {
			block::stmt_block::Ptr block = block::to<block::stmt_block>(if_s->then_stmt);
			if (block->stmts[0] == to_find) {
				parent_found = if_s;
				return;
			}
		}
		block::block_visitor::visit(if_s);
	}
};

void ignore_patchup(block::stmt::Ptr s) {
	s->annotation += ":cannot_fix";
}

void handle_atomic_patchups(block::block::Ptr ast) {
	while (1) {	
		block::stmt::Ptr needs_atomic = block::annotation_finder::find_annotation(ast, NEEDS_ATOMICS_ANNOTATION);
		if(needs_atomic == nullptr)
			break;	
		
		if (!block::isa<block::expr_stmt>(needs_atomic)) {
			ignore_patchup(needs_atomic);
			continue;
		}	
		block::expr_stmt::Ptr e = block::to<block::expr_stmt>(needs_atomic);
		if (!block::isa<block::assign_expr>(e->expr1)) {
			ignore_patchup(needs_atomic);
			continue;
		}
		block::expr::Ptr lhs1 = block::to<block::assign_expr>(e->expr1)->var1;
		block::expr::Ptr rhs1 = block::to<block::assign_expr>(e->expr1)->expr1;
		
		parent_if_finder finder;
		finder.to_find = needs_atomic;
		ast->accept(&finder);
		block::if_stmt::Ptr if_s = finder.parent_found;
                	
		if (!if_s || !block::isa<block::equals_expr>(if_s->cond)) {
			ignore_patchup(needs_atomic);
			continue;
		}
		block::expr::Ptr lhs2 = block::to<block::equals_expr>(if_s->cond)->expr1;	
		block::expr::Ptr rhs2 = block::to<block::equals_expr>(if_s->cond)->expr2;
	
		// Let us now serialize both lhs and check
		// There has to be a better way to compare equivalence of two expressions
		// But this works for now
		std::ostringstream str1, str2;
		block::c_code_generator::generate_code(lhs1, str1, 0);
		block::c_code_generator::generate_code(lhs2, str2, 0);
		if (str1.str() != str2.str()) {
			ignore_patchup(needs_atomic);
			continue;
		}
		
		// Everything works, let us create a compare and swap and patch it in
		block::function_call_expr::Ptr f = std::make_shared<block::function_call_expr>();
		block::var::Ptr v = std::make_shared<block::var>();
		v->var_type = builder::dyn_var<int(int*, int, int)>::create_block_type();
		v->var_name = "graphit_runtime::CAS";
		block::var_expr::Ptr ve = std::make_shared<block::var_expr>();		
		ve->var1 = v;
		f->expr1 = ve;
		
		block::addr_of_expr::Ptr adr = std::make_shared<block::addr_of_expr>();		
		adr->expr1 = lhs1;
		f->args.push_back(adr);
		f->args.push_back(rhs2);
		f->args.push_back(rhs1);	
		
		if_s->cond = f;
		
		// Now fix all the statements;
		std::vector<block::stmt::Ptr> &thens = block::to<block::stmt_block>(if_s->then_stmt)->stmts;
		thens.erase(thens.begin());
		
	}	
}
	
void run_graphit_pipeline(block::block::Ptr ast, std::ostream& oss) {

	// Before we do anything, let us include the headers
	oss << "#define NUM_CTA (" << graphit::SimpleGPUSchedule::default_max_cta << ")" << std::endl;
	oss << "#define CTA_SIZE (" << graphit::SimpleGPUSchedule::default_cta_size << ")" << std::endl;
	oss << "#include \"gpu_intrinsics.h\"" << std::endl;
	oss << "#include <cooperative_groups.h>" << std::endl;

	block::eliminate_redundant_vars(ast);
	handle_atomic_patchups(ast);	

	std::vector<block::decl_stmt::Ptr> global_vars = gather_global_decls(ast);
			
	for (auto decl: global_vars) 
		block::c_code_generator::generate_code(decl, oss);

	block::block::Ptr kernel;
	std::vector<block::decl_stmt::Ptr> new_decls;
	while (kernel = pipeline::extract_single_cuda(block::to<block::func_decl>(ast)->body, new_decls)) {
		for (auto d: new_decls) {
			block::c_code_generator::generate_code(d, oss);
		}
		block::c_code_generator::generate_code(kernel, oss);
		new_decls.clear();
	}

	block::c_code_generator::generate_code(ast, oss);
}
}
