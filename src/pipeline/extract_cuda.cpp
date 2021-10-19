#include "pipeline/extract_cuda.h"
#include "builder/dyn_var.h"
#include <sstream>
#include "blocks/c_code_generator.h"


using block::isa;
using block::to;

namespace pipeline {

int total_created_kernels = 0;


struct parent_finder: public block::block_visitor {
public:
	using block::block_visitor::visit;
	block::stmt::Ptr to_find;
	block::stmt_block::Ptr parent_found = nullptr;
	virtual void visit(block::stmt_block::Ptr block) {
		block::block_visitor::visit(block);
		for (auto stmt: block->stmts) {
			if (to_find == stmt) {
				parent_found = block;
			}
		}
	}
};


block::block::Ptr extract_single_cuda(block::block::Ptr from, std::vector<block::decl_stmt::Ptr> &new_decls) {
	if (!isa<block::stmt_block>(from)) {
		std::cerr << "extract_single_cuda() expects a stmt_block" << std::endl;
		return nullptr;
	}

	int is_coop = 0;
	block::stmt::Ptr found_loop = block::annotation_finder::find_annotation(from, CUDA_ANNOTATION_STRING);
	if (found_loop == nullptr) {
		found_loop = block::annotation_finder::find_annotation(from, CUDA_COOP_ANNOTATION_STRING);
		if (found_loop == nullptr) {
			return nullptr;
		}
		is_coop = 1;
	}

	// First we assert that the stmt we have found is a for loop
	assert(isa<block::for_stmt>(found_loop) && "The " CUDA_ANNOTATION_STRING " annotation should be only applied to a for loop");
	// We will also assert that this for loop has a single other for loop
	block::for_stmt::Ptr outer_loop = to<block::for_stmt>(found_loop);
	assert(isa<block::stmt_block>(outer_loop->body) && to<block::stmt_block>(outer_loop->body)->stmts.size() == 1 && isa<block::for_stmt>(to<block::stmt_block>(outer_loop->body)->stmts[0]) &&  "Loops for device should be doubly nested");
	block::for_stmt::Ptr inner_loop = to<block::for_stmt>(to<block::stmt_block>(outer_loop->body)->stmts[0]);
	
	block::var::Ptr outer_var;
	if (isa<block::decl_stmt>(outer_loop->decl_stmt)) {
		outer_var = to<block::decl_stmt>(outer_loop->decl_stmt)->decl_var;
	} else {
		// If this is not a decl, it must be an assign
		outer_var = to<block::var_expr>(to<block::assign_expr>(to<block::expr_stmt>(outer_loop->decl_stmt)->expr1)->var1)->var1;
	}
	block::var::Ptr inner_var;
	if (isa<block::decl_stmt>(inner_loop->decl_stmt)) {
		inner_var = to<block::decl_stmt>(inner_loop->decl_stmt)->decl_var;
	} else {
		// If this is not a decl, it must be an assign
		inner_var = to<block::var_expr>(to<block::assign_expr>(to<block::expr_stmt>(inner_loop->decl_stmt)->expr1)->var1)->var1;
	}
	std::vector<block::var::Ptr> vars = extract_extern_vars(from, inner_loop->body, outer_var, inner_var);

	int this_kern_index = total_created_kernels;
	total_created_kernels++;

	if (is_coop) {
		// If this is coop, we will create some extra decls to return the copied values
		int i = 0;
		for (auto v: vars) {
			auto v_new = std::make_shared<block::var>();
			v_new->var_type = v->var_type;
			v_new->var_name = "ret_" + std::to_string(this_kern_index) + "_" + std::to_string(i);
			i++;
			v_new->setMetadata<std::vector<std::string>>("attributes", {"__device__"});

			auto decl_new = std::make_shared<block::decl_stmt>();
			decl_new->decl_var = v_new;
			decl_new->init_expr = nullptr;
			new_decls.push_back(decl_new);	
		}
	}

	assert(isa<block::lt_expr>(outer_loop->cond) && "CUDA loops should have condition of the form < ...");
	assert(isa<block::lt_expr>(inner_loop->cond) && "CUDA loops should have condition of the form < ...");
	
	block::expr::Ptr cta_count = to<block::lt_expr>(outer_loop->cond)->expr2;
	block::expr::Ptr thread_count = to<block::lt_expr>(inner_loop->cond)->expr2;
	
	
	block::var::Ptr cta_id = std::make_shared<block::var>();
	cta_id->var_name = "blockIdx.x";
	cta_id->var_type = builder::dyn_var<int>::create_block_type();
	
	block::var::Ptr thread_id = std::make_shared<block::var>();
	thread_id->var_name = "threadIdx.x";
	thread_id->var_type = builder::dyn_var<int>::create_block_type();
	
	var_replace_all(inner_loop->body, outer_var, cta_id);
	var_replace_all(inner_loop->body, inner_var, thread_id);

	block::func_decl::Ptr kernel = std::make_shared<block::func_decl>();
	kernel->func_name = "cuda_kernel_" + std::to_string(this_kern_index);

	kernel->return_type = builder::dyn_var<void>::create_block_type();
	

	block::function_call_expr::Ptr call = std::make_shared<block::function_call_expr>();
	block::var::Ptr call_name = std::make_shared<block::var>();
	if (!is_coop) {
		call_name->var_type = builder::dyn_var<int>::create_block_type();
		call_name->var_name = kernel->func_name;
		call_name->var_name += "<<<";

		std::ostringstream cta_count_str, thread_count_str;
		block::c_code_generator::generate_code(cta_count, cta_count_str, 0);
		block::c_code_generator::generate_code(thread_count, thread_count_str, 0);

		call_name->var_name += cta_count_str.str();
		// There is new line at the end - always
		call_name->var_name.pop_back();
		call_name->var_name += ", " + thread_count_str.str();
		call_name->var_name.pop_back();
		call_name->var_name += ">>>";
		
	} else {
		call_name->var_type = builder::dyn_var<int>::create_block_type();
		call_name->var_name = "graphit_runtime::LaunchCooperativeKernel";
		block::var_expr::Ptr param1 = std::make_shared<block::var_expr>();
		block::var::Ptr param1_var = std::make_shared<block::var>();
		param1_var->var_type = builder::dyn_var<int>::create_block_type();
		param1_var->var_name = "(void*)" + kernel->func_name;
		param1->var1 = param1_var;
		call->args.push_back(param1);
		call->args.push_back(cta_count);
		call->args.push_back(thread_count);
	}
	block::var_expr::Ptr call_var_expr = std::make_shared<block::var_expr>();
	call_var_expr->var1 = call_name;
	call->expr1 = call_var_expr;
	block::expr_stmt::Ptr call_stmt = std::make_shared<block::expr_stmt>();
	call_stmt->expr1 = call;

	block::function_call_expr::Ptr call_sync = std::make_shared<block::function_call_expr>();
	block::var::Ptr call_name_sync = std::make_shared<block::var>();
	call_name_sync->var_type = builder::dyn_var<int>::create_block_type();
	call_name_sync->var_name = "cudaDeviceSynchronize";
	block::var_expr::Ptr call_var_expr_sync = std::make_shared<block::var_expr>();
	call_var_expr_sync->var1 = call_name_sync;
	call_sync->expr1 = call_var_expr_sync;
	block::expr_stmt::Ptr call_stmt_sync = std::make_shared<block::expr_stmt>();
	call_stmt_sync->expr1 = call_sync;

	for (unsigned int i = 0; i < vars.size(); i++) {
		std::string arg_name = "arg" + std::to_string(i);	
		block::var::Ptr arg = std::make_shared<block::var>();
		arg->var_name = arg_name;
		arg->var_type = vars[i]->var_type;
		var_replace_all(inner_loop->body, vars[i], arg);		
		kernel->args.push_back(arg);
		block::var_expr::Ptr arg_expr = std::make_shared<block::var_expr>();
		arg_expr->var1 = vars[i];
		call->args.push_back(arg_expr);
	}
	block::stmt_block::Ptr new_stmts = std::make_shared<block::stmt_block>();
	
	//block::stmt_block::Ptr old_stmts = to<block::stmt_block>(from);	
	parent_finder finder;
	finder.to_find = found_loop;
	from->accept(&finder);
	block::stmt_block::Ptr old_stmts = finder.parent_found;
	
	kernel->body = inner_loop->body;

	// If this is a coop kernel, return the values
	std::vector<block::stmt::Ptr> copy_backs;
	if (is_coop) {
		auto if_s = std::make_shared<block::if_stmt>();
		auto var = std::make_shared<block::var>();
		var->var_type = builder::dyn_var<int>::create_block_type();
		var->var_name = "!(blockIdx.x * blockDim.x + threadIdx.x)";
		auto var_expr = std::make_shared<block::var_expr>();
		var_expr->var1 = var;
		if_s->cond = var_expr;
		if_s->then_stmt = std::make_shared<block::stmt_block>();
		if_s->else_stmt = std::make_shared<block::stmt_block>();
		// Add an assignment for each variable
		int i = 0;
		auto v_copy = std::make_shared<block::var>();
		v_copy->var_type = builder::dyn_var<void(void)>::create_block_type();
		v_copy->var_name = "graphit_runtime::cudaMemcpyFromSymbolMagic";
		auto v_copy_expr = std::make_shared<block::var_expr>();
		v_copy_expr->var1 = v_copy;
		

		for (auto v: kernel->args) {
			auto rhs = std::make_shared<block::var_expr>();
			rhs->var1 = v;
			auto lhs = std::make_shared<block::var_expr>();
			lhs->var1 = new_decls[i]->decl_var;	
			auto assign_expr = std::make_shared<block::assign_expr>();
			assign_expr->var1 = lhs;
			assign_expr->expr1 = rhs;
			auto expr_stmt = std::make_shared<block::expr_stmt>();
			expr_stmt->expr1 = assign_expr;
			block::to<block::stmt_block>(if_s->then_stmt)->stmts.push_back(expr_stmt);

			// Also create a copy back
			auto f = std::make_shared<block::function_call_expr>();
			f->expr1 = v_copy_expr;
			auto addr = std::make_shared<block::addr_of_expr>();
			addr->expr1 = call->args[i+3];
			f->args.push_back(addr);
			f->args.push_back(lhs);
			
			auto stmt = std::make_shared<block::expr_stmt>();
			stmt->expr1 = f;
			copy_backs.push_back(stmt);

			i++;
		}
		to<block::stmt_block>(kernel->body)->stmts.push_back(if_s);
	}
	
	for (auto stmt: old_stmts->stmts) {
		if (stmt != found_loop)
			new_stmts->stmts.push_back(stmt);
		else {
			new_stmts->stmts.push_back(call_stmt);
			new_stmts->stmts.push_back(call_stmt_sync);
			for (auto a: copy_backs)
				new_stmts->stmts.push_back(a);
		}
	}

	old_stmts->stmts = new_stmts->stmts;	
	
	std::vector<std::string> attrs;
	attrs.push_back("__global__");
			
	kernel->setMetadata("attributes", attrs);
	
	return kernel;
		
}

void var_replace_all(block::stmt::Ptr body, block::var::Ptr from, block::var::Ptr to) {
	var_replacer replacer;
	replacer.to_replace = from;
	replacer.replace_with = to;
	body->accept(&replacer);	
}

void var_replacer::visit(block::var_expr::Ptr expr) {
	if (expr->var1 == to_replace)
		expr->var1 = replace_with;
}

void gather_declared_vars::visit(block::decl_stmt::Ptr stmt) {
	block::var::Ptr var1 = stmt->decl_var;
	if (std::find(declared.begin(), declared.end(), var1) == declared.end())
		declared.push_back(var1);
}

void gather_declared_vars::visit(block::func_decl::Ptr stmt) {
	stmt->return_type->accept(this);
	for (auto arg : stmt->args) {
		if (std::find(declared.begin(), declared.end(), arg) == declared.end())
			declared.push_back(arg);
	}
	stmt->body->accept(this);
}

void gather_extern_vars::visit(block::var_expr::Ptr expr) {
	block::var::Ptr var1 = expr->var1;
	// We want to ignore all the variables with function types
	if (block::isa<block::function_type>(var1->var_type))
		return;
	if (std::find(gathered.begin(), gathered.end(), var1) == gathered.end() && std::find(declared.begin(), declared.end(), var1) == declared.end())
		gathered.push_back(var1);
}

std::vector<block::var::Ptr> extract_extern_vars(block::block::Ptr function, block::stmt::Ptr from, block::var::Ptr outer, block::var::Ptr inner) {
	// This visitor finds the list of the variables that are declared inside the function
	// This is done to ignore the globals
	// Not being used right now, we want to hoist the globals too
	gather_declared_vars func_dec;
	function->accept(&func_dec);

	// This identifies the variables that are declared inside the block
	gather_declared_vars dec;
	from->accept(&dec);

	// This finally identifies the variables that are used inside the loop nest, declared in the function
	// But not declared inside the block
	gather_extern_vars exts;
	exts.declared = dec.declared;
	exts.declared.push_back(outer);
	exts.declared.push_back(inner);
	//exts.func_declared = func_dec.declared;
	
	from->accept(&exts);

	return exts.gathered;
	
}



}
