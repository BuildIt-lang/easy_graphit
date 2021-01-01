#ifndef GRAPHIT_EXTRACT_CUDA_H
#define GRAPHIT_EXTRACT_CUDA_H
#include <iostream>
#include "blocks/block_visitor.h"
#include "blocks/block.h"
#include "blocks/expr.h"
#include "blocks/stmt.h"
#include "blocks/annotation_finder.h"
#include <vector>
#include <algorithm>

#define CUDA_ANNOTATION_STRING "run_on_device"

namespace pipeline {

block::block::Ptr extract_single_cuda(block::block::Ptr from);

std::vector<block::var::Ptr> extract_extern_vars(block::block::Ptr function, block::stmt::Ptr from, block::var::Ptr, block::var::Ptr);

extern int total_created_kernels;

class gather_extern_vars: public block::block_visitor {
public:
	using block_visitor::visit;
	std::vector<block::var::Ptr> gathered;
	std::vector<block::var::Ptr> declared;
	std::vector<block::var::Ptr> func_declared;
	virtual void visit(block::var_expr::Ptr);
};


class gather_declared_vars: public block::block_visitor {
public:
	using block_visitor::visit;
	std::vector<block::var::Ptr> declared;
	virtual void visit(block::decl_stmt::Ptr);
	virtual void visit(block::func_decl::Ptr);
};


class var_replacer: public block::block_visitor {
public: 
	using block_visitor::visit;
	block::var::Ptr to_replace;
	block::var::Ptr replace_with;
	
	virtual void visit(block::var_expr::Ptr);	
};

void var_replace_all(block::stmt::Ptr, block::var::Ptr from, block::var::Ptr to);


}



#endif
