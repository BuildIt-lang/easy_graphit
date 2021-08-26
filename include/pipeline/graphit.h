#ifndef PIPELINE_GRAPHIT_H
#define PIPELINE_GRAPHIT_H

#include "pipeline/extract_cuda.h"
#include "builder/dyn_var.h"
#include "builder/builder_context.h"
#include "blocks/c_code_generator.h"

#include "graphit/operators.h"
#include "graphit/runtime.h"
#include "blocks/rce.h"

namespace pipeline {
extern void run_graphit_pipeline(block::block::Ptr ast, std::ostream &oss);
extern std::vector<block::decl_stmt::Ptr> gather_global_decls(block::block::Ptr);

class gather_global_vars: public block::block_visitor {
public:
	using block_visitor::visit;
	std::vector<block::var::Ptr> gathered;
	std::vector<block::var::Ptr> declared;
	virtual void visit(block::var_expr::Ptr);
};
void ignore_patchup(block::stmt::Ptr s);
void handle_atomic_patchups(block::block::Ptr ast);

}

#endif

