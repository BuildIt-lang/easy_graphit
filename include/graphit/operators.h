#ifndef GRAPHIT_OPERATORS_H

#define GRAPHIT_OPERATORS_H

#include "graphit/graphit_types.h"

namespace graphit {

// The basic vertices.apply operator
typedef void (*vertexset_apply_udf_t) (dyn_var<int>);

void vertexset_apply(dyn_var<VertexSubset> set, vertexset_apply_udf_t);

}

#endif
