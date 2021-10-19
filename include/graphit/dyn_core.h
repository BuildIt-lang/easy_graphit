#ifndef DYN_CORE_H
#define DYN_CORE_H
#include "builder/dyn_var.h"
#include "builder/builder.h"
namespace graphit {
template <int recur>
struct member;

#define MAX_MEMBER_DEPTH (1)
using gbuilder = builder::builder_base<member<MAX_MEMBER_DEPTH>>;

template <typename T>
using dyn_var = builder::dyn_var_base<T, member<MAX_MEMBER_DEPTH>, gbuilder>;
}
#endif
