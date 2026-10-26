// parse.h — parser internals used by sema.
#pragma once
#include "scc.h"
extern int g_typename[NSTR_MAX];
extern long g_fold;
int const_fold(int e);
void kw_init();
