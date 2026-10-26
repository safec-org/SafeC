// assert.h — playground subset
#pragma once
extern void __rt_assert_fail(const char* expr, const char* file, int line);
#define assert(e) ((e) ? (void)0 : __rt_assert_fail(#e, __FILE__, __LINE__))
