// stddef.h — playground subset. size_t is 64-bit (SafeC's unsigned long),
// matching the native 64-bit targets programs are usually written for.
#pragma once
typedef unsigned long size_t;
typedef long ssize_t;
typedef long ptrdiff_t;
typedef unsigned long uintptr_t;
typedef long intptr_t;
#ifndef NULL
#define NULL 0
#endif
