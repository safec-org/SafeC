// SafeC Standard Library — Boolean type (C99/C11/C17)
// Defines bool, true, and false.  SafeC uses int-sized booleans.
#pragma once

#ifndef __bool_true_false_are_defined

#define bool   int
#define true   1
#define false  0

#define __bool_true_false_are_defined  1

#endif
