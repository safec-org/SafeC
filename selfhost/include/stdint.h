// stdint.h — the fixed-width types are built into SafeC; limits only
#pragma once
#include <stddef.h>
#define INT8_MIN (-128)
#define INT8_MAX 127
#define UINT8_MAX 255
#define INT16_MIN (-32768)
#define INT16_MAX 32767
#define UINT16_MAX 65535
#define INT32_MIN (-2147483647 - 1)
#define INT32_MAX 2147483647
#define UINT32_MAX 4294967295U
#define INT64_MIN (-9223372036854775807L - 1L)
#define INT64_MAX 9223372036854775807L
#define UINT64_MAX 18446744073709551615UL
#define SIZE_MAX 18446744073709551615UL
