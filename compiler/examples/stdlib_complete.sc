// SafeC — Comprehensive standard library test
// Exercises stdint, stddef, stdbool, limits, float, inttypes,
// ctype, assert, stdckdint, and the expanded io/sys/str APIs.
// Compile: ./build/safec examples/stdlib_complete.sc --emit-llvm -o /tmp/sc.ll -I ../std
// Link:    clang -isysroot $(xcrun --show-sdk-path) /tmp/sc.ll -o /tmp/sc && /tmp/sc

#include "stdint.h"
#include "limits.h"
#include "float.h"
#include "stdbool.h"
#include "inttypes.h"
#include "ctype.h"
#include "assert.h"
#include "stdckdint.h"
#include "io.h"
#include "sys.h"
#include "str.h"
#include "mem.h"

extern int printf(char* fmt, ...);

int main() {
    // ── stdint: limits are correct ────────────────────────────────────────────
    unsafe {
        printf("=== stdint limits ===\n");
        printf("INT8:  [%d, %d]\n",   INT8_MIN,  INT8_MAX);
        printf("INT16: [%d, %d]\n",   INT16_MIN, INT16_MAX);
        printf("INT32: [%d, %d]\n",   INT32_MIN, INT32_MAX);
        printf("INT64_MAX=%lld\n",    INT64_MAX);
        printf("UINT8_MAX=%u  UINT16_MAX=%u  UINT32_MAX=%u\n",
               UINT8_MAX, UINT16_MAX, UINT32_MAX);
    }

    // ── stdbool ───────────────────────────────────────────────────────────────
    unsafe {
        printf("=== stdbool ===\n");
        printf("true=%d  false=%d\n", true, false);
        int t = true;
        int f = false;
        printf("!true=%d  !false=%d  (true&&false)=%d\n", !t, !f, t && f);
    }

    // ── limits ────────────────────────────────────────────────────────────────
    unsafe {
        printf("=== limits ===\n");
        printf("CHAR_BIT=%d  SHRT_MAX=%d  INT_MAX=%d\n",
               CHAR_BIT, SHRT_MAX, INT_MAX);
        printf("UINT_MAX=%u  LLONG_MAX=%lld\n", UINT_MAX, LLONG_MAX);
    }

    // ── float ─────────────────────────────────────────────────────────────────
    unsafe {
        printf("=== float ===\n");
        printf("FLT_EPSILON=%g  FLT_MAX=%g\n",
               (double)FLT_EPSILON, (double)FLT_MAX);
        printf("DBL_EPSILON=%g  DBL_MIN=%g\n",
               DBL_EPSILON, DBL_MIN);
    }

    // ── ctype: all 14 functions ───────────────────────────────────────────────
    unsafe {
        printf("=== ctype ===\n");
        printf("is_alpha: A=%d  5=%d\n",
               char_is_alpha((int)'A'), char_is_alpha((int)'5'));
        printf("is_digit: 7=%d  x=%d\n",
               char_is_digit((int)'7'), char_is_digit((int)'x'));
        printf("is_alnum: b=%d  !=%d\n",
               char_is_alnum((int)'b'), char_is_alnum((int)'!'));
        printf("is_xdigit: f=%d  g=%d\n",
               char_is_xdigit((int)'f'), char_is_xdigit((int)'g'));
        printf("is_space: ' '=%d  a=%d\n",
               char_is_space((int)' '), char_is_space((int)'a'));
        printf("is_blank: tab=%d  nl=%d\n",
               char_is_blank((int)'\t'), char_is_blank((int)'\n'));
        printf("is_upper: Z=%d  z=%d\n",
               char_is_upper((int)'Z'), char_is_upper((int)'z'));
        printf("is_lower: z=%d  Z=%d\n",
               char_is_lower((int)'z'), char_is_lower((int)'Z'));
        printf("is_print: a=%d  ctrl=%d\n",
               char_is_print((int)'a'), char_is_print(1));
        printf("is_graph: a=%d  ' '=%d\n",
               char_is_graph((int)'a'), char_is_graph((int)' '));
        printf("is_punct: !=%d  a=%d\n",
               char_is_punct((int)'!'), char_is_punct((int)'a'));
        printf("is_ctrl: 1=%d  a=%d\n",
               char_is_ctrl(1), char_is_ctrl((int)'a'));
        printf("to_upper: a->%c  B->%c\n",
               char_to_upper((int)'a'), char_to_upper((int)'B'));
        printf("to_lower: A->%c  b->%c\n",
               char_to_lower((int)'A'), char_to_lower((int)'b'));
    }

    // ── assert: runtime checks ────────────────────────────────────────────────
    runtime_assert(1 + 1 == 2, "arithmetic");
    runtime_assert(INT32_MAX > 0, "INT32_MAX positive");
    assert_true(LLONG_MAX > 0LL);
    unsafe { printf("=== assert: all checks passed ===\n"); }

    // ── stdckdint: checked arithmetic ─────────────────────────────────────────
    unsafe {
        printf("=== stdckdint ===\n");

        // Use heap-allocated output params to avoid &stack->int* conversion
        int* r32  = (int*)alloc((unsigned long)4);
        int* ov   = (int*)alloc((unsigned long)4);

        *ov = ckd_add_i32(r32, INT32_MAX, 1);
        printf("INT32_MAX+1:     result=%d  overflow=%d\n", *r32, *ov);

        *ov = ckd_add_i32(r32, 100, 200);
        printf("100+200:         result=%d  overflow=%d\n", *r32, *ov);

        *ov = ckd_mul_i32(r32, 100000, 100000);
        printf("100000*100000:   result=%d  overflow=%d\n", *r32, *ov);

        long long* r64 = (long long*)alloc((unsigned long)8);
        *ov = ckd_add_i64(r64, INT64_MAX, (long long)1);
        printf("INT64_MAX+1:     overflow=%d\n", *ov);

        *ov = ckd_sub_i64(r64, (long long)INT64_MIN, (long long)1);
        printf("INT64_MIN-1:     overflow=%d\n", *ov);

        unsigned int* ur32 = (unsigned int*)alloc((unsigned long)4);
        *ov = ckd_add_u32(ur32, (unsigned int)UINT32_MAX, (unsigned int)1);
        printf("UINT32_MAX+1:    result=%u  overflow=%d\n", *ur32, *ov);

        dealloc((void*)r32);
        dealloc((void*)ov);
        dealloc((void*)r64);
        dealloc((void*)ur32);
    }

    // ── io: buffer formatting ─────────────────────────────────────────────────
    unsafe {
        printf("=== io_fmt ===\n");
        char buf[128];

        io_fmt_int((char*)buf, 128, (long long)-42);
        printf("fmt_int(-42)     = '%s'\n", buf);

        io_fmt_uint((char*)buf, 128, (unsigned long long)18446744073709551615ULL);
        printf("fmt_uint(2^64-1) = '%s'\n", buf);

        io_fmt_float((char*)buf, 128, 3.14159265358979);
        printf("fmt_float(pi)    = '%s'\n", buf);

        io_fmt_float_prec((char*)buf, 128, 2.71828182845904, 6);
        printf("fmt_float_prec(e,6) = '%s'\n", buf);

        io_fmt_hex((char*)buf, 128, (unsigned long long)3735928559);
        printf("fmt_hex(0xDEADBEEF) = '%s'\n", buf);

        io_fmt_str((char*)buf, 128, "hello world");
        printf("fmt_str          = '%s'\n", buf);
    }

    // ── sys: PRNG ─────────────────────────────────────────────────────────────
    unsafe {
        printf("=== sys_rand ===\n");
        sys_srand((unsigned int)42);
        int r1 = sys_rand();
        int r2 = sys_rand();
        sys_srand((unsigned int)42);     // reseed — should reproduce r1
        int r3 = sys_rand();
        printf("rand()=%d  rand()=%d  (reseed)  rand()=%d  reproducible=%d\n",
               r1, r2, r3, r1 == r3);
    }

    // ── sys: integer utilities ────────────────────────────────────────────────
    unsafe {
        printf("=== sys_abs ===\n");
        printf("abs(-7)=%d  abs(7)=%d\n", sys_abs(-7), sys_abs(7));
        printf("llabs(-9000000000)=%lld\n", sys_llabs(-9000000000LL));
    }

    // ── sys: string-to-number conversions ────────────────────────────────────
    unsafe {
        printf("=== sys_str_conv ===\n");
        printf("atoi('123')=%d\n",           sys_atoi("123"));
        printf("atoll('-12345')=%lld\n",      sys_atoll("-12345"));
        printf("atof('3.14')=%g\n",           sys_atof("3.14"));
        printf("strtoll('0xFF',0)=%lld\n",    sys_strtoll("0xFF", (char**)0, 0));
        printf("strtoll('0b1010',2)=%lld\n",  sys_strtoll("1010", (char**)0, 2));
        printf("strtod('6.28e1')=%g\n",       sys_strtod("6.28e1", (char**)0));
        printf("strtof('2.5')=%g\n", (double) sys_strtof("2.5", (char**)0));
    }

    // ── str: new string functions ─────────────────────────────────────────────
    unsafe {
        printf("=== str_new_functions ===\n");

        // Concatenation
        char dst[128];
        str_copy((char*)dst, "Hello", (unsigned long)128);
        str_cat((char*)dst, ", SafeC");
        printf("str_cat:  '%s'\n", dst);

        char dst2[32];
        str_copy((char*)dst2, "foo", (unsigned long)32);
        str_ncat((char*)dst2, "barbaz", (unsigned long)3);
        printf("str_ncat: '%s'\n", dst2);

        // Reverse character search
        const char* last = str_rfind_char("abcXabc", (int)'X');
        printf("str_rfind_char 'X': '%s'\n", last);

        // Span / cspan
        unsigned long sp = str_span("aabbbccc", "ab");
        unsigned long cs = str_cspan("aabbbccc", "c");
        printf("str_span(ab)=%llu  str_cspan(c)=%llu\n", sp, cs);

        // Find-any
        const char* pb = str_find_any("hello world", "aeiou");
        printf("str_find_any first vowel: '%s'\n", pb);

        // Tokenisation (reentrant)
        // Use heap-allocated char** for save state (SafeC needs raw pointer, not &stack ref)
        char** sv = (char**)alloc((unsigned long)8);
        *sv = (char*)0;
        char sentence[64];
        str_copy((char*)sentence, "one two three", (unsigned long)64);
        char* t1 = str_tok((char*)sentence, " ", sv);
        char* t2 = str_tok((char*)0, " ", sv);
        char* t3 = str_tok((char*)0, " ", sv);
        printf("str_tok: '%s' '%s' '%s'\n", t1, t2, t3);
        dealloc((void*)sv);

        // mem_chr
        const char* needle = "find the X here";
        void* found = mem_chr((void*)needle, (int)'X', (unsigned long)15);
        printf("mem_chr 'X': '%s'\n", (char*)found);

        // str_ndup
        char* nd = str_ndup("hello world", (unsigned long)5);
        printf("str_ndup(5): '%s'\n", nd);
        dealloc((void*)nd);
    }

    unsafe { printf("\nAll stdlib_complete tests passed.\n"); }
    return 0;
}
