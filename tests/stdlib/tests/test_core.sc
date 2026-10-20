// std core: mem, str, fmt, convert, bit, ctype, math, stdckdint, complex, sys
#include <std/test/test.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/str.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/bit.h>
#include <std/ctype.h>
#include <std/math.h>
#include <std/stdckdint.h>
#include <std/complex.h>
#include <std/sys.h>
#include <std/io.h>

static int near_d(double a, double b, double eps) {
    double d = a - b;
    if (d < 0.0) d = -d;
    return d <= eps;
}

void t_mem() {
    unsafe {
        ?&heap char pq = alloc(64UL);
        ?&heap char zq = alloc_zeroed(32UL);
        ASSERT_FALSE(pq.is_null());
        ASSERT_FALSE(zq.is_null());
        if (pq.is_null() || zq.is_null()) return;
        &char p = std::expect_ref(pq, "pq is null");
        &char z = std::expect_ref(zq, "zq is null");
        safe_memset(p, 7, 64UL);
        ASSERT_EQ(p[63], 7);
        ASSERT_EQ(z[0], 0);
        ASSERT_EQ(z[31], 0);
        safe_memcpy(z, "hello", 6UL);
        ASSERT_STR_EQ(z, "hello");
        safe_memmove(&z[1], z, 5UL);
        ASSERT_EQ(z[1], 'h');
        ASSERT_EQ(safe_memcmp("abc", "abd", 3UL) < 0, 1);
        ?&heap char rq = realloc_buf(pq, 4096UL);
        ASSERT_FALSE(rq.is_null());
        if (rq.is_null()) return;
        &char r = std::expect_ref(rq, "rq is null");
        ASSERT_EQ(r[10], 7);
        dealloc(rq);
        dealloc(zq);
        ASSERT_EQ(mem_align_up(13UL, 8UL), 16);
        ASSERT_EQ(mem_align_down(13UL, 8UL), 8);
        ASSERT_TRUE(mem_is_aligned(64UL, 64UL));
        ASSERT_EQ(checked_mul_size(3UL, 5UL), 15);
        ASSERT_EQ(checked_add_size(3UL, 5UL), 8);
    }
}

void t_str() {
    unsafe {
        ASSERT_EQ(str_len("hello"), 5);
        ASSERT_TRUE(str_eq("a", "a"));
        ASSERT_TRUE(str_cmp("a", "b") < 0);
        ASSERT_EQ(str_ncmp("abcx", "abcy", 3UL), 0);
        char buf[32];
        str_copy((char*)buf, "foo", 32UL);
        str_cat((char*)buf, "bar");
        ASSERT_STR_EQ((const char*)buf, "foobar");
        str_ncat((char*)buf, "bazqux", 3UL);
        ASSERT_STR_EQ((const char*)buf, "foobarbaz");
        ASSERT_STR_EQ(str_find("hello world", "wor"), "world");
        ASSERT_TRUE(str_find("hello", "xyz") == (const char*)0);
        ASSERT_STR_EQ(str_find_char("a.b.c", '.'), ".b.c");
        ASSERT_STR_EQ(str_rfind_char("a.b.c", '.'), ".c");
        ASSERT_EQ(str_span("aaab", "a"), 3);
        ASSERT_EQ(str_cspan("abc,d", ","), 3);
        ?&heap char d = str_dup("dup");
        ASSERT_FALSE(d.is_null());
        ASSERT_STR_EQ(d, "dup");
        dealloc(d);
        ?&heap char nd = str_ndup("abcdef", 3UL);
        ASSERT_FALSE(nd.is_null());
        ASSERT_STR_EQ(nd, "abc");
        dealloc(nd);
        char tok[32];
        str_copy(tok, "a,b,,c", 32UL);
        ?&char save = null;
        ?&char t1 = str_tok(tok, ",", &save);
        ?&char t2 = str_tok(null, ",", &save);
        ?&char t3 = str_tok(null, ",", &save);
        ?&char t4 = str_tok(null, ",", &save);
        ASSERT_STR_EQ(t1, "a");
        ASSERT_STR_EQ(t2, "b");
        ASSERT_STR_EQ(t3, "c");
        ASSERT_TRUE(t4.is_null());
    }
}

void t_fmt() {
    unsafe {
        char buf[64];
        fmt_int((char*)buf, 64UL, -1234LL);
        ASSERT_STR_EQ((const char*)buf, "-1234");
        fmt_uint((char*)buf, 64UL, 18446744073709551615ULL);
        ASSERT_STR_EQ((const char*)buf, "18446744073709551615");
        fmt_hex((char*)buf, 64UL, 0xdeadbeefULL);
        ASSERT_TRUE(str_find((const char*)buf, "deadbeef") != (const char*)0);
        fmt_float((char*)buf, 64UL, 3.14159, 2);
        ASSERT_STR_EQ((const char*)buf, "3.14");
        fmt_bool((char*)buf, 64UL, 1);
        ASSERT_STR_EQ((const char*)buf, "true");
        fmt_str((char*)buf, 64UL, "ab");
        fmt_append((char*)buf, 64UL, "cd");
        ASSERT_STR_EQ((const char*)buf, "abcd");
        // Truncation must never overflow and must NUL-terminate.
        char small[4];
        fmt_str((char*)small, 4UL, "abcdefgh");
        ASSERT_EQ(str_len((const char*)small) < 4, 1);
        io_fmt_int((char*)buf, 64, -42LL);
        ASSERT_STR_EQ((const char*)buf, "-42");
    }
}

void t_convert() {
    unsafe {
        int ok = 0;
        ASSERT_EQ(str_to_int("-987", &ok), -987);
        ASSERT_TRUE(ok);
        str_to_int("12x", &ok);
        ASSERT_FALSE(ok);
        ASSERT_EQ(str_to_uint("42", &ok), 42);
        ASSERT_EQ(str_to_hex("ff", &ok), 255);
        ASSERT_TRUE(near_d(str_to_float("2.5", &ok), 2.5, 1e-12));
        ?&heap char s = int_to_str(-55LL);
        ?&heap char f = float_to_str(1.5, 1);
        ASSERT_FALSE(s.is_null());
        ASSERT_FALSE(f.is_null());
        ASSERT_STR_EQ(s, "-55");
        ASSERT_STR_EQ(f, "1.5");
        dealloc(s);
        dealloc(f);
        ASSERT_TRUE(str_is_int("123"));
        ASSERT_FALSE(str_is_int("1.5"));
        ASSERT_TRUE(str_is_float("1.5"));
        int out = 0;
        ASSERT_FALSE(ll_to_int(10000000000LL, &out));
        ASSERT_TRUE(ll_to_int(77LL, &out));
        ASSERT_EQ(out, 77);
        ASSERT_EQ(float_clamp_to_ll(1e30, 0LL, 100LL), 100);
    }
}

void t_bit() {
    ASSERT_EQ(popcount32(0xF0F0u), 8);
    ASSERT_EQ(popcount64(0xFFFFFFFFFFFFFFFFULL), 64);
    ASSERT_EQ(clz32(1u), 31);
    ASSERT_EQ(ctz32(8u), 3);
    ASSERT_EQ(clz64(1ULL), 63);
    ASSERT_EQ(ctz64(0x100ULL), 8);
    ASSERT_EQ(rotl32(0x80000001u, 1), 3);
    ASSERT_EQ(rotr32(3u, 1), 0x80000001u);
    ASSERT_EQ(bswap32(0x11223344u), 0x44332211u);
    test_assert_eq_u(bswap64(0x1122334455667788ULL), 0x8877665544332211ULL, "bswap64", __FILE__, __LINE__);
    ASSERT_TRUE(is_pow2(64ULL));
    ASSERT_FALSE(is_pow2(65ULL));
    ASSERT_EQ(next_pow2(65ULL), 128);
    ASSERT_EQ(ilog2(1024ULL), 10);
    ASSERT_EQ(bits32(0xABCDu, 4, 11), 0xBCu);
    ASSERT_EQ(set_bits32(0u, 4, 7, 0xFu), 0xF0u);
}

void t_ctype() {
    ASSERT_TRUE(char_is_alpha('A'));
    ASSERT_FALSE(char_is_alpha('5'));
    ASSERT_TRUE(char_is_digit('5'));
    ASSERT_TRUE(char_is_xdigit('f'));
    ASSERT_TRUE(char_is_space('\n'));
    ASSERT_TRUE(char_is_punct('!'));
    ASSERT_EQ(char_to_upper('a'), 'A');
    ASSERT_EQ(char_to_lower('Z'), 'z');
}

void t_math() {
    ASSERT_TRUE(near_d(sqrt_d(2.0), 1.4142135623730951, 1e-12));
    ASSERT_TRUE(near_d(sin_d(PI_D() / 2.0), 1.0, 1e-12));
    ASSERT_TRUE(near_d(pow_d(2.0, 10.0), 1024.0, 1e-9));
    ASSERT_TRUE(near_d(log2_d(8.0), 3.0, 1e-12));
    ASSERT_TRUE(near_d(atan2_d(1.0, 1.0), PI_D() / 4.0, 1e-12));
    ASSERT_TRUE(near_d((double)sqrt_f((float)9.0), 3.0, 1e-6));
    ASSERT_EQ(abs_int(-5), 5);
    ASSERT_EQ(clamp_int(50, 0, 10), 10);
    ASSERT_EQ(min_ll(3LL, -3LL), -3);
    ASSERT_TRUE(isinf_d(INF_D()));
    ASSERT_TRUE(isnan_d(0.0 / 0.0));
    ASSERT_TRUE(near_d(floor_d(-1.5), -2.0, 0.0));
    ASSERT_TRUE(near_d(round_d(2.5), 3.0, 0.0));
    ASSERT_TRUE(near_d(fmod_d(7.0, 3.0), 1.0, 1e-12));
    ASSERT_TRUE(near_d(hypot_d(3.0, 4.0), 5.0, 1e-12));
}

void t_ckd() {
    unsafe {
        int r = 0;
        ASSERT_TRUE(ckd_add_i32(&r, 2147483647, 1));
        ASSERT_FALSE(ckd_add_i32(&r, 2, 3));
        ASSERT_EQ(r, 5);
        ASSERT_TRUE(ckd_mul_i32(&r, 100000, 100000));
        long long r64 = 0LL;
        ASSERT_TRUE(ckd_sub_i64(&r64, -9223372036854775807LL - 1LL, 1LL));
        unsigned int u = 0u;
        ASSERT_TRUE(ckd_sub_u32(&u, 0u, 1u));
        unsigned long long u64 = 0ULL;
        ASSERT_FALSE(ckd_mul_u64(&u64, 1000000ULL, 1000000ULL));
        test_assert_eq_u(u64, 1000000000000ULL, "ckd_mul_u64", __FILE__, __LINE__);
    }
}

void t_complex() {
    unsafe {
        double a[2];
        double b[2];
        double o[2];
        cmplx_d((double*)a, 1.0, 2.0);
        cmplx_d((double*)b, 3.0, -1.0);
        cmul_d((double*)o, (const double*)a, (const double*)b);
        ASSERT_TRUE(near_d(o[0], 5.0, 1e-12));
        ASSERT_TRUE(near_d(o[1], 5.0, 1e-12));
        cdiv_d((double*)o, (const double*)o, (const double*)b);
        ASSERT_TRUE(near_d(o[0], 1.0, 1e-12));
        ASSERT_TRUE(near_d(o[1], 2.0, 1e-12));
        cmplx_d((double*)a, 3.0, 4.0);
        ASSERT_TRUE(near_d(cabs_d((const double*)a), 5.0, 1e-12));
        cmplx_d((double*)a, -4.0, 0.0);
        csqrt_d((double*)o, (const double*)a);
        ASSERT_TRUE(near_d(o[0], 0.0, 1e-12));
        ASSERT_TRUE(near_d(o[1], 2.0, 1e-12));
    }
}

static int cmp_int(const &int a, const &int b) {
    return *a - *b;
}

void t_sys() {
    unsafe {
        sys_srand(42u);
        int r1 = sys_rand();
        sys_srand(42u);
        ASSERT_EQ(sys_rand(), r1);
        ASSERT_EQ(sys_atoi("123"), 123);
        ASSERT_EQ(sys_strtoll("ff", null, 16), 255);
        int arr[5];
        arr[0] = 5; arr[1] = 1; arr[2] = 4; arr[3] = 2; arr[4] = 3;
        sys_qsort(&arr[0], 5UL, cmp_int);
        ASSERT_EQ(arr[0], 1);
        ASSERT_EQ(arr[4], 5);
        int key = 4;
        ?&int hit = sys_bsearch(&key, &arr[0], 5UL, cmp_int);
        ASSERT_EQ(hit.default(-1), 4);
        ASSERT_TRUE(sys_monotonic_ns() > 0LL);
    }
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("mem", t_mem);
        t.add("str", t_str);
        t.add("fmt", t_fmt);
        t.add("convert", t_convert);
        t.add("bit", t_bit);
        t.add("ctype", t_ctype);
        t.add("math", t_math);
        t.add("stdckdint", t_ckd);
        t.add("complex", t_complex);
        t.add("sys", t_sys);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
