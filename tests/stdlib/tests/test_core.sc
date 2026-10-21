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
#include <std/atomic.h>
#include <std/cmp.h>
#include <std/fenv.h>
#include <std/errno.h>
#include <std/locale.h>
#include <std/log.h>
#include <std/time.h>
#include <std/signal.h>

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

// Edge cases the original suite missed: full-width bit fields, wrapped
// results of checked arithmetic, and the less common math/complex/str APIs.
void t_core_extra() {
    unsafe {
        test_assert_eq_u(bits32(0xDEADBEEFu, 0, 31), 0xDEADBEEFULL, "bits32 full", __FILE__, __LINE__);
        test_assert_eq_u(bits64(0x0123456789ABCDEFULL, 0, 63), 0x0123456789ABCDEFULL, "bits64 full", __FILE__, __LINE__);
        test_assert_eq_u(bits64(0xF0ULL, 4, 7), 0xFULL, "bits64", __FILE__, __LINE__);
        test_assert_eq_u(set_bits32(0x12345678u, 0, 31, 0xCAFEBABEu), 0xCAFEBABEULL, "set_bits32 full", __FILE__, __LINE__);
        test_assert_eq_u(rotl64(0x8000000000000001ULL, 4), 0x18ULL, "rotl64", __FILE__, __LINE__);
        ASSERT_EQ(bsf32(0u), -1);
        ASSERT_EQ(bsf32(0x50u), 4);
        ASSERT_EQ(bsr32(0x50u), 6);

        // ckd_*: the wrapped two's-complement result is stored on overflow.
        int r = 0;
        ASSERT_TRUE(ckd_add_i32(&r, 2147483647, 1));
        ASSERT_EQ(r, -2147483647 - 1);
        ASSERT_TRUE(ckd_sub_i32(&r, -2147483647 - 1, 1));
        ASSERT_EQ(r, 2147483647);
        long long r64 = 0LL;
        ASSERT_TRUE(ckd_add_i64(&r64, 9223372036854775807LL, 2LL));
        ASSERT_EQ(r64, -9223372036854775807LL);
        ASSERT_TRUE(ckd_mul_i64(&r64, 4611686018427387904LL, 2LL));
        ASSERT_EQ(r64, -9223372036854775807LL - 1LL);
        ASSERT_FALSE(ckd_mul_i64(&r64, -3LL, 7LL));
        ASSERT_EQ(r64, -21);
        unsigned int u = 0u;
        ASSERT_TRUE(ckd_add_u32(&u, 4294967295u, 2u));
        ASSERT_EQ(u, 1);
        ASSERT_TRUE(ckd_mul_u32(&u, 65536u, 65536u));
        unsigned long long u64 = 0ULL;
        ASSERT_TRUE(ckd_sub_u64(&u64, 0ULL, 1ULL));
        test_assert_eq_u(u64, 18446744073709551615ULL, "ckd_sub_u64 wrap", __FILE__, __LINE__);

        // str_ndup stops at the terminator.
        ?&heap char nd = str_ndup("ab", 100UL);
        ASSERT_STR_EQ(nd, "ab");
        dealloc(nd);
        ASSERT_STR_EQ(str_find_any("hello, world", ",!"), ", world");
        ASSERT_TRUE(str_find_any("abc", "xyz").is_null());
        const ?&unsigned char hit = mem_chr((const &unsigned char)"abcdef", 'd', 6UL);
        match (hit) {
            case null: { ASSERT_TRUE(0); }
            case some(h): { ASSERT_EQ(h[0], 'd'); ASSERT_EQ(h[1], 'e'); }
        }
        ASSERT_TRUE(mem_chr((const &unsigned char)"abc", 'z', 3UL).is_null());

        // float_clamp_to_ll rounds to nearest and handles NaN / the top end.
        ASSERT_EQ(float_clamp_to_ll(2.6, 0LL, 100LL), 3);
        ASSERT_EQ(float_clamp_to_ll(-2.6, -100LL, 100LL), -3);
        ASSERT_EQ(float_clamp_to_ll(0.0 / 0.0, -5LL, 5LL), -5);
        ASSERT_EQ(float_clamp_to_ll(1e300, 0LL, 9223372036854775807LL), 9223372036854775807LL);

        // ctype / math odds and ends
        ASSERT_TRUE(char_is_alnum('z'));
        ASSERT_FALSE(char_is_alnum('-'));
        ASSERT_TRUE(char_is_print(' '));
        ASSERT_FALSE(char_is_print(127));
        ASSERT_TRUE(char_is_upper('Q'));
        ASSERT_TRUE(char_is_ctrl(10));
        ASSERT_EQ(abs_ll(-7LL), 7);
        ASSERT_EQ(max_int(3, 9), 9);
        ASSERT_EQ(max_ll(-3LL, -9LL), -3);
        ASSERT_EQ(clamp_ll(-50LL, -10LL, 10LL), -10);
        ASSERT_TRUE(near_d((double)clamp_f((float)2.5, (float)0.0, (float)1.0), 1.0, 0.0));
        ASSERT_TRUE(near_d((double)min_f((float)2.0, (float)-1.0), -1.0, 0.0));
        ASSERT_TRUE(near_d(max_d(1.0, 2.0), 2.0, 0.0));
        ASSERT_TRUE(near_d((double)pow_f((float)2.0, (float)8.0), 256.0, 1e-3));
        ASSERT_TRUE(near_d((double)atan2_f((float)1.0, (float)1.0), PI_D() / 4.0, 1e-6));
        ASSERT_TRUE(near_d((double)exp2_f((float)5.0), 32.0, 1e-4));
        ASSERT_TRUE(near_d((double)log2_f((float)64.0), 6.0, 1e-5));
        ASSERT_TRUE(near_d((double)hypot_f((float)5.0, (float)12.0), 13.0, 1e-5));
        ASSERT_TRUE(near_d((double)copysign_f((float)3.0, (float)-0.5), -3.0, 0.0));
        ASSERT_TRUE(near_d((double)floor_f((float)-1.5), -2.0, 0.0));
        ASSERT_TRUE(near_d((double)round_f((float)2.5), 3.0, 0.0));
        ASSERT_TRUE(near_d((double)tan_f((float)0.0), 0.0, 1e-7));
        ASSERT_TRUE(near_d((double)acos_f((float)1.0), 0.0, 1e-6));
        ASSERT_TRUE(near_d((double)cosh_f((float)0.0), 1.0, 1e-7));
        ASSERT_TRUE(isinf_f((float)1e30 * (float)1e30));
        ASSERT_FALSE(isfinite_d(INF_D()));
        ASSERT_TRUE(isfinite_d(1.0));
        ASSERT_TRUE(near_d(asin_d(1.0), PI_D() / 2.0, 1e-12));
        ASSERT_TRUE(near_d(atan_d(1.0), PI_D() / 4.0, 1e-12));
        ASSERT_TRUE(near_d(cbrt_d(27.0), 3.0, 1e-12));
        ASSERT_TRUE(near_d(ceil_d(1.2), 2.0, 0.0));
        ASSERT_TRUE(near_d(trunc_d(-1.7), -1.0, 0.0));
        ASSERT_TRUE(near_d(fma_d(2.0, 3.0, 4.0), 10.0, 0.0));
        ASSERT_TRUE(near_d(log_d(E_D()), 1.0, 1e-12));
        ASSERT_TRUE(near_d(sinh_d(0.0), 0.0, 0.0));
        ASSERT_TRUE(near_d(tanh_d(0.0), 0.0, 0.0));
        ASSERT_TRUE(near_d(LN2_D(), 0.6931471805599453, 1e-15));
        ASSERT_TRUE(near_d(SQRT2_D() * SQRT2_D(), 2.0, 1e-15));
        ASSERT_TRUE(near_d((double)PI_F(), 3.14159265, 1e-6));

        // single-precision complex
        float a[2];
        float b[2];
        float o[2];
        cmplx_f(a, (float)3.0, (float)4.0);
        ASSERT_TRUE(near_d((double)cabs_f(a), 5.0, 1e-6));
        ASSERT_TRUE(near_d((double)creal_f(a), 3.0, 0.0));
        ASSERT_TRUE(near_d((double)cimag_f(a), 4.0, 0.0));
        cconj_f(o, a);
        ASSERT_TRUE(near_d((double)o[1], -4.0, 0.0));
        cneg_f(o, a);
        ASSERT_TRUE(near_d((double)o[0], -3.0, 0.0));
        cmplx_f(b, (float)1.0, (float)-1.0);
        cadd_f(o, a, b);
        ASSERT_TRUE(near_d((double)o[0], 4.0, 0.0));
        csub_f(o, a, b);
        ASSERT_TRUE(near_d((double)o[1], 5.0, 0.0));
        cmul_f(o, a, b);                          // (3+4i)(1-i) = 7+i
        ASSERT_TRUE(near_d((double)o[0], 7.0, 1e-6));
        ASSERT_TRUE(near_d((double)o[1], 1.0, 1e-6));
        cdiv_f(o, o, b);
        ASSERT_TRUE(near_d((double)o[0], 3.0, 1e-5));
        cmplx_f(a, (float)0.0, (float)0.0);
        cexp_f(o, a);
        ASSERT_TRUE(near_d((double)o[0], 1.0, 1e-6));
        cmplx_f(a, (float)-1.0, (float)0.0);
        clog_f(o, a);                             // log(-1) = i*pi
        ASSERT_TRUE(near_d((double)o[1], PI_D(), 1e-5));
        ASSERT_TRUE(near_d((double)carg_f(a), PI_D(), 1e-5));
        cmplx_f(b, (float)2.0, (float)0.0);
        cmplx_f(a, (float)3.0, (float)0.0);
        cpow_f(o, a, b);                          // 3^2
        ASSERT_TRUE(near_d((double)o[0], 9.0, 1e-4));
        cmplx_f(a, (float)0.0, (float)0.0);
        csin_f(o, a);
        ASSERT_TRUE(near_d((double)o[0], 0.0, 1e-7));
        ccos_f(o, a);
        ASSERT_TRUE(near_d((double)o[0], 1.0, 1e-7));
        cmplx_f(a, (float)-4.0, (float)0.0);
        csqrt_f(o, a);
        ASSERT_TRUE(near_d((double)o[1], 2.0, 1e-5));

        // atomics (single-threaded semantics) and orderings
        int ai = 5;
        ASSERT_EQ(atomic_load_int(&ai), 5);
        ASSERT_EQ(atomic_fetch_and_int(&ai, 4), 5);
        ASSERT_EQ(atomic_fetch_xor_int(&ai, 6), 4);
        ASSERT_EQ(ai, 2);
        long long al = 10LL;
        atomic_store_ll(&al, 20LL);
        ASSERT_EQ(atomic_fetch_sub_ll(&al, 5LL), 20);
        long long expect = 99LL;
        ASSERT_FALSE(atomic_cas_ll(&al, &expect, 1LL));
        ASSERT_EQ(expect, 15);
        ASSERT_TRUE(atomic_cas_ll(&al, &expect, 1LL));
        ASSERT_EQ(al, 1);
        atomic_thread_fence();
        int x = 3;
        int y = 8;
        struct Less<int> lt = order_less();
        struct Greater<int> gt = order_greater();
        ASSERT_TRUE(lt.cmp(&x, &y) < 0);
        ASSERT_TRUE(gt.cmp(&x, &y) > 0);
        ASSERT_EQ(lt.cmp(&x, &x), 0);

        char hb[32];
        fmt_hex_upper((char*)hb, 32UL, 0xabcULL);
        ASSERT_STR_EQ((const char*)hb, "ABC");
        io_fmt_float((char*)hb, 32, 0.5);
        ASSERT_STR_EQ((const char*)hb, "0.5");
        io_fmt_hex((char*)hb, 32, 255ULL);
        ASSERT_STR_EQ((const char*)hb, "0xff");
        io_fmt_uint((char*)hb, 32, 7ULL);
        ASSERT_STR_EQ((const char*)hb, "7");
        io_fmt_float_prec((char*)hb, 32, 1.25, 1);
        ASSERT_STR_EQ((const char*)hb, "1.2");
        io_fmt_str((char*)hb, 32, "xy");
        ASSERT_STR_EQ((const char*)hb, "xy");
    }
}

static int gLogCalls = 0;
static int gLogLevel = 0;
static void log_capture(int level, const &char tag, const &char msg, const &char file, int line) {
    gLogCalls = gLogCalls + 1;
    gLogLevel = level;
}

// fenv, errno, locale, log, time and sys clock/env wrappers.
void t_env() {
    unsafe {
        int mode = fenv_get_round();
        ASSERT_EQ(mode, FE_TONEAREST());
        ASSERT_EQ(fenv_set_round(FE_UPWARD()), 0);
        ASSERT_EQ(fenv_get_round(), FE_UPWARD());
        volatile double one = 1.0;
        volatile double three = 3.0;
        double up = one / three;
        fenv_set_round(FE_DOWNWARD());
        double down = one / three;
        ASSERT_TRUE(up > down);                  // the mode really reached the FPU
        fenv_set_round(FE_TONEAREST());
        fenv_clear(FE_ALL_EXCEPT());
        ASSERT_EQ(fenv_test(FE_ALL_EXCEPT()), 0);
        fenv_raise(FE_DIVBYZERO());
        ASSERT_TRUE(fenv_test(FE_DIVBYZERO()) != 0);
        unsigned char env[64];
        ASSERT_EQ(fenv_save(&env[0]), 0);
        fenv_clear(FE_ALL_EXCEPT());
        ASSERT_EQ(fenv_restore(&env[0]), 0);
        ASSERT_TRUE(fenv_test(FE_DIVBYZERO()) != 0);
        ASSERT_EQ(fenv_save_clear(&env[0]), 0);
        ASSERT_EQ(fenv_test(FE_ALL_EXCEPT()), 0);

        errno_set(ERRNO_ENOENT());
        ASSERT_EQ(errno_get(), ERRNO_ENOENT());
        ASSERT_TRUE(str_len(errno_str(ERRNO_ENOENT())) > 0UL);
        errno_set(0);

        ASSERT_STR_EQ(locale_set(LC_NUMERIC(), "C"), "C");
        ASSERT_STR_EQ(locale_get(LC_NUMERIC()), "C");

        log_set_backend(log_capture);
        log_write(LOG_LEVEL_ERROR, "t", "m", "f.sc", 1);
        log_write(LOG_LEVEL_DEBUG, "t", "filtered", "f.sc", 2);   // above LOG_LEVEL
        ASSERT_EQ(gLogCalls, 1);
        ASSERT_EQ(gLogLevel, LOG_LEVEL_ERROR);

        ASSERT_TRUE(time_is_leap(2024));
        ASSERT_FALSE(time_is_leap(1900));
        ASSERT_TRUE(time_is_leap(2000));
        ASSERT_EQ(time_days_in_month(1, 2023), 28);
        ASSERT_EQ(time_days_in_month(12, 2023), -1);
        ASSERT_TRUE(near_d(time_diff(100LL, 40LL), 60.0, 0.0));
        ASSERT_EQ(time_elapsed_ns(5LL, 12LL), 7);
        int tm[9];
        ASSERT_TRUE(time_parse("2024-02-29 13:45:10", "%Y-%m-%d %H:%M:%S", tm));
        ASSERT_EQ(tm[5], 124);
        ASSERT_EQ(tm[4], 1);
        ASSERT_EQ(tm[3], 29);
        ASSERT_EQ(tm[2], 13);
        time_gmtime(951782400LL, tm);            // 2000-02-29 00:00:00 UTC
        ASSERT_EQ(tm[5], 100);
        ASSERT_EQ(tm[3], 29);
        time_localtime(time_now(), tm);
        ASSERT_TRUE(tm[5] >= 124);
        char tb[32];
        time_gmtime(0LL, tm);
        ASSERT_EQ(time_format(tb, 32UL, "%Y-%m-%d", tm), 10);
        ASSERT_STR_EQ((const char*)tb, "1970-01-01");

        // sys: clocks go through time.h (correct clock ids on Linux and macOS)
        long long c0 = sys_cpu_time_ns();
        ASSERT_TRUE(c0 >= 0LL);
        volatile long long spin = 0LL;
        while (spin < 2000000LL) { spin = spin + 1LL; }
        ASSERT_TRUE(sys_cpu_time_ns() > c0);
        long long m0 = sys_monotonic_ns();
        time_sleep_ns(1500000LL);
        long long dm = sys_monotonic_ns() - m0;
        ASSERT_TRUE(dm >= 1500000LL);
        ASSERT_TRUE(sys_time_now_ns() / 1000000000LL >= time_now() - 1LL);
        ASSERT_TRUE(sys_getenv("SAFEC_SURELY_UNSET_VAR_42").is_null());
        ASSERT_EQ(sys_atoll("-9000000000"), -9000000000LL);
        ASSERT_EQ(sys_llabs(-9000000000LL), 9000000000LL);
        const &char end = "";
        ASSERT_TRUE(near_d(sys_strtod("2.5rest", (?&&const char)&end), 2.5, 0.0));
        ASSERT_STR_EQ(end, "rest");
    }
}

static int gSigCount = 0;
static void on_usr1(int sig) { gSigCount = gSigCount + sig; }
extern int getpid();

void t_signals() {
    unsafe {
        signal_handle(SIG_USR1(), on_usr1);
        ASSERT_EQ(signal_raise(SIG_USR1()), 0);
        ASSERT_EQ(gSigCount, SIG_USR1());
        ASSERT_EQ(signal_kill(getpid(), SIG_USR1()), 0);
        ASSERT_EQ(gSigCount, 2 * SIG_USR1());
        signal_ignore(SIG_USR2());
        ASSERT_EQ(signal_raise(SIG_USR2()), 0);      // ignored: still alive
        signal_default(SIG_USR1());
        ASSERT_EQ(signal_kill(getpid(), 0), 0);      // signal 0: existence check only
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
        t.add("core edge cases", t_core_extra);
        t.add("fenv/errno/locale/log/time", t_env);
        t.add("signals", t_signals);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
