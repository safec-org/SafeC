// SafeC Standard Library — Unit Testing Framework
// Freestanding-safe (uses only io.h for output).
//
// SafeC usage (no function-like macros needed):
//   struct TestSuite t = TestSuite::new();   // std::TestSuite::new() outside namespace std
//   t.add("addition works", my_test);
//   t.run();
//   return t.all_passed() ? 0 : 1;
//
// C usage: the ASSERT_* macros expand to the typed assertion calls.
#pragma once

#define TEST_MAX      256
#define TEST_NAME_MAX  64

// ── Assertion functions (call directly from SafeC test functions) ──────────────
// Each must be called inside a test function. They record the first failure
// into the currently running TestCase via a global thread-local pointer.

// Assert that cond is non-zero.
namespace std {

void test_assert_true(int cond, const &char expr, const &char file, int line);

// Assert that two long-long integers are equal.
void test_assert_eq_i(long long a, long long b, const &char desc,
                      const &char file, int line);

// Assert two unsigned long-long values are equal.
void test_assert_eq_u(unsigned long long a, unsigned long long b,
                      const &char desc, const &char file, int line);

// Assert two null-terminated strings are equal.
void test_assert_eq_s(const ?&char a, const ?&char b, const &char desc,
                      const &char file, int line);

// Assert a nullable reference is null / not null.
generic<T>
void test_assert_null(const ?&T p, const &char desc, const &char file, int line) {
    test_assert_true(p.is_null(), desc, file, line);
}

generic<T>
void test_assert_not_null(const ?&T p, const &char desc, const &char file, int line) {
    test_assert_true(!p.is_null(), desc, file, line);
}

// ── C-compatible convenience macros ───────────────────────────────────────────
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define ASSERT_TRUE(e)        test_assert_true((e)!=0, #e, __FILE__, __LINE__)
#endif
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define ASSERT_FALSE(e)       test_assert_true((e)==0, "!" #e, __FILE__, __LINE__)
#endif
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define ASSERT_EQ(a,b)        test_assert_eq_i((long long)(a),(long long)(b),\
                                  #a " == " #b, __FILE__, __LINE__)
#endif
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define ASSERT_NE(a,b)        test_assert_true((a)!=(b), #a " != " #b, __FILE__, __LINE__)
#endif
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define ASSERT_NULL(p)        test_assert_null((p), #p, __FILE__, __LINE__)
#endif
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define ASSERT_NOT_NULL(p)    test_assert_not_null((p), #p, __FILE__, __LINE__)
#endif
#ifdef __SAFEC_COMPAT_PREPROCESSOR__   // function-like macro: compat mode only
#define ASSERT_STR_EQ(a,b)    test_assert_eq_s((a),(b), #a " == " #b, __FILE__, __LINE__)
#endif

// ── TestCase ──────────────────────────────────────────────────────────────────

struct TestCase {
    char        name[TEST_NAME_MAX];
    fn void()   func;
    int         result;      // 1 = passed, 0 = failed, -1 = not run yet
    char        fail_expr[128];
    char        fail_file[128];
    int         fail_line;
};

// ── TestSuite ─────────────────────────────────────────────────────────────────

struct TestSuite {
    struct TestCase cases[TEST_MAX];
    int count;
    int passed;
    int failed;

    // An empty suite.
    static struct TestSuite new();

    // Register a test case.
    void add(const &char name, fn void() func);

    // Run all registered tests, print each result, update passed/failed.
    void run();

    // Print a one-line summary: "N passed, M failed."
    void print_summary() const;

    // Return 1 if every registered test passed, 0 otherwise.
    int  all_passed() const;
};

// Run all tests in `suite`, print summary, then call exit(1) on any failure.
// Useful as the last line of a test main().
void test_run_and_exit(&TestSuite suite);

} // namespace std
