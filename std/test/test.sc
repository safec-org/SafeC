// SafeC Standard Library — Unit Testing Framework
#pragma once
#include "test.h"

extern int    printf(const char* fmt, ...);
extern int    strcmp(const char* a, const char* b);
extern void*  memset(void* p, int v, unsigned long n);
extern void*  memcpy(void* d, const void* s, unsigned long n);
extern void   exit(int code);
extern unsigned long strlen(const char* s);

// ── Global test context ───────────────────────────────────────────────────────
// Points to the TestCase currently being executed; set by run() before each call.
static struct TestCase* test_current_ = (struct TestCase*)0;

// ── strncpy helper ────────────────────────────────────────────────────────────
static void tc_copy_(char* dst, const char* src, unsigned long n) {
    unsigned long i = (unsigned long)0;
    while (i < n - (unsigned long)1 && src[i] != '\0') {
        dst[i] = src[i];
        i = i + (unsigned long)1;
    }
    dst[i] = '\0';
}

// ── Assertion functions ────────────────────────────────────────────────────────

void test_assert_true(int cond, const char* expr, const char* file, int line) {
    if (cond != 0) { return; }
    if (test_current_ == (struct TestCase*)0) { return; }
    if (test_current_->result == 0) { return; }   // already failed; keep first
    test_current_->result    = 0;
    test_current_->fail_line = line;
    tc_copy_(test_current_->fail_expr, expr, (unsigned long)128);
    tc_copy_(test_current_->fail_file, file, (unsigned long)128);
}

void test_assert_eq_i(long long a, long long b, const char* desc,
                      const char* file, int line) {
    test_assert_true(a == b, desc, file, line);
}

void test_assert_eq_u(unsigned long long a, unsigned long long b,
                      const char* desc, const char* file, int line) {
    test_assert_true(a == b, desc, file, line);
}

void test_assert_eq_s(const char* a, const char* b, const char* desc,
                      const char* file, int line) {
    int ok = 0;
    if (a != (const char*)0 && b != (const char*)0) {
        unsafe { ok = (strcmp(a, b) == 0) ? 1 : 0; }
    }
    test_assert_true(ok, desc, file, line);
}

void test_assert_null(const void* p, const char* desc,
                      const char* file, int line) {
    test_assert_true(p == (const void*)0, desc, file, line);
}

void test_assert_not_null(const void* p, const char* desc,
                           const char* file, int line) {
    test_assert_true(p != (const void*)0, desc, file, line);
}

// ── TestSuite ─────────────────────────────────────────────────────────────────

struct TestSuite test_suite_init() {
    struct TestSuite s;
    s.count  = 0;
    s.passed = 0;
    s.failed = 0;
    return s;
}

void TestSuite::add(const char* name, void* fn) {
    if (self.count >= TEST_MAX) { return; }
    int idx = self.count;
    tc_copy_(self.cases[idx].name, name, (unsigned long)TEST_NAME_MAX);
    self.cases[idx].fn        = fn;
    self.cases[idx].result    = -1;
    self.cases[idx].fail_line = 0;
    self.cases[idx].fail_expr[0] = '\0';
    self.cases[idx].fail_file[0] = '\0';
    self.count = self.count + 1;
}

void TestSuite::run() {
    int i = 0;
    while (i < self.count) {
        self.cases[i].result = 1;       // assume pass
        test_current_ = &self.cases[i]; // set global context

        unsafe {
            ((void(*)(void))self.cases[i].fn)();
        }

        test_current_ = (struct TestCase*)0;

        if (self.cases[i].result == 1) {
            self.passed = self.passed + 1;
            unsafe { printf("  PASS  %s\n", self.cases[i].name); }
        } else {
            self.failed = self.failed + 1;
            unsafe {
                printf("  FAIL  %s\n", self.cases[i].name);
                printf("        %s:%d: %s\n",
                       self.cases[i].fail_file,
                       self.cases[i].fail_line,
                       self.cases[i].fail_expr);
            }
        }
        i = i + 1;
    }
}

void TestSuite::print_summary() const {
    unsafe {
        printf("--- %d passed, %d failed ---\n", self.passed, self.failed);
    }
}

int TestSuite::all_passed() const {
    return self.failed == 0 ? 1 : 0;
}

void test_run_and_exit(&stack TestSuite suite) {
    suite.run();
    suite.print_summary();
    if (suite.all_passed() == 0) {
        unsafe { exit(1); }
    }
}
