// std/wasm/wasm_rt checks, as a wasm32 module with no imports (run.sh).
// run_tests() returns 0, or the number of the first failed check.
#include <std/wasm/wasm_rt.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>
#include <std/collections/map.h>
#include <std/serial/json.h>
#include <std/sys.h>
#include <std/str.h>

static int cmp_int(const &int a, const &int b) { return *a - *b; }

static int check_heap() {
    // Reuse: a freed block of a class comes back for the next allocation.
    unsigned char* a = malloc(100U);
    unsigned char* b = malloc(100U);
    if (a == (unsigned char*)0 || b == (unsigned char*)0 || a == b) { return 1; }
    free(a);
    unsigned char* c = malloc(90U);
    if (c != a) { return 2; }
    if (malloc_usable_size(c) < 90U) { return 3; }
    free(b); free(c);
    free(c);                       // double free is ignored
    unsigned char* d = malloc(100U);
    unsigned char* e = malloc(100U);
    if (d == e) { return 4; }
    free(d); free(e);
    // Growth: far more than the initial memory, in a loop that frees, so
    // reuse keeps it bounded.
    for (int round = 0; round < 64; round = round + 1) {
        unsigned char* big = malloc(1048576U);   // 1 MiB
        if (big == (unsigned char*)0) { return 5; }
        unsafe { ((unsigned char*)big)[1048575] = (unsigned char)round; }
        free(big);
    }
    unsigned char* huge = malloc(20U * 1048576U);    // grows memory past 20 MiB
    if (huge == (unsigned char*)0) { return 6; }
    unsafe { ((unsigned char*)huge)[20U * 1048576U - 1U] = (unsigned char)1; }
    free(huge);
    // realloc keeps contents.
    unsafe {
        char* s = (char*)malloc(8U);
        memcpy((unsigned char*)s, (const unsigned char*)"abcdefg", 8U);
        s = (char*)realloc((unsigned char*)s, 5000U);
        if (strcmp(s, "abcdefg") != 0) { return 7; }
        free((unsigned char*)s);
        int* z = (int*)calloc(16U, 4U);
        for (int i = 0; i < 16; i = i + 1) { if (z[i] != 0) { return 8; } }
        free((unsigned char*)z);
    }
    return 0;
}

static int check_strings() {
    unsafe {
        if (strlen("hello") != 5U) { return 20; }
        if (strcmp("abc", "abd") >= 0 || strcmp("b", "a") <= 0 || strcmp("x", "x") != 0) { return 21; }
        if (strncmp("abcx", "abcy", 3U) != 0) { return 22; }
        const char* s = "path/to/file.md";
        if (strchr(s, '/') != s + 4 || strrchr(s, '/') != s + 7 || strchr(s, 'z') != (char*)0) { return 23; }
        if (strstr(s, "file") != s + 8 || strstr(s, "nope") != (char*)0) { return 24; }
        if (strspn("aaab", "a") != 3U || strcspn("abc,d", ",") != 3U) { return 25; }
        if (memchr((const unsigned char*)s, (int)'.', 15U) != (unsigned char*)(s + 12)) { return 26; }
        if (strtoll("-42x", (char**)0, 10) != -42LL || strtoull("0x1F", (char**)0, 0) != 31ULL) { return 27; }
        if (atoi("  17") != 17) { return 28; }
        double d = strtod("-1.25e2", (char**)0);
        if (d != -125.0) { return 29; }
    }
    // std on top of them.
    struct String t = String::from("Hello, wasm");
    defer t.drop();
    if (!t.contains("wasm") || t.index_of(",") != 5LL) { return 30; }
    struct String u = t.to_upper();
    defer u.drop();
    if (!u.eq_cstr("HELLO, WASM")) { return 31; }
    struct String num = String::from("12345");
    long long n = num.parse_int((?&int)null);
    num.drop();
    if (n != 12345LL) { return 32; }
    return 0;
}

static int check_sort_map_json() {
    struct Vec<int> v = Vec::new();
    defer v.drop();
    int vals[7] = { 5, 3, 9, 1, 7, 2, 8 };
    for (int i = 0; i < 7; i = i + 1) { v.push(vals[i]); }
    v.sort(cmp_int);
    for (unsigned long i = 1UL; i < v.len; i = i + 1UL) { if (*v.at(i - 1UL) > *v.at(i)) { return 40; } }
    // HashMap growth goes through checked size arithmetic (__multi3).
    struct HashMap<int, long long> m = HashMap::new();
    defer m.drop();
    for (int i = 0; i < 500; i = i + 1) { m.insert(i, (long long)i * 3LL); }
    ?&long long got = m.get(321);
    match (got) { case some(x): { if (*x != 963LL) { return 41; } } case null: { return 42; } }
    int ok = 0;
    struct Value doc = json_parse("[{\"u\":\"/a.html#x\",\"t\":\"Title\"},{\"u\":\"/b\",\"n\":2}]", &ok);
    defer doc.drop();
    if (!ok || doc.array_len() != 2UL) { return 43; }
    struct String back = value_to_json(&doc);
    defer back.drop();
    if (!back.contains("\"t\":\"Title\"")) { return 44; }
    // __multi3 directly: (2^40 + 3) * (2^30 + 5).
    unsigned long long r[2];
    __multi3(&r[0], 1099511627779ULL, 0ULL, 1073741829ULL, 0ULL);
    // The product is 64 * 2^64 + (the low 64 bits a wrapping multiply gives).
    unsigned long long lo = 1099511627779ULL * 1073741829ULL;   // wraps mod 2^64
    if (r[0] != lo || r[1] != 64ULL) { return 45; }
    return 0;
}

__attribute__((export_name)) int run_tests() {
    int f = check_heap();
    if (f != 0) { return f; }
    f = check_strings();
    if (f != 0) { return f; }
    return check_sort_map_json();
}
