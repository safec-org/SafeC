// std::collections — vec, string, stack, queue, list, map, bst, btree, ringbuffer, slice
#include <std/test/test.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/str.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>
#include <std/collections/stack.h>
#include <std/collections/queue.h>
#include <std/collections/list.h>
#include <std/collections/map.h>
#include <std/collections/bst.h>
#include <std/collections/btree.h>
#include <std/collections/ringbuffer.h>
#include <std/collections/slice.h>
#include <std/collections/static_vec.h>

static int cmp_int(const &int a, const &int b) {
    return *a - *b;
}
static int is_even(const &int a) {
    return (*a % 2) == 0;
}

static long long triple(const &int a) { return (long long)(*a) * 3LL; }
struct Pair { int a; int b; };

void t_vec() {
    struct Vec<int> v = Vec::new();
    int i = 0;
    while (i < 1000) { v.push(i); i = i + 1; }
    ASSERT_EQ(v.length(), 1000);
    ASSERT_EQ(v.get(999UL).default(-1), 999);
    ASSERT_TRUE(v.get(1000UL).is_null());
    ASSERT_EQ(*v.at(10UL), 10);
    *v.at(10UL) = 77;                         // write through the reference
    ASSERT_EQ(v.get(10UL).default(-1), 77);
    v.set(10UL, 10);
    ASSERT_EQ(v.pop().default(-1), 999);
    ASSERT_TRUE(v.insert(0UL, -1));
    ASSERT_EQ(v.front().default(0), -1);
    ASSERT_EQ(v.remove(0UL).default(0), -1);
    ASSERT_TRUE(v.remove(5000UL).is_none());
    v.reverse();
    ASSERT_EQ(v.front().default(-1), 998);
    v.sort(cmp_int);
    ASSERT_EQ(v.front().default(-1), 0);
    struct Greater<int> desc = Greater::new();
    vec_sort_by(&v, desc);
    ASSERT_EQ(v.front().default(-1), 998);
    vec_sort(&v);
    ASSERT_EQ(v.front().default(-1), 0);
    ASSERT_EQ(v.back().default(-1), 998);
    int key = 500;
    ASSERT_EQ(v.find(&key, cmp_int), 500);
    ASSERT_TRUE(v.contains(&key, cmp_int));
    struct Vec<int> ev = v.filter(is_even);
    ASSERT_EQ(ev.length(), 500);
    struct Vec<long long> tr = vec_map(&v, triple);
    ASSERT_EQ(tr.get(998UL).default(0LL), 2994);
    struct Vec<int> c = v.clone();
    ASSERT_EQ(c.length(), v.length());
    int extra[3];
    extra[0] = 7; extra[1] = 8; extra[2] = 9;
    c.extend(&extra[0], 3UL);
    ASSERT_EQ(c.back().default(0), 9);
    // Large random input exercises the introsort partitioning paths.
    struct Vec<int> big = Vec::with_cap(20000UL);
    unsigned int seed = 12345u;
    i = 0;
    while (i < 20000) { seed = seed * 1103515245u + 12345u; big.push((int)(seed >> 8) % 1000); i = i + 1; }
    big.sort(cmp_int);
    int sorted = 1;
    unsigned long k = 1UL;
    while (k < big.length()) { if (*big.at(k - 1UL) > *big.at(k)) sorted = 0; k = k + 1UL; }
    ASSERT_TRUE(sorted);
    // Struct elements, mutated in place through match.
    struct Vec<struct Pair> ps = Vec::new();
    struct Pair p;
    p.a = 1; p.b = 2;
    ps.push(p);
    match (ps.get(0UL)) {
        case null: { ASSERT_TRUE(0); }
        case some(q): { q.a = 40; }
    }
    ASSERT_EQ(ps.at(0UL).a + ps.at(0UL).b, 42);
    ps.free(); big.free(); c.free(); tr.free(); ev.free();
    v.clear();
    ASSERT_TRUE(v.is_empty());
    v.free();
}

void t_string() {
    unsafe {
        struct String s = String::from("  Hello, World  ");
        struct String t = s.trim();
        ASSERT_TRUE(t.eq_cstr("Hello, World"));
        ASSERT_TRUE(t.starts_with("Hello"));
        ASSERT_TRUE(t.ends_with("World"));
        ASSERT_EQ(t.index_of("World"), 7);
        struct String up = t.to_upper();
        ASSERT_STR_EQ(up.as_ptr(), "HELLO, WORLD");
        t.replace_all("l", "L");
        ASSERT_STR_EQ(t.as_ptr(), "HeLLo, WorLd");
        t.push_int(-42LL);
        ASSERT_TRUE(t.ends_with("-42"));
        struct String parts[4];
        struct String csv = String::from("a,b,c");
        unsigned long n = csv.split(",", &parts[0], 4UL);
        ASSERT_EQ(n, 3);
        ASSERT_TRUE(parts[2].eq_cstr("c"));
        struct String joined = string_join("-", &parts[0], n);
        ASSERT_STR_EQ(joined.as_ptr(), "a-b-c");
        int ok = 0;
        struct String num = String::from("12345");
        ASSERT_EQ(num.parse_int(&ok), 12345);
        ASSERT_TRUE(ok);
        struct String big = String::new();
        int i = 0;
        while (i < 10000) { big.push_char('x'); i = i + 1; }
        ASSERT_EQ(big.length(), 10000);
        struct String sub = big.substr(10UL, 20UL);
        ASSERT_EQ(sub.length(), 10);
        ASSERT_EQ(String::from("abcabc").count("bc"), 2);
        sub.free(); big.free(); num.free(); joined.free(); csv.free();
        unsigned long k = 0UL;
        while (k < n) { parts[k].free(); k = k + 1UL; }
        up.free(); t.free(); s.free();
    }
}

void t_stack_queue() {
    struct Stack<int> st = Stack::new();
    int i = 0;
    while (i < 100) { st.push(i); i = i + 1; }
    ASSERT_EQ(st.pop().default(-1), 99);
    ASSERT_EQ(st.peek().default(-1), 98);
    ASSERT_EQ(st.length(), 99);
    st.clear();
    ASSERT_TRUE(st.pop().is_none());
    ASSERT_TRUE(st.peek().is_null());
    st.free();

    struct Queue<int> q = Queue::with_cap(4UL);
    i = 0;
    while (i < 100) { q.enqueue(i); i = i + 1; }
    ASSERT_EQ(q.dequeue().default(-1), 0);
    ASSERT_EQ(q.front().default(-1), 1);
    ASSERT_EQ(q.back().default(-1), 99);
    // wrap-around
    i = 0;
    while (i < 50) { q.dequeue(); i = i + 1; }
    i = 100;
    while (i < 150) { q.enqueue(i); i = i + 1; }
    ASSERT_EQ(q.length(), 99);
    ASSERT_EQ(q.dequeue().default(-1), 51);
    ASSERT_EQ(q.back().default(-1), 149);
    q.free();
    struct Queue<long long> e = Queue::new();
    ASSERT_TRUE(e.dequeue().is_none());
    e.enqueue(5LL);
    ASSERT_EQ(e.dequeue().default(0LL), 5);
    e.free();
}

void t_list() {
    struct List<int> l = List::new();
    int i = 0;
    while (i < 10) { l.push_back(i); i = i + 1; }
    l.push_front(-1);
    ASSERT_EQ(l.length(), 11);
    ASSERT_EQ(l.front().default(0), -1);
    ASSERT_EQ(l.back().default(0), 9);
    int five = 5;
    ASSERT_TRUE(l.contains(&five, cmp_int));
    ASSERT_TRUE(l.remove(&five, cmp_int));
    ASSERT_FALSE(l.contains(&five, cmp_int));
    ASSERT_EQ(l.length(), 10);
    l.reverse();
    ASSERT_EQ(l.front().default(0), 9);
    ASSERT_EQ(l.pop_back().default(0), -1);
    ASSERT_EQ(l.pop_front().default(0), 9);
    int three = 3;
    match (l.find(&three, cmp_int)) {
        case some(r): *r = 33;
        case null: ASSERT_TRUE(0);
    }
    ASSERT_TRUE(l.find(&three, cmp_int).is_null());
    l.free();
    ASSERT_TRUE(l.pop_front().is_none());
}

void t_map() {
    struct HashMap<int, long long> m = HashMap::new();
    int i = 0;
    while (i < 20000) { m.insert(i, (long long)i * 3LL); i = i + 1; }
    ASSERT_EQ(m.length(), 20000);
    ASSERT_EQ(m.get(12345).default(-1LL), 37035);
    m.insert(12345, 1LL);                       // overwrite
    ASSERT_EQ(m.length(), 20000);
    match (m.get(12345)) {
        case some(v): *v = 2LL;                 // write through the reference
        case null: ASSERT_TRUE(0);
    }
    ASSERT_EQ(m.get(12345).default(-1LL), 2);
    i = 0;
    while (i < 20000) { if (i % 2 == 0) m.remove(i); i = i + 1; }
    ASSERT_EQ(m.length(), 10000);
    ASSERT_FALSE(m.contains(4));
    ASSERT_TRUE(m.contains(5));
    ASSERT_EQ(m.remove(7).default(-1LL), 21);
    ASSERT_TRUE(m.remove(7).is_none());
    m.insert(4, 9LL);                           // reinsert after tombstones
    ASSERT_EQ(m.get(4).default(-1LL), 9);
    ASSERT_TRUE(m.get(-3).is_null());
    m.free();

    struct StrMap<int> sm = StrMap::new();
    sm.insert("one", 1);
    sm.insert("two", 2);
    ASSERT_EQ(sm.get("two").default(-1), 2);
    ASSERT_TRUE(sm.contains("one"));
    ASSERT_EQ(sm.remove("one").default(-1), 1);
    ASSERT_FALSE(sm.contains("one"));
    ASSERT_TRUE(sm.get("three").is_null());
    i = 0;
    while (i < 500) {
        struct String k = String::from("k");
        k.push_int((long long)i);
        sm.insert(k.as_ptr(), i);
        k.free();
        i = i + 1;
    }
    ASSERT_EQ(sm.length(), 501);
    ASSERT_EQ(sm.get("k321").default(-1), 321);
    sm.free();
}

static int cmp_ll(const &long long a, const &long long b) { return *a < *b ? -1 : (*a > *b ? 1 : 0); }
static int g_bst_prev = -1;
static int g_bst_sorted = 1;
static int g_bst_seen = 0;
static void bst_visit(const &int k, &int v) {
    if (*k <= g_bst_prev) g_bst_sorted = 0;
    g_bst_prev = *k;
    g_bst_seen = g_bst_seen + 1;
    *v = *v + 1;
}

void t_bst() {
    struct BST<int, int, Less<int>> t = BST::new();
    int keys[7];
    unsafe { keys[0] = 50; keys[1] = 30; keys[2] = 70; keys[3] = 20; keys[4] = 40; keys[5] = 60; keys[6] = 80; }
    int i = 0;
    while (i < 7) { int k0 = 0; unsafe { k0 = keys[i]; } t.insert(k0, k0); i = i + 1; }
    ASSERT_EQ(t.length(), 7);
    ASSERT_EQ(t.min_key().default(0), 20);
    ASSERT_EQ(t.max_key().default(0), 80);
    ASSERT_EQ(t.remove(30).default(0), 30);
    ASSERT_FALSE(t.contains(30));
    ASSERT_EQ(t.remove(50).default(0), 50);   // root with two children
    ASSERT_TRUE(t.remove(50).is_none());
    ASSERT_EQ(t.get(60).default(0), 60);
    ASSERT_EQ(t.length(), 5);
    t.foreach_inorder(bst_visit);
    ASSERT_TRUE(g_bst_sorted);
    ASSERT_EQ(g_bst_seen, 5);
    ASSERT_EQ(t.get(80).default(0), 81);      // visitor wrote through &V
    t.foreach_inorder(bst_visit);             // tree intact after threading
    ASSERT_EQ(g_bst_seen, 10);
    t.free();
    ASSERT_TRUE(t.min_key().is_null());

    // degenerate (sorted) inserts: iterative ops, no recursion
    struct BST<long long, int, FnOrder<long long>> d = BST::new_with(FnOrder::of(cmp_ll));
    long long j = 0LL;
    while (j < 3000LL) { d.insert(j, (int)j); j = j + 1LL; }
    ASSERT_EQ(d.get(2999LL).default(-1), 2999);
    ASSERT_EQ(d.remove(0LL).default(-1), 0);
    ASSERT_EQ(d.min_key().default(-1LL), 1);
    d.free();
}

static unsigned long g_bt_prev = 0UL;
static int g_bt_sorted = 1;
static void bt_visit(unsigned long k, &int v) {
    if (k < g_bt_prev) g_bt_sorted = 0;
    g_bt_prev = k;
}

void t_btree() {
    ?&heap BTree<int> tq = alloc(sizeof(struct BTree<int>));
    ASSERT_FALSE(tq.is_null());
    if (tq.is_null()) return;
    &BTree<int> t = std::expect_ref(tq, "tq is null");
    t.init();
    unsigned long i = 0UL;
    while (i < 200UL) { t.insert((i * 7919UL) % 1000UL, (int)i * 10); i = i + 1UL; }
    ASSERT_EQ(t.len(), 200);
    t.insert((3UL * 7919UL) % 1000UL, 30);          // overwrite keeps the count
    ASSERT_EQ(t.len(), 200);
    i = 0UL;
    int okc = 1;
    while (i < 200UL) {
        if (t.get((i * 7919UL) % 1000UL).default(-1) != (int)i * 10) okc = 0;
        i = i + 1UL;
    }
    ASSERT_TRUE(okc);
    ASSERT_EQ(t.remove((5UL * 7919UL) % 1000UL).default(-1), 50);
    ASSERT_FALSE(t.contains((5UL * 7919UL) % 1000UL));
    ASSERT_TRUE(t.remove(5000UL).is_none());
    ASSERT_EQ(t.len(), 199);
    t.foreach(bt_visit);
    ASSERT_TRUE(g_bt_sorted);
    // churn: remove everything and refill; merged nodes are recycled
    int round = 0;
    while (round < 5) {
        i = 0UL;
        while (i < 200UL) { t.remove((i * 7919UL) % 1000UL); i = i + 1UL; }
        ASSERT_EQ(t.len(), 0);
        i = 0UL;
        while (i < 200UL) { ASSERT_EQ(t.insert((i * 7919UL) % 1000UL, (int)i), 0); i = i + 1UL; }
        round = round + 1;
    }
    ASSERT_EQ(t.get((7UL * 7919UL) % 1000UL).default(-1), 7);
    dealloc(tq);
}

static unsigned char ring_storage[16];

static int g_sv_buf[4];
static struct StaticMapSlot<long long> g_sm_slots[8];

void t_static() {
    struct StaticVec<int> v = StaticVec::over(&g_sv_buf[0], 4UL);
    int i = 0;
    while (i < 4) { ASSERT_TRUE(v.push(i * 10)); i = i + 1; }
    ASSERT_FALSE(v.push(99));
    ASSERT_TRUE(v.is_full());
    ASSERT_EQ(v.top().default(-1), 30);
    ASSERT_EQ(v.pop().default(-1), 30);
    ASSERT_EQ(v.get(1UL).default(-1), 10);
    ASSERT_TRUE(v.get(3UL).is_null());

    struct StaticMap<long long> m = StaticMap::over(&g_sm_slots[0], 8UL);
    unsigned long k = 0UL;
    while (k < 8UL) { ASSERT_TRUE(m.insert(k * 7UL, (long long)k)); k = k + 1UL; }
    ASSERT_FALSE(m.insert(1000UL, 1LL));    // full
    ASSERT_TRUE(m.insert(14UL, 42LL));      // update in place
    ASSERT_EQ(m.get(14UL).default(-1LL), 42);
    ASSERT_EQ(m.length(), 8);
    ASSERT_TRUE(m.get(3UL).is_null());
}

void t_ring() {
    unsafe {
        struct RingBuffer r = RingBuffer::over((&static unsigned char)ring_storage, 16UL);
        unsigned char data[10];
        int i = 0;
        while (i < 10) { data[i] = (unsigned char)i; i = i + 1; }
        ASSERT_EQ(r.write(data, 10UL), 10);
        ASSERT_EQ(r.readable(), 10);
        unsigned char out[16];
        ASSERT_EQ(r.read(out, 6UL), 6);
        ASSERT_EQ(out[5], 5);
        ASSERT_EQ(r.write(data, 10UL), 10);   // wraps
        ASSERT_EQ(r.readable(), 14);
        ASSERT_EQ(r.write(data, 10UL), 2);    // only 2 bytes free
        ASSERT_TRUE(r.is_full());
        r.read(out, 4UL);
        ASSERT_EQ(out[0], 6);
        ASSERT_EQ(out[3], 9);
    }
}

void t_slice() {
    int arr[8];
    int i = 0;
    while (i < 8) { unsafe { arr[i] = i * i; } i = i + 1; }
    struct Slice<int> s = Slice::of(&arr[0], 8UL);
    ASSERT_TRUE(s.in_bounds(7UL));
    ASSERT_FALSE(s.in_bounds(8UL));
    ASSERT_TRUE(s.get(8UL).is_null());
    struct Slice<int> sub = s.sub(2UL, 5UL);
    ASSERT_EQ(sub.length(), 3);
    ASSERT_EQ(sub.get(0UL).default(-1), 4);
    ASSERT_TRUE(sub.get(3UL).is_null());
    ASSERT_TRUE(sub.set(1UL, 99));
    unsafe { ASSERT_EQ(arr[3], 99); }
    ASSERT_TRUE(s.sub(5UL, 9UL).is_empty());
    ASSERT_EQ(*s.at(7UL), 49);
    ASSERT_EQ(arr_max(&arr[0], 8UL).default(0), 99);
    ASSERT_EQ(arr_min(&arr[0], 8UL).default(-1), 0);
    arr_reverse(&arr[0], 8UL);
    unsafe { ASSERT_EQ(arr[0], 49); }
    ASSERT_TRUE(arr_at(&arr[0], 8UL, 8UL).is_null());
    struct Slice<long long> h = Slice::alloc(16UL);
    ASSERT_EQ(h.length(), 16);
    ASSERT_EQ(h.get(15UL).default(-1LL), 0);
    h.free();
}

// String methods and capacity APIs the main cases don't reach.
void t_string_more() {
    unsafe {
        struct String s = String::from_n("abcdefgh", 5UL);
        ASSERT_STR_EQ(s.as_ptr(), "abcde");
        ASSERT_TRUE(s.reserve(100UL));
        ASSERT_TRUE(s.cap >= 106UL);
        s.shrink_to_fit();
        ASSERT_EQ(s.cap, 6);
        ASSERT_STR_EQ(s.as_ptr(), "abcde");
        struct String t = String::from("-x");
        s.push_str(&t);
        s.push_uint(18446744073709551615ULL);
        s.push_bool(0);
        ASSERT_STR_EQ(s.as_ptr(), "abcde-x18446744073709551615false");
        s.truncate(7UL);
        ASSERT_STR_EQ(s.as_ptr(), "abcde-x");
        s.truncate(100UL);                          // longer: no-op
        ASSERT_EQ(s.length(), 7);
        ASSERT_TRUE(s.remove_range(1UL, 3UL));
        ASSERT_STR_EQ(s.as_ptr(), "ade-x");
        ASSERT_FALSE(s.remove_range(9UL, 12UL));
        ASSERT_TRUE(s.replace("de", "XYZ"));
        ASSERT_STR_EQ(s.as_ptr(), "aXYZ-x");
        ASSERT_FALSE(s.replace("nope", "?"));
        ASSERT_EQ(s.find_char('Z'), 3);
        ASSERT_EQ(s.find_char('q'), -1);
        struct String low = s.to_lower();
        ASSERT_STR_EQ(low.as_ptr(), "axyz-x");
        ASSERT_TRUE(low.eq_ignore_case(&s));
        ASSERT_TRUE(low.lt(&s) == 0);
        ASSERT_TRUE(s.lt(&low));                    // 'X' < 'x'
        struct String pad = low.pad_left(9UL, '*');
        ASSERT_STR_EQ(pad.as_ptr(), "***axyz-x");
        struct String sp = pad.strip_prefix("***");
        ASSERT_STR_EQ(sp.as_ptr(), "axyz-x");
        struct String rep = String::from("ab").repeat(3UL);
        ASSERT_STR_EQ(rep.as_ptr(), "ababab");
        struct String ws = String::from("  \t lead");
        struct String tl = ws.trim_left();
        ASSERT_STR_EQ(tl.as_ptr(), "lead");
        ASSERT_TRUE(tl.is_ascii());
        ASSERT_TRUE(tl.is_alphanumeric());
        struct String ne = String::from("caf\xC3\xA9");
        ASSERT_FALSE(ne.is_ascii());
        struct String words = String::from("  one two\tthree  four ");
        struct String parts[3];
        ASSERT_EQ(words.split_whitespace(&parts[0], 3UL), 3);
        ASSERT_STR_EQ(parts[0].as_ptr(), "one");
        ASSERT_STR_EQ(parts[2].as_ptr(), "three  four");   // last slot takes the rest
        int ok = 0;
        struct String fl = String::from("-2.5e2");
        ASSERT_TRUE(fl.parse_float(&ok) == -250.0);
        ASSERT_TRUE(ok);
        struct String bad = String::from("2.5x");
        bad.parse_float(&ok);
        ASSERT_FALSE(ok);
        int i = 0;
        while (i < 3) { parts[i].free(); i = i + 1; }
        s.free(); t.free(); low.free(); pad.free(); sp.free(); rep.free(); ws.free();
        tl.free(); ne.free(); words.free(); fl.free(); bad.free();

        // reserve / with_cap variants
        int arr[4];
        arr[0] = 4; arr[1] = 3; arr[2] = 2; arr[3] = 1;
        struct Vec<int> v = Vec::from_arr(&arr[0], 4UL);
        ASSERT_EQ(v.length(), 4);
        ASSERT_EQ(*v.at(3UL), 1);
        ASSERT_TRUE(v.reserve(64UL));
        ASSERT_TRUE(v.total_capacity() >= 64UL);
        v.shrink();
        ASSERT_EQ(v.total_capacity(), 4);
        v.free();
        struct Stack<int> st = Stack::with_cap(16UL);
        ASSERT_TRUE(st.reserve(32UL));
        st.push(5);
        ASSERT_EQ(st.pop().default(0), 5);
        st.free();
        struct Queue<int> q = Queue::new();
        ASSERT_TRUE(q.reserve(10UL));
        q.enqueue(9);
        ASSERT_EQ(q.dequeue().default(0), 9);
        q.free();
        struct HashMap<int, int> m = HashMap::with_cap(100UL);
        ASSERT_TRUE(m.reserve(500UL));
        m.insert(3, 30);
        ASSERT_EQ(m.get(3).default(0), 30);
        m.free();

        int buf[5];
        arr_fill(&buf[0], 5UL, 7);
        ASSERT_EQ(buf[4], 7);
        ASSERT_TRUE(arr_set(&buf[0], 5UL, 2UL, 1));
        ASSERT_FALSE(arr_set(&buf[0], 5UL, 5UL, 1));
        int cp[5];
        arr_copy(&cp[0], &buf[0], 5UL);
        ASSERT_EQ(cp[2], 1);
        ASSERT_EQ(cp[0], 7);
    }
}

int main() {
    struct TestSuite t = test_suite_init();
    unsafe {
        t.add("vec", t_vec);
        t.add("string", t_string);
        t.add("stack+queue", t_stack_queue);
        t.add("list", t_list);
        t.add("hashmap", t_map);
        t.add("bst", t_bst);
        t.add("btree", t_btree);
        t.add("ringbuffer", t_ring);
        t.add("static_vec/static_map", t_static);
        t.add("slice", t_slice);
        t.add("string/capacity extras", t_string_more);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
