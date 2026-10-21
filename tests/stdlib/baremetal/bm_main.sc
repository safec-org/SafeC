// Bare-metal stdlib test image. No OS, no console: results are left in the
// globals below and read back by the emulator harness (run.py) once main
// returns. Covers the freestanding-safe modules — core, collections,
// allocators, crypto, encoding, fixed-point DSP, net, serial, sync — on
// every ISA, including 64-bit arithmetic on 32-bit cores, soft-float, and
// RV32I/E without the M extension (multiply/divide via builtins).
#include <std/mem.h>
#include <std/panic.h>
#include <std/str.h>
#include <std/bit.h>
#include <std/ctype.h>
#include <std/fmt.h>
#include <std/convert.h>
#include <std/stdckdint.h>
#include <std/collections/vec.h>
#include <std/collections/map.h>
#include <std/collections/string.h>
#include <std/collections/list.h>
#include <std/collections/btree.h>
#include <std/collections/ringbuffer.h>
#include <std/alloc/bump.h>
#include <std/alloc/slab.h>
#include <std/alloc/pool.h>
#include <std/alloc/tlsf.h>
#include <std/crypto/sha256.h>
#include <std/crypto/sha512.h>
#include <std/crypto/hmac.h>
#include <std/crypto/aes.h>
#include <std/encoding/base64.h>
#include <std/encoding/utf8.h>
#include <std/dsp/fixed.h>
#include <std/net/ipv4.h>
#include <std/net/udp.h>
#include <std/serial/value.h>
#include <std/serial/json.h>
#include <std/sync/spinlock.h>

volatile int g_checks = 0;
volatile int g_fails = 0;
volatile int g_first_fail_line = 0;
volatile int g_done = 0;

static void check(int ok, int line) {
    g_checks = g_checks + 1;
    if (!ok) {
        if (g_fails == 0) { g_first_fail_line = line; }
        g_fails = g_fails + 1;
    }
}
#define CHECK(e) check((e) != 0, __LINE__)

static int str_is(const &char a, const &char b) { return std::str_cmp(a, b) == 0; }

static void hex(const &unsigned char d, unsigned long n, &char out) {
    unsafe {
        const &char digits = "0123456789abcdef";
        unsigned long i = 0UL;
        while (i < n) {
            out[i * 2UL] = digits[d[i] >> 4];
            out[i * 2UL + 1UL] = digits[d[i] & 15];
            i = i + 1UL;
        }
        out[n * 2UL] = (char)0;
    }
}

static void t_core() {
    CHECK(std::popcount64(0xF0F0F0F0F0F0F0F0ULL) == 32);
    CHECK(std::clz32(1u) == 31);
    CHECK(std::ctz64(1ULL << 40) == 40);
    CHECK(std::bswap64(0x0102030405060708ULL) == 0x0807060504030201ULL);
    CHECK(std::char_to_upper('q') == 'Q');
    unsafe {
        char buf[32];
        std::fmt_int(buf, 32UL, -9223372036854775807LL);
        CHECK(str_is(buf, "-9223372036854775807"));
        std::fmt_hex(buf, 32UL, 0xDEADBEEFCAFEULL);
        CHECK(std::str_find(buf, "deadbeefcafe") != null);
        int ok = 0;
        CHECK(std::str_to_int("-123456789012", &ok) == -123456789012LL);
        CHECK(ok);
        long long r = 0LL;
        CHECK(std::ckd_mul_i64(&r, 3037000500LL, 3037000500LL));   // overflows
        CHECK(!std::ckd_mul_i64(&r, 1000000007LL, 1000000009LL));
        CHECK(r == 1000000016000000063LL);
    }
    // 64-bit divide/modulo (a library call on every 32-bit core here).
    unsigned long long big = 18446744073709551557ULL;   // largest 64-bit prime
    CHECK(big % 1000000007ULL == 582343949ULL);
    CHECK(big / 3ULL == 6148914691236517185ULL);
    long long neg = -7000000000000LL;
    CHECK(neg / 7LL == -1000000000000LL);
    // Soft-float on cores without an FPU.
    double d = 1.5;
    d = d * 2.25 + 0.125;
    CHECK(d == 3.5);
    CHECK((int)(d * 1000.0) == 3500);
}

static int cmp_int(const &int a, const &int b) { return *a - *b; }

static void t_collections() {
    unsafe {
        struct Vec<int> v = std::Vec::new();
        int i = 0;
        while (i < 500) { v.push((i * 7919) % 500); i = i + 1; }
        v.sort(cmp_int);
        CHECK(*v.at(0UL) == 0);
        CHECK(*v.at(499UL) == 499);
        v.free();

        struct HashMap<int, int> m = std::HashMap::new();
        i = 0;
        while (i < 300) { m.insert(i, i * i); i = i + 1; }
        CHECK(m.get(17).default(-1) == 289);
        CHECK(m.remove(17).default(-1) == 289);
        CHECK(!m.contains(17));
        m.insert(17, 289);
        CHECK(m.length() == 300UL);
        m.free();

        struct String s = std::String::from("bare metal");
        s.push(" safec");
        struct String up = s.to_upper();
        CHECK(up.eq_cstr("BARE METAL SAFEC"));
        CHECK(s.index_of("metal") == 5LL);
        up.free();
        s.free();

        struct List<int> l = std::List::new();
        i = 0;
        while (i < 10) { l.push_back(i); i = i + 1; }
        CHECK(l.length() == 10UL);
        CHECK(l.pop_back().default(0) == 9);
        l.free();

        ?&heap BTree<int> tq = std::alloc(sizeof(struct BTree<int>));
        if (!tq.is_null()) {
            &BTree<int> t = std::expect_ref(tq, "tq is null");
            t.init();
            unsigned long j = 0UL;
            while (j < 50UL) { t.insert(j * 37UL % 101UL, (int)j); j = j + 1UL; }
            ?&int bv = t.get(10UL * 37UL % 101UL);
            CHECK(bv.default(-1) == 10);
            std::dealloc(tq);
        }
    }
}

static unsigned char g_ring_store[64];

static void t_ring_alloc() {
    unsafe {
        struct RingBuffer r = std::RingBuffer::over((&static unsigned char)g_ring_store, 64UL);
        unsigned char in[40];
        unsigned char out[40];
        int i = 0;
        while (i < 40) { in[i] = (unsigned char)i; i = i + 1; }
        r.write(in, 40UL);
        r.read(out, 30UL);
        r.write(in, 40UL);                       // wraps
        CHECK(r.readable() == 50UL);
        r.read(out, 10UL);
        CHECK(out[9] == 39);

        struct TlsfAllocator t = std::TlsfAllocator::new(32768UL);
        ?&heap void a = t.alloc(100UL, 8UL);
        ?&heap void b = t.alloc(3000UL, 8UL);
        CHECK(!a.is_null() && !b.is_null());
        match (a) { case null: { } case some(x): { t.dealloc(x); } }
        match (b) { case null: { } case some(x): { t.dealloc(x); } }
        ?&heap void c = t.alloc(30000UL, 8UL);   // fits only if freed blocks coalesced
        CHECK(!c.is_null());
        t.destroy();

        struct SlabAllocator sl = std::SlabAllocator::new(24UL, 8UL);
        ?&heap unsigned long s1 = std::alloc_t(&sl);   // through the Allocator trait
        CHECK(!s1.is_null());
        CHECK(sl.available() == 7UL);
        sl.destroy();

        struct PoolAllocator p = std::PoolAllocator::new(2048UL);
        unsigned long before = p.available();
        ?&heap void pa = p.alloc(200UL, 8UL);
        CHECK(!pa.is_null());
        match (pa) { case null: { } case some(x): { p.dealloc(x); } }
        CHECK(p.available() == before);
        p.destroy();
    }
}

static void t_crypto() {
    unsafe {
        unsigned char d[64];
        char h[160];
        std::sha256((const unsigned char*)"abc", 3UL, d);
        hex(d, 32UL, h);
        CHECK(str_is(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
        std::sha512((const unsigned char*)"abc", 3UL, d);
        hex(d, 64UL, h);
        CHECK(str_is(h, "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"));
        std::hmac_sha256((const unsigned char*)"Jefe", 4UL,
                         (const unsigned char*)"what do ya want for nothing?", 28UL, (unsigned char*)d);
        hex(d, 32UL, h);
        CHECK(str_is(h, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"));
        unsigned char key[16];
        unsigned char blk[16];
        int i = 0;
        while (i < 16) { key[i] = (unsigned char)i; blk[i] = (unsigned char)(i * 17); i = i + 1; }
        struct AesCtx c = std::AesCtx::new_128(key);
        c.encrypt_block(blk);
        hex(blk, 16UL, h);
        CHECK(str_is(h, "69c4e0d86a7b0430d8cdb78070b4c55a"));

        struct String e = std::base64_encode((const unsigned char*)"bare-metal!", 11UL);
        CHECK(e.eq_cstr("YmFyZS1tZXRhbCE="));
        e.free();
        CHECK(std::utf8_len("h€llo") == 5UL);
    }
}

static void t_fixed_net() {
    Fixed a = std::fixed_from_float(2.5);
    Fixed b = std::fixed_from_float(-1.25);
    CHECK(std::fixed_to_float(std::fixed_mul(a, b)) == -3.125);
    CHECK(std::fixed_to_float(std::fixed_div(a, b)) == -2.0);
    CHECK(std::fixed_to_float(std::fixed_sqrt(std::fixed_from_int(100))) == 10.0);
    unsafe {
        unsigned char mac[6];
        int i = 0;
        while (i < 6) { mac[i] = (unsigned char)i; i = i + 1; }
        struct PacketBuf pkt;
        pkt.reset();
        unsigned long off = std::udp_frame(&pkt, mac, mac,
            std::net_ip4((unsigned char)10, (unsigned char)0, (unsigned char)0, (unsigned char)1),
            std::net_ip4((unsigned char)10, (unsigned char)0, (unsigned char)0, (unsigned char)2),
            (unsigned short)1234, (unsigned short)5678, (unsigned short)0);
        CHECK(off == 42UL);
        CHECK(pkt.data[14 + 12] == 10);
        CHECK(std::ip_checksum((const unsigned char*)&pkt.data[14], 20UL) == (unsigned short)0);
    }
}

static void t_serial_sync() {
    unsafe {
        struct Value o = std::Value::object();
        o.object_set("n", std::Value::from_int(-42LL));
        o.object_set("s", std::Value::from_str("bm"));
        struct String js = std::value_to_json(&o);
        int ok = 0;
        struct Value back = std::json_parse(js.as_ptr(), &ok);
        CHECK(ok);
        ?&Value n = back.object_get("n");
        CHECK(!n.is_null());
        if (!n.is_null()) CHECK(n->as_int() == -42LL);
        js.free();
        back.free();
        o.free();

        struct Spinlock sp = std::Spinlock::new();
        sp.lock();
        CHECK(sp.is_locked());
        CHECK(!sp.trylock());
        sp.unlock();
        CHECK(sp.trylock());
        sp.unlock();
    }
}

int main() {
    t_core();
    t_collections();
    t_ring_alloc();
    t_crypto();
    t_fixed_net();
    t_serial_sync();
    g_done = 1;
    return g_fails;
}
