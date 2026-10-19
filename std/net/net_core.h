// SafeC Standard Library — Network Core
// Packet buffer and network interface abstraction. Freestanding-safe.
#pragma once

#define NET_MTU        1514   // Ethernet max frame size (excl. preamble/FCS)
#define NET_MAC_LEN       6
#define NET_IP4_LEN       4

// ── Byte-order conversions (network = big-endian) ─────────────────────────────
namespace std {

unsigned short net_htons(unsigned short v);
unsigned short net_ntohs(unsigned short v);
unsigned int   net_htonl(unsigned int v);
unsigned int   net_ntohl(unsigned int v);

// ── IP / MAC utilities ────────────────────────────────────────────────────────
// Format an IPv4 address into buf as "a.b.c.d\0" (at least 16 bytes).
void net_ip4_str(unsigned int ip_be, char* buf);

// Format a MAC address into buf as "xx:xx:xx:xx:xx:xx\0" (at least 18 bytes).
void net_mac_str(const unsigned char mac[NET_MAC_LEN], char* buf);

// Build an IPv4 address from four octets: a<<24 | b<<16 | c<<8 | d.
// Every std::net API takes and returns IPv4 addresses (and ports, seq
// numbers, …) as plain host integers in this numeric form; the build/parse
// functions do the big-endian wire (de)serialization themselves, so callers
// never need net_htons/net_htonl for them.
unsigned int net_ip4(unsigned char a, unsigned char b,
                     unsigned char c, unsigned char d);

// ── PacketBuf ─────────────────────────────────────────────────────────────────

struct PacketBuf {
    unsigned char data[NET_MTU];
    unsigned long len;   // bytes in use

    // The byte at `offset`, or null past the end of the buffer.
    ?&unsigned char at(unsigned long offset);

    // Zero the buffer and reset len.
    void  reset();
};

// ── NetIf — abstract network interface ───────────────────────────────────────
// Implementors fill in the function pointers and the mac[] field.

struct NetIf {
    unsigned char mac[NET_MAC_LEN];    // hardware MAC address
    unsigned int  ip4;                 // assigned IPv4, numeric form (see net_ip4)
    unsigned int  gateway;             // default gateway IPv4 (numeric form, see net_ip4)
    unsigned int  netmask;             // subnet mask (numeric form, see net_ip4)

    // Driver transmit callback and context, stored type-erased (one stack
    // may run interfaces from different drivers); netif_new<T> is the
    // only writer and always pairs fn int(&T, ...) with a &T.
    void*         tx_fn;               // fn int(&T, const &unsigned char buf, unsigned long len)
    void*         iface_ctx;           // &T

    // Transmit a packet.  Returns 0 on success.
    int  tx(&stack PacketBuf pkt);
};

// An interface whose tx() calls tx(ctx, frame, len) (0 on success).
generic<T>
struct NetIf netif_new(&T ctx, fn int(&T, const &unsigned char, unsigned long) tx) {
    struct NetIf nif;
    unsigned long i = 0UL;
    while (i < (unsigned long)NET_MAC_LEN) { unsafe { nif.mac[i] = (unsigned char)0; } i = i + 1UL; }
    nif.ip4 = 0U;
    nif.gateway = 0U;
    nif.netmask = 0U;
    unsafe {
        nif.tx_fn = (void*)tx;
        nif.iface_ctx = (void*)ctx;
    }
    return nif;
}

} // namespace std
