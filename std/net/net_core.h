// SafeC Standard Library — Network Core
// Packet buffer and network interface abstraction. Freestanding-safe.
#pragma once

#define NET_MTU        1514   // Ethernet max frame size (excl. preamble/FCS)
#define NET_MAC_LEN       6
#define NET_IP4_LEN       4

// ── Byte-order conversions (network = big-endian) ─────────────────────────────
unsigned short net_htons(unsigned short v);
unsigned short net_ntohs(unsigned short v);
unsigned int   net_htonl(unsigned int v);
unsigned int   net_ntohl(unsigned int v);

// ── IP / MAC utilities ────────────────────────────────────────────────────────
// Format an IPv4 address into buf as "a.b.c.d\0" (at least 16 bytes).
void net_ip4_str(unsigned int ip_be, char* buf);

// Format a MAC address into buf as "xx:xx:xx:xx:xx:xx\0" (at least 18 bytes).
void net_mac_str(const unsigned char mac[NET_MAC_LEN], char* buf);

// Build an IPv4 address from four octets (returns network byte order).
unsigned int net_ip4(unsigned char a, unsigned char b,
                     unsigned char c, unsigned char d);

// ── PacketBuf ─────────────────────────────────────────────────────────────────

struct PacketBuf {
    unsigned char data[NET_MTU];
    unsigned long len;   // bytes in use

    // Return a pointer to byte at offset.
    void* at(unsigned long offset);

    // Zero the buffer and reset len.
    void  reset();
};

// ── NetIf — abstract network interface ───────────────────────────────────────
// Implementors fill in the function pointers and the mac[] field.

struct NetIf {
    unsigned char mac[NET_MAC_LEN];    // hardware MAC address
    unsigned int  ip4;                 // assigned IPv4 in network byte order
    unsigned int  gateway;             // default gateway IPv4 (network order)
    unsigned int  netmask;             // subnet mask (network order)

    // Transmit a packet.  fn(iface_ctx, buf, len) → 0 on success.
    void*         tx_fn;
    void*         iface_ctx;           // opaque driver context

    // Transmit a packet.  Returns 0 on success.
    int  tx(&stack PacketBuf pkt);
};
