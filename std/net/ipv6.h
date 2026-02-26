// SafeC Standard Library — IPv6 Layer
// Parse and build IPv6 headers. Freestanding-safe.
#pragma once
#include "net_core.h"

#define IPV6_HDR_LEN  40   // fixed header length

#define IPV6_ADDR_LEN 16   // bytes in an IPv6 address

struct Ipv6Addr {
    unsigned char bytes[IPV6_ADDR_LEN];
};

struct Ipv6Hdr {
    unsigned int   ver_tc_fl;   // version(4) + traffic class(8) + flow label(20)
    unsigned short payload_len; // length of payload (after fixed header)
    unsigned char  next_hdr;    // next header type (same as IPv4 protocol)
    unsigned char  hop_limit;
    struct Ipv6Addr src;
    struct Ipv6Addr dst;
};

// Compare two IPv6 addresses. Returns 1 if equal.
int  ipv6_addr_eq(const struct Ipv6Addr* a, const struct Ipv6Addr* b);

// Is address the all-zeros (unspecified) address?
int  ipv6_addr_is_unspecified(const struct Ipv6Addr* a);

// Is address a loopback (::1)?
int  ipv6_addr_is_loopback(const struct Ipv6Addr* a);

// Is address a link-local (fe80::/10)?
int  ipv6_addr_is_link_local(const struct Ipv6Addr* a);

// Format IPv6 address into buf as "xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx\0" (40 bytes).
void ipv6_addr_str(const struct Ipv6Addr* addr, char* buf);

// Parse IPv6 header from pkt at byte offset `offset`.
// Returns 0 on success, -1 on truncation.
int  ipv6_parse(&stack PacketBuf pkt, unsigned long offset,
                &stack Ipv6Hdr hdr_out);

// Write IPv6 header into pkt at byte offset `offset`.
// Returns byte offset of first payload byte.
unsigned long ipv6_build(&stack PacketBuf pkt, unsigned long offset,
                          unsigned char next_hdr, unsigned char hop_limit,
                          const struct Ipv6Addr* src, const struct Ipv6Addr* dst,
                          unsigned short payload_len);

// IPv6-over-Ethernet full frame helper (Ethernet + IPv6 header, caller writes payload).
// Returns offset where payload should be written.
unsigned long ipv6_frame(&stack PacketBuf pkt,
                          const unsigned char eth_src[NET_MAC_LEN],
                          const unsigned char eth_dst[NET_MAC_LEN],
                          unsigned char next_hdr,
                          const struct Ipv6Addr* src,
                          const struct Ipv6Addr* dst,
                          unsigned short payload_len);
