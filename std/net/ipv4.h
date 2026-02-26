// SafeC Standard Library — IPv4 Layer
// Parse, build, and checksum IPv4 headers. Freestanding-safe.
#pragma once
#include "net_core.h"

#define IPV4_HDR_LEN  20    // minimum (no options)

#define IP_PROTO_ICMP  1
#define IP_PROTO_TCP   6
#define IP_PROTO_UDP   17

struct Ipv4Hdr {
    unsigned char  ihl;       // header length in 32-bit words (usually 5)
    unsigned char  dscp;
    unsigned short total_len; // host byte order
    unsigned short id;
    unsigned short frag_off;  // host byte order (includes flags)
    unsigned char  ttl;
    unsigned char  proto;
    unsigned short checksum;  // 0 after parse (not verified here)
    unsigned int   src;       // network byte order
    unsigned int   dst;       // network byte order
};

// Compute Internet checksum over `len` bytes starting at `data`.
unsigned short ip_checksum(const unsigned char* data, unsigned long len);

// Parse IPv4 header from pkt at byte offset `offset`.
// Returns 0 on success, -1 on truncation or bad version.
int  ipv4_parse(&stack PacketBuf pkt, unsigned long offset,
                &stack Ipv4Hdr hdr_out);

// Write IPv4 header into pkt at byte offset `offset` (pkt must have capacity).
// Computes and fills checksum. Returns byte offset of first payload byte.
unsigned long ipv4_build(&stack PacketBuf pkt, unsigned long offset,
                         unsigned char proto, unsigned int src, unsigned int dst,
                         unsigned short payload_len);
