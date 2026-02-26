// SafeC Standard Library — IPv6 Layer Implementation
#pragma once
#include "ipv6.h"
#include "ethernet.h"

// ── ipv6_addr_eq ──────────────────────────────────────────────────────────────

int ipv6_addr_eq(const struct Ipv6Addr* a, const struct Ipv6Addr* b) {
    unsafe {
        int i = 0;
        while (i < IPV6_ADDR_LEN) {
            if (a->bytes[i] != b->bytes[i]) { return 0; }
            i = i + 1;
        }
    }
    return 1;
}

// ── ipv6_addr_is_unspecified ──────────────────────────────────────────────────

int ipv6_addr_is_unspecified(const struct Ipv6Addr* a) {
    unsafe {
        int i = 0;
        while (i < IPV6_ADDR_LEN) {
            if (a->bytes[i] != (unsigned char)0) { return 0; }
            i = i + 1;
        }
    }
    return 1;
}

// ── ipv6_addr_is_loopback ─────────────────────────────────────────────────────
// ::1 = 15 zero bytes followed by 0x01.

int ipv6_addr_is_loopback(const struct Ipv6Addr* a) {
    unsafe {
        int i = 0;
        while (i < 15) {
            if (a->bytes[i] != (unsigned char)0) { return 0; }
            i = i + 1;
        }
        if (a->bytes[15] != (unsigned char)1) { return 0; }
    }
    return 1;
}

// ── ipv6_addr_is_link_local ───────────────────────────────────────────────────
// fe80::/10 — top 10 bits are 1111111010.
// bytes[0] == 0xFE, bytes[1] & 0xC0 == 0x80

int ipv6_addr_is_link_local(const struct Ipv6Addr* a) {
    unsafe {
        if (a->bytes[0] != (unsigned char)0xFE) { return 0; }
        if ((a->bytes[1] & (unsigned char)0xC0) != (unsigned char)0x80) { return 0; }
    }
    return 1;
}

// ── Hex nibble helper ─────────────────────────────────────────────────────────

static char hex_nibble_(unsigned int v) {
    v = v & (unsigned int)0xF;
    if (v < (unsigned int)10) { return (char)('0' + v); }
    return (char)('a' + (v - (unsigned int)10));
}

// ── ipv6_addr_str ─────────────────────────────────────────────────────────────
// Writes 8 groups of 4 hex digits separated by ':'.
// No :: compression — always 39 printable chars + NUL (40 bytes total).

void ipv6_addr_str(const struct Ipv6Addr* addr, char* buf) {
    unsigned long pos = (unsigned long)0;
    int group = 0;
    unsafe {
        while (group < 8) {
            unsigned int hi = (unsigned int)addr->bytes[group * 2];
            unsigned int lo = (unsigned int)addr->bytes[group * 2 + 1];
            unsigned int word = (hi << 8) | lo;
            buf[pos] = hex_nibble_((word >> 12) & (unsigned int)0xF); pos = pos + (unsigned long)1;
            buf[pos] = hex_nibble_((word >>  8) & (unsigned int)0xF); pos = pos + (unsigned long)1;
            buf[pos] = hex_nibble_((word >>  4) & (unsigned int)0xF); pos = pos + (unsigned long)1;
            buf[pos] = hex_nibble_( word        & (unsigned int)0xF); pos = pos + (unsigned long)1;
            if (group < 7) {
                buf[pos] = ':';
                pos = pos + (unsigned long)1;
            }
            group = group + 1;
        }
        buf[pos] = (char)0;
    }
}

// ── ipv6_parse ────────────────────────────────────────────────────────────────

int ipv6_parse(&stack PacketBuf pkt, unsigned long offset,
               &stack Ipv6Hdr hdr_out) {
    if (pkt.len < offset + (unsigned long)IPV6_HDR_LEN) { return -1; }
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + offset;
        // ver_tc_fl: big-endian 32 bits.
        hdr_out.ver_tc_fl = ((unsigned int)d[0] << 24)
                          | ((unsigned int)d[1] << 16)
                          | ((unsigned int)d[2] <<  8)
                          |  (unsigned int)d[3];
        if ((hdr_out.ver_tc_fl >> 28) != (unsigned int)6) { return -1; }
        hdr_out.payload_len = net_ntohs(
            (unsigned short)(((unsigned short)d[4] << 8) | (unsigned short)d[5]));
        hdr_out.next_hdr  = d[6];
        hdr_out.hop_limit = d[7];
        int i = 0;
        while (i < IPV6_ADDR_LEN) {
            hdr_out.src.bytes[i] = d[8  + i];
            hdr_out.dst.bytes[i] = d[24 + i];
            i = i + 1;
        }
    }
    return 0;
}

// ── ipv6_build ────────────────────────────────────────────────────────────────

unsigned long ipv6_build(&stack PacketBuf pkt, unsigned long offset,
                          unsigned char next_hdr, unsigned char hop_limit,
                          const struct Ipv6Addr* src, const struct Ipv6Addr* dst,
                          unsigned short payload_len) {
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + offset;
        // Version=6, TC=0, Flow Label=0.
        unsigned int vtcfl = (unsigned int)6 << 28;
        d[0] = (unsigned char)((vtcfl >> 24) & (unsigned int)0xFF);
        d[1] = (unsigned char)((vtcfl >> 16) & (unsigned int)0xFF);
        d[2] = (unsigned char)((vtcfl >>  8) & (unsigned int)0xFF);
        d[3] = (unsigned char)( vtcfl        & (unsigned int)0xFF);
        unsigned short plen = net_htons(payload_len);
        d[4] = (unsigned char)((plen >> 8) & (unsigned short)0xFF);
        d[5] = (unsigned char)( plen       & (unsigned short)0xFF);
        d[6] = next_hdr;
        d[7] = hop_limit;
        int i = 0;
        while (i < IPV6_ADDR_LEN) {
            d[8  + i] = src->bytes[i];
            d[24 + i] = dst->bytes[i];
            i = i + 1;
        }
    }
    unsigned long payload_start = offset + (unsigned long)IPV6_HDR_LEN;
    if (pkt.len < payload_start + (unsigned long)payload_len) {
        pkt.len = payload_start + (unsigned long)payload_len;
    }
    return payload_start;
}

// ── ipv6_frame ────────────────────────────────────────────────────────────────
// Ethernet ethertype for IPv6 is 0x86DD.

#define ETHERTYPE_IPV6  0x86DD

unsigned long ipv6_frame(&stack PacketBuf pkt,
                          const unsigned char eth_src[NET_MAC_LEN],
                          const unsigned char eth_dst[NET_MAC_LEN],
                          unsigned char next_hdr,
                          const struct Ipv6Addr* src,
                          const struct Ipv6Addr* dst,
                          unsigned short payload_len) {
    // Write Ethernet header (14 bytes).
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data;
        int i = 0;
        while (i < NET_MAC_LEN) {
            d[i]     = eth_dst[i];
            d[6 + i] = eth_src[i];
            i = i + 1;
        }
        unsigned short et = net_htons((unsigned short)ETHERTYPE_IPV6);
        d[12] = (unsigned char)((et >> 8) & (unsigned short)0xFF);
        d[13] = (unsigned char)( et       & (unsigned short)0xFF);
    }
    pkt.len = (unsigned long)14;

    // Write IPv6 header after Ethernet header.
    return ipv6_build(pkt, (unsigned long)14,
                      next_hdr, (unsigned char)64,
                      src, dst, payload_len);
}
