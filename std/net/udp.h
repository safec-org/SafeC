// SafeC Standard Library — UDP Layer
// Freestanding-safe.
#pragma once
#include "ipv4.h"

#define UDP_HDR_LEN  8

struct UdpHdr {
    unsigned short src_port;   // host byte order
    unsigned short dst_port;   // host byte order
    unsigned short length;     // total UDP datagram length (header + payload)
    unsigned short checksum;
};

// Parse UDP header from pkt at byte offset `offset`.
// Returns 0 on success, -1 on truncation.
int udp_parse(&stack PacketBuf pkt, unsigned long offset,
              &stack UdpHdr hdr_out);

// Write UDP header into pkt at `offset`. payload_len = bytes of data after header.
// Checksum is left 0 (optional in IPv4).
// Returns byte offset of first payload byte.
unsigned long udp_build(&stack PacketBuf pkt, unsigned long offset,
                        unsigned short src_port, unsigned short dst_port,
                        unsigned short payload_len);

// Full frame: Ethernet + IPv4 + UDP header, then caller fills payload.
// Returns offset where payload should be written.
// pkt is reset first.
unsigned long udp_frame(&stack PacketBuf pkt,
                        const unsigned char eth_src[NET_MAC_LEN],
                        const unsigned char eth_dst[NET_MAC_LEN],
                        unsigned int ip_src, unsigned int ip_dst,
                        unsigned short src_port, unsigned short dst_port,
                        unsigned short payload_len);
