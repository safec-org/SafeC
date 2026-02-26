// SafeC Standard Library — Ethernet Layer
// Parse and build Ethernet II frames. Freestanding-safe.
#pragma once
#include "net_core.h"

#define ETH_TYPE_IPV4   0x0800
#define ETH_TYPE_ARP    0x0806
#define ETH_TYPE_IPV6   0x86DD

#define ETH_HDR_LEN     14   // 6 dst + 6 src + 2 ethertype

struct EthernetHdr {
    unsigned char  dst[NET_MAC_LEN];
    unsigned char  src[NET_MAC_LEN];
    unsigned short ethertype;   // host byte order after parse
};

// Parse Ethernet header from packet (returns 0 on success, -1 if too short).
int eth_parse(&stack PacketBuf pkt, &stack EthernetHdr hdr_out);

// Write Ethernet header into packet at offset 0; sets pkt.len to ETH_HDR_LEN.
void eth_build(&stack PacketBuf pkt, const unsigned char dst[NET_MAC_LEN],
               const unsigned char src[NET_MAC_LEN], unsigned short ethertype);
