// SafeC Standard Library — ARP (Address Resolution Protocol)
// Freestanding-safe; uses a fixed-size table.
#pragma once
#include "net_core.h"

#define ARP_TABLE_SIZE  16   // max cached entries

#define ARP_OP_REQUEST  1
#define ARP_OP_REPLY    2

struct ArpEntry {
    unsigned int  ip4;                  // IPv4 in network byte order (0 = empty)
    unsigned char mac[NET_MAC_LEN];
};

struct ArpTable {
    struct ArpEntry entries[ARP_TABLE_SIZE];

    // Add or update an entry.
    void  update(unsigned int ip4, const unsigned char mac[NET_MAC_LEN]);

    // Look up MAC for ip4.  Returns 1 on hit (mac_out filled), 0 on miss.
    int   lookup(unsigned int ip4, unsigned char mac_out[NET_MAC_LEN]) const;

    // Evict an entry.
    void  evict(unsigned int ip4);

    // Clear all entries.
    void  clear();
};

// Build an ARP request/reply packet into pkt (starting after Ethernet header).
// pkt.len must already include the Ethernet header length (ETH_HDR_LEN).
void arp_build_packet(&stack PacketBuf pkt,
                      unsigned short op,
                      const unsigned char sha[NET_MAC_LEN], unsigned int spa,
                      const unsigned char tha[NET_MAC_LEN], unsigned int tpa);

// Parse an ARP packet starting at pkt.data + offset.
// Fills op, sha, spa, tha, tpa on success (returns 0), -1 on bad length.
int  arp_parse_packet(&stack PacketBuf pkt, unsigned long offset,
                      &stack unsigned short op,
                      unsigned char sha[NET_MAC_LEN], &stack unsigned int spa,
                      unsigned char tha[NET_MAC_LEN], &stack unsigned int tpa);
