// SafeC Standard Library — DNS (minimal stub resolver)
// Builds a DNS A-record query packet; parses the first A-record from the reply.
// Freestanding-safe.
#pragma once
#include "udp.h"

#define DNS_PORT       53
#define DNS_MAX_NAME  255
#define DNS_MAX_MSG   512

// Build a DNS query for hostname `name` (null-terminated) into pkt.
// Returns the transaction ID used (caller must match reply).
unsigned short dns_query(&stack PacketBuf pkt,
                         const unsigned char eth_src[NET_MAC_LEN],
                         const unsigned char eth_dst[NET_MAC_LEN],
                         unsigned int ip_src, unsigned int ip_dns,
                         unsigned short src_port,
                         const char* name,
                         unsigned short txid);

// Parse a DNS reply in pkt (starting at UDP payload offset).
// On success fills `ip4_out` with the first A record and returns 1.
// Returns 0 on parse error / no A record.
int dns_parse_reply(&stack PacketBuf pkt, unsigned long udp_payload_offset,
                    unsigned short expected_txid,
                    &stack unsigned int ip4_out);
