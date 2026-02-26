// SafeC Standard Library — DHCP Client (Discover/Offer/Request/Ack)
// Freestanding-safe; implements the 4-step DORA handshake.
#pragma once
#include "udp.h"

#define DHCP_SERVER_PORT  67
#define DHCP_CLIENT_PORT  68
#define DHCP_MAGIC_COOKIE 0x63825363U

// DHCP message types (option 53)
#define DHCP_DISCOVER  1
#define DHCP_OFFER     2
#define DHCP_REQUEST   3
#define DHCP_DECLINE   4
#define DHCP_ACK       5
#define DHCP_NAK       6
#define DHCP_RELEASE   7
#define DHCP_INFORM    8

// DHCP client state machine
#define DHCP_STATE_IDLE       0
#define DHCP_STATE_SELECTING  1
#define DHCP_STATE_REQUESTING 2
#define DHCP_STATE_BOUND      3

struct DhcpLease {
    unsigned int  your_ip;       // offered/assigned IPv4 (network order)
    unsigned int  server_ip;     // DHCP server IPv4 (network order)
    unsigned int  gateway;       // default gateway (network order)
    unsigned int  subnet_mask;   // subnet mask (network order)
    unsigned int  dns;           // DNS server (network order)
    unsigned int  lease_secs;    // lease duration in seconds
};

struct DhcpClient {
    int           state;
    unsigned int  xid;           // transaction ID
    struct DhcpLease lease;
    unsigned char mac[NET_MAC_LEN];

    // Build a DHCPDISCOVER packet.
    void discover(&stack PacketBuf pkt,
                  const unsigned char eth_dst[NET_MAC_LEN]);

    // Build a DHCPREQUEST for `offered_ip` from `server_ip`.
    void request(&stack PacketBuf pkt,
                 const unsigned char eth_dst[NET_MAC_LEN],
                 unsigned int offered_ip, unsigned int server_ip);

    // Parse an incoming DHCP reply; fills lease on ACK/OFFER.
    // Returns DHCP_OFFER, DHCP_ACK, DHCP_NAK, or 0 on unrecognised.
    int  parse_reply(&stack PacketBuf pkt, unsigned long udp_payload_offset);

    // Is the client bound (has a valid lease)?
    int  is_bound() const;
};
