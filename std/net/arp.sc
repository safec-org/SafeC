// SafeC Standard Library — ARP Implementation
#pragma once
#include "arp.h"
#include "ethernet.h"

// ARP packet size over Ethernet: 8 bytes fixed + 2*(6+4) = 28 bytes
#define ARP_PKT_LEN  28

void ArpTable::update(unsigned int ip4, const unsigned char mac[NET_MAC_LEN]) {
    // Find existing entry first.
    int i = 0;
    while (i < ARP_TABLE_SIZE) {
        if (self.entries[i].ip4 == ip4) {
            int j = 0;
            while (j < NET_MAC_LEN) {
                unsafe { self.entries[i].mac[j] = mac[j]; }
                j = j + 1;
            }
            return;
        }
        i = i + 1;
    }
    // Find empty slot.
    i = 0;
    while (i < ARP_TABLE_SIZE) {
        if (self.entries[i].ip4 == (unsigned int)0) {
            self.entries[i].ip4 = ip4;
            int j = 0;
            while (j < NET_MAC_LEN) {
                unsafe { self.entries[i].mac[j] = mac[j]; }
                j = j + 1;
            }
            return;
        }
        i = i + 1;
    }
    // Table full: evict slot 0 (simple FIFO).
    self.entries[0].ip4 = ip4;
    int j = 0;
    while (j < NET_MAC_LEN) {
        unsafe { self.entries[0].mac[j] = mac[j]; }
        j = j + 1;
    }
}

int ArpTable::lookup(unsigned int ip4, unsigned char mac_out[NET_MAC_LEN]) const {
    int i = 0;
    while (i < ARP_TABLE_SIZE) {
        if (self.entries[i].ip4 == ip4) {
            int j = 0;
            while (j < NET_MAC_LEN) {
                unsafe { mac_out[j] = self.entries[i].mac[j]; }
                j = j + 1;
            }
            return 1;
        }
        i = i + 1;
    }
    return 0;
}

void ArpTable::evict(unsigned int ip4) {
    int i = 0;
    while (i < ARP_TABLE_SIZE) {
        if (self.entries[i].ip4 == ip4) {
            self.entries[i].ip4 = (unsigned int)0;
            return;
        }
        i = i + 1;
    }
}

void ArpTable::clear() {
    int i = 0;
    while (i < ARP_TABLE_SIZE) {
        self.entries[i].ip4 = (unsigned int)0;
        i = i + 1;
    }
}

// ARP packet layout (Ethernet HW/proto type = 1/0x0800, HW len=6, proto len=4):
// [0..1]  HTYPE=1
// [2..3]  PTYPE=0x0800
// [4]     HLEN=6
// [5]     PLEN=4
// [6..7]  OP
// [8..13] SHA
// [14..17] SPA
// [18..23] THA
// [24..27] TPA

void arp_build_packet(&stack PacketBuf pkt,
                      unsigned short op,
                      const unsigned char sha[NET_MAC_LEN], unsigned int spa,
                      const unsigned char tha[NET_MAC_LEN], unsigned int tpa) {
    unsigned long base = pkt.len;   // write after Ethernet header
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + base;
        d[0]  = (unsigned char)0; d[1]  = (unsigned char)1;      // HTYPE
        d[2]  = (unsigned char)0x08; d[3] = (unsigned char)0x00; // PTYPE IPv4
        d[4]  = (unsigned char)6;    d[5] = (unsigned char)4;    // HLEN, PLEN
        unsigned short op_n = net_htons(op);
        d[6]  = (unsigned char)((op_n >> 8) & (unsigned short)0xFF);
        d[7]  = (unsigned char)(op_n & (unsigned short)0xFF);
        int i = 0;
        while (i < NET_MAC_LEN) { d[8  + i] = sha[i]; i = i + 1; }
        unsigned int spa_n = net_htonl(spa);
        d[14] = (unsigned char)((spa_n >> 24) & (unsigned int)0xFF);
        d[15] = (unsigned char)((spa_n >> 16) & (unsigned int)0xFF);
        d[16] = (unsigned char)((spa_n >>  8) & (unsigned int)0xFF);
        d[17] = (unsigned char)(spa_n & (unsigned int)0xFF);
        i = 0;
        while (i < NET_MAC_LEN) { d[18 + i] = tha[i]; i = i + 1; }
        unsigned int tpa_n = net_htonl(tpa);
        d[22] = (unsigned char)((tpa_n >> 24) & (unsigned int)0xFF);
        d[23] = (unsigned char)((tpa_n >> 16) & (unsigned int)0xFF);
        d[24] = (unsigned char)((tpa_n >>  8) & (unsigned int)0xFF);
        d[25] = (unsigned char)(tpa_n & (unsigned int)0xFF);
    }
    pkt.len = base + (unsigned long)ARP_PKT_LEN;
}

int arp_parse_packet(&stack PacketBuf pkt, unsigned long offset,
                     &stack unsigned short op,
                     unsigned char sha[NET_MAC_LEN], &stack unsigned int spa,
                     unsigned char tha[NET_MAC_LEN], &stack unsigned int tpa) {
    if (pkt.len < offset + (unsigned long)ARP_PKT_LEN) { return -1; }
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + offset;
        op = net_ntohs((unsigned short)(((unsigned short)d[6] << 8) | (unsigned short)d[7]));
        int i = 0;
        while (i < NET_MAC_LEN) { sha[i] = d[8 + i]; i = i + 1; }
        spa = net_ntohl(((unsigned int)d[14] << 24) | ((unsigned int)d[15] << 16)
                       | ((unsigned int)d[16] << 8) | (unsigned int)d[17]);
        i = 0;
        while (i < NET_MAC_LEN) { tha[i] = d[18 + i]; i = i + 1; }
        tpa = net_ntohl(((unsigned int)d[22] << 24) | ((unsigned int)d[23] << 16)
                       | ((unsigned int)d[24] << 8) | (unsigned int)d[25]);
    }
    return 0;
}
