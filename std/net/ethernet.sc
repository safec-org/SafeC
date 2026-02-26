// SafeC Standard Library — Ethernet Layer Implementation
#pragma once
#include "ethernet.h"

int eth_parse(&stack PacketBuf pkt, &stack EthernetHdr hdr_out) {
    if (pkt.len < (unsigned long)ETH_HDR_LEN) { return -1; }
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data;
        int i = 0;
        while (i < NET_MAC_LEN) {
            hdr_out.dst[i] = d[i];
            hdr_out.src[i] = d[6 + i];
            i = i + 1;
        }
        hdr_out.ethertype = net_ntohs(
            (unsigned short)(((unsigned short)d[12] << 8) | (unsigned short)d[13]));
    }
    return 0;
}

void eth_build(&stack PacketBuf pkt, const unsigned char dst[NET_MAC_LEN],
               const unsigned char src[NET_MAC_LEN], unsigned short ethertype) {
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data;
        int i = 0;
        while (i < NET_MAC_LEN) {
            d[i]     = dst[i];
            d[6 + i] = src[i];
            i = i + 1;
        }
        unsigned short et = net_htons(ethertype);
        d[12] = (unsigned char)((et >> 8) & (unsigned short)0xFF);
        d[13] = (unsigned char)(et & (unsigned short)0xFF);
    }
    pkt.len = (unsigned long)ETH_HDR_LEN;
}
