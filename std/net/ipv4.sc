// SafeC Standard Library — IPv4 Implementation
#pragma once
#include "ipv4.h"

unsigned short ip_checksum(const unsigned char* data, unsigned long len) {
    unsigned long sum = (unsigned long)0;
    unsigned long i = (unsigned long)0;
    unsafe {
        while (i + (unsigned long)1 < len) {
            unsigned long word = ((unsigned long)data[i] << 8)
                               | (unsigned long)data[i + (unsigned long)1];
            sum = sum + word;
            i = i + (unsigned long)2;
        }
        if (i < len) {
            sum = sum + ((unsigned long)data[i] << 8);
        }
    }
    // Fold 32-bit sum to 16 bits.
    while ((sum >> 16) != (unsigned long)0) {
        sum = (sum & (unsigned long)0xFFFF) + (sum >> 16);
    }
    return (unsigned short)(~sum & (unsigned long)0xFFFF);
}

int ipv4_parse(&stack PacketBuf pkt, unsigned long offset,
               &stack Ipv4Hdr hdr_out) {
    if (pkt.len < offset + (unsigned long)IPV4_HDR_LEN) { return -1; }
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + offset;
        unsigned char ver_ihl = d[0];
        if ((ver_ihl >> 4) != (unsigned char)4) { return -1; }
        hdr_out.ihl      = ver_ihl & (unsigned char)0xF;
        hdr_out.dscp     = d[1];
        hdr_out.total_len= net_ntohs((unsigned short)(((unsigned short)d[2] << 8)
                                                     | (unsigned short)d[3]));
        hdr_out.id       = net_ntohs((unsigned short)(((unsigned short)d[4] << 8)
                                                     | (unsigned short)d[5]));
        hdr_out.frag_off = net_ntohs((unsigned short)(((unsigned short)d[6] << 8)
                                                     | (unsigned short)d[7]));
        hdr_out.ttl      = d[8];
        hdr_out.proto    = d[9];
        hdr_out.checksum = (unsigned short)0;
        hdr_out.src = ((unsigned int)d[12] << 24) | ((unsigned int)d[13] << 16)
                    | ((unsigned int)d[14] << 8)  |  (unsigned int)d[15];
        hdr_out.dst = ((unsigned int)d[16] << 24) | ((unsigned int)d[17] << 16)
                    | ((unsigned int)d[18] << 8)  |  (unsigned int)d[19];
    }
    return 0;
}

unsigned long ipv4_build(&stack PacketBuf pkt, unsigned long offset,
                         unsigned char proto, unsigned int src, unsigned int dst,
                         unsigned short payload_len) {
    unsigned short total = (unsigned short)((unsigned short)IPV4_HDR_LEN + payload_len);
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + offset;
        d[0] = (unsigned char)0x45;  // version=4, IHL=5
        d[1] = (unsigned char)0;     // DSCP
        unsigned short tlen = net_htons(total);
        d[2] = (unsigned char)((tlen >> 8) & (unsigned short)0xFF);
        d[3] = (unsigned char)(tlen & (unsigned short)0xFF);
        d[4] = (unsigned char)0; d[5] = (unsigned char)0; // id
        d[6] = (unsigned char)0x40; d[7] = (unsigned char)0; // DF flag, no frag
        d[8] = (unsigned char)64;   // TTL
        d[9] = proto;
        d[10]= (unsigned char)0; d[11]= (unsigned char)0; // checksum placeholder
        unsigned int s = net_htonl(src);
        d[12]= (unsigned char)((s >> 24) & (unsigned int)0xFF);
        d[13]= (unsigned char)((s >> 16) & (unsigned int)0xFF);
        d[14]= (unsigned char)((s >>  8) & (unsigned int)0xFF);
        d[15]= (unsigned char)(s & (unsigned int)0xFF);
        unsigned int ds = net_htonl(dst);
        d[16]= (unsigned char)((ds >> 24) & (unsigned int)0xFF);
        d[17]= (unsigned char)((ds >> 16) & (unsigned int)0xFF);
        d[18]= (unsigned char)((ds >>  8) & (unsigned int)0xFF);
        d[19]= (unsigned char)(ds & (unsigned int)0xFF);
        // Fill checksum.
        unsigned short csum = ip_checksum((const unsigned char*)d,
                                          (unsigned long)IPV4_HDR_LEN);
        unsigned short csum_n = net_htons(csum);
        d[10]= (unsigned char)((csum_n >> 8) & (unsigned short)0xFF);
        d[11]= (unsigned char)(csum_n & (unsigned short)0xFF);
    }
    unsigned long payload_start = offset + (unsigned long)IPV4_HDR_LEN;
    if (pkt.len < payload_start + (unsigned long)payload_len) {
        pkt.len = payload_start + (unsigned long)payload_len;
    }
    return payload_start;
}
