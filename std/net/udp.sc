// SafeC Standard Library — UDP Implementation
#pragma once
#include "udp.h"
#include "ethernet.h"

int udp_parse(&stack PacketBuf pkt, unsigned long offset,
              &stack UdpHdr hdr_out) {
    if (pkt.len < offset + (unsigned long)UDP_HDR_LEN) { return -1; }
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + offset;
        hdr_out.src_port = net_ntohs((unsigned short)(((unsigned short)d[0] << 8)
                                                     | (unsigned short)d[1]));
        hdr_out.dst_port = net_ntohs((unsigned short)(((unsigned short)d[2] << 8)
                                                     | (unsigned short)d[3]));
        hdr_out.length   = net_ntohs((unsigned short)(((unsigned short)d[4] << 8)
                                                     | (unsigned short)d[5]));
        hdr_out.checksum = net_ntohs((unsigned short)(((unsigned short)d[6] << 8)
                                                     | (unsigned short)d[7]));
    }
    return 0;
}

unsigned long udp_build(&stack PacketBuf pkt, unsigned long offset,
                        unsigned short src_port, unsigned short dst_port,
                        unsigned short payload_len) {
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + offset;
        unsigned short sp = net_htons(src_port);
        unsigned short dp = net_htons(dst_port);
        unsigned short ln = net_htons((unsigned short)((unsigned short)UDP_HDR_LEN
                                                       + payload_len));
        d[0] = (unsigned char)((sp >> 8) & (unsigned short)0xFF);
        d[1] = (unsigned char)(sp & (unsigned short)0xFF);
        d[2] = (unsigned char)((dp >> 8) & (unsigned short)0xFF);
        d[3] = (unsigned char)(dp & (unsigned short)0xFF);
        d[4] = (unsigned char)((ln >> 8) & (unsigned short)0xFF);
        d[5] = (unsigned char)(ln & (unsigned short)0xFF);
        d[6] = (unsigned char)0; d[7] = (unsigned char)0;  // checksum = 0
    }
    unsigned long payload_start = offset + (unsigned long)UDP_HDR_LEN;
    if (pkt.len < payload_start + (unsigned long)payload_len) {
        pkt.len = payload_start + (unsigned long)payload_len;
    }
    return payload_start;
}

unsigned long udp_frame(&stack PacketBuf pkt,
                        const unsigned char eth_src[NET_MAC_LEN],
                        const unsigned char eth_dst[NET_MAC_LEN],
                        unsigned int ip_src, unsigned int ip_dst,
                        unsigned short src_port, unsigned short dst_port,
                        unsigned short payload_len) {
    pkt.reset();
    // Ethernet header.
    eth_build(pkt, eth_dst, eth_src, (unsigned short)ETH_TYPE_IPV4);
    // IPv4 header.
    unsigned short udp_total = (unsigned short)((unsigned short)UDP_HDR_LEN + payload_len);
    unsigned long ip_off = (unsigned long)ETH_HDR_LEN;
    ipv4_build(pkt, ip_off, (unsigned char)IP_PROTO_UDP, ip_src, ip_dst, udp_total);
    // UDP header.
    unsigned long udp_off = ip_off + (unsigned long)IPV4_HDR_LEN;
    unsigned long data_off = udp_build(pkt, udp_off, src_port, dst_port, payload_len);
    return data_off;
}
