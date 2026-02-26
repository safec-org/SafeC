// SafeC Standard Library — DNS Implementation
#pragma once
#include "dns.h"

// Encode `name` (e.g. "example.com") as DNS labels into `buf`.
// Returns number of bytes written.
static unsigned long dns_encode_name_(const char* name, unsigned char* buf) {
    unsigned long pos = (unsigned long)0;
    unsigned long label_len_pos = pos;
    pos = pos + (unsigned long)1;
    unsigned long label_len = (unsigned long)0;
    unsigned long i = (unsigned long)0;
    unsafe {
        while (name[i] != (char)0) {
            if (name[i] == '.') {
                buf[label_len_pos] = (unsigned char)label_len;
                label_len_pos = pos;
                pos = pos + (unsigned long)1;
                label_len = (unsigned long)0;
            } else {
                buf[pos] = (unsigned char)name[i];
                pos = pos + (unsigned long)1;
                label_len = label_len + (unsigned long)1;
            }
            i = i + (unsigned long)1;
        }
        buf[label_len_pos] = (unsigned char)label_len;
        buf[pos] = (unsigned char)0;  // root label
        pos = pos + (unsigned long)1;
    }
    return pos;
}

unsigned short dns_query(&stack PacketBuf pkt,
                         const unsigned char eth_src[NET_MAC_LEN],
                         const unsigned char eth_dst[NET_MAC_LEN],
                         unsigned int ip_src, unsigned int ip_dns,
                         unsigned short src_port,
                         const char* name,
                         unsigned short txid) {
    // Encode QNAME first to know payload size.
    unsigned char qname[DNS_MAX_NAME + 2];
    unsigned long qname_len = dns_encode_name_(name, (unsigned char*)qname);
    // DNS message: 12-byte header + QNAME + QTYPE(2) + QCLASS(2)
    unsigned short dns_payload = (unsigned short)((unsigned long)12 + qname_len
                                                  + (unsigned long)4);
    unsigned long data_off = udp_frame(pkt, eth_src, eth_dst,
                                       ip_src, ip_dns,
                                       src_port, (unsigned short)DNS_PORT,
                                       dns_payload);
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + data_off;
        // Transaction ID
        unsigned short txid_n = net_htons(txid);
        d[0] = (unsigned char)((txid_n >> 8) & (unsigned short)0xFF);
        d[1] = (unsigned char)(txid_n & (unsigned short)0xFF);
        // Flags: standard query, recursion desired
        d[2] = (unsigned char)0x01; d[3] = (unsigned char)0x00;
        // QDCOUNT=1
        d[4] = (unsigned char)0; d[5] = (unsigned char)1;
        // ANCOUNT, NSCOUNT, ARCOUNT = 0
        d[6] = (unsigned char)0; d[7] = (unsigned char)0;
        d[8] = (unsigned char)0; d[9] = (unsigned char)0;
        d[10]= (unsigned char)0; d[11]= (unsigned char)0;
        unsigned long i = (unsigned long)0;
        while (i < qname_len) { d[12 + i] = qname[i]; i = i + (unsigned long)1; }
        unsigned long q = (unsigned long)12 + qname_len;
        d[q+0] = (unsigned char)0; d[q+1] = (unsigned char)1; // QTYPE A
        d[q+2] = (unsigned char)0; d[q+3] = (unsigned char)1; // QCLASS IN
    }
    pkt.len = data_off + (unsigned long)dns_payload;
    return txid;
}

int dns_parse_reply(&stack PacketBuf pkt, unsigned long off,
                    unsigned short expected_txid,
                    &stack unsigned int ip4_out) {
    if (pkt.len < off + (unsigned long)12) { return 0; }
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + off;
        unsigned short txid = net_ntohs((unsigned short)(((unsigned short)d[0] << 8)
                                                        | (unsigned short)d[1]));
        if (txid != expected_txid) { return 0; }
        // Flags: check QR=1 (reply)
        if ((d[2] & (unsigned char)0x80) == (unsigned char)0) { return 0; }
        unsigned short ancount = (unsigned short)(((unsigned short)d[6] << 8)
                                                 | (unsigned short)d[7]);
        if (ancount == (unsigned short)0) { return 0; }

        // Skip question section.
        unsigned long pos = off + (unsigned long)12;
        // Skip QNAME labels.
        while (pos < pkt.len && d[pos - off] != (unsigned char)0) {
            if ((d[pos - off] & (unsigned char)0xC0) == (unsigned char)0xC0) {
                pos = pos + (unsigned long)2;
                break;
            }
            unsigned long label_len = (unsigned long)d[pos - off];
            pos = pos + (unsigned long)1 + label_len;
        }
        if (pos < pkt.len && d[pos - off] == (unsigned char)0) {
            pos = pos + (unsigned long)1;
        }
        pos = pos + (unsigned long)4;  // skip QTYPE + QCLASS

        // Parse answers.
        unsigned short ans = (unsigned short)0;
        while (ans < ancount && pos + (unsigned long)10 <= pkt.len) {
            // Skip name (possibly compressed pointer).
            if ((d[pos - off] & (unsigned char)0xC0) == (unsigned char)0xC0) {
                pos = pos + (unsigned long)2;
            } else {
                while (pos < pkt.len && d[pos - off] != (unsigned char)0) {
                    if ((d[pos - off] & (unsigned char)0xC0) == (unsigned char)0xC0) {
                        pos = pos + (unsigned long)2;
                        break;
                    }
                    unsigned long l = (unsigned long)d[pos - off];
                    pos = pos + (unsigned long)1 + l;
                }
                if (pos < pkt.len && d[pos - off] == (unsigned char)0) {
                    pos = pos + (unsigned long)1;
                }
            }
            if (pos + (unsigned long)10 > pkt.len) { break; }
            unsigned short rtype = (unsigned short)(((unsigned short)d[pos-off] << 8)
                                                   | (unsigned short)d[pos-off+1]);
            unsigned short rdlen = (unsigned short)(((unsigned short)d[pos-off+8] << 8)
                                                   | (unsigned short)d[pos-off+9]);
            pos = pos + (unsigned long)10;  // skip TYPE CLASS TTL RDLENGTH
            if (rtype == (unsigned short)1 && rdlen == (unsigned short)4
                && pos + (unsigned long)4 <= pkt.len) {
                // A record
                ip4_out = net_ntohl(((unsigned int)d[pos-off+0] << 24)
                                  | ((unsigned int)d[pos-off+1] << 16)
                                  | ((unsigned int)d[pos-off+2] <<  8)
                                  |  (unsigned int)d[pos-off+3]);
                return 1;
            }
            pos = pos + (unsigned long)rdlen;
            ans = ans + (unsigned short)1;
        }
    }
    return 0;
}
