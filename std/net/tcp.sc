// SafeC Standard Library — TCP Implementation (minimal state machine)
#pragma once
#include "tcp.h"
#include "ethernet.h"

// ── Checksum helpers ──────────────────────────────────────────────────────────

unsigned short tcp_checksum(unsigned int src_ip, unsigned int dst_ip,
                             const unsigned char* tcp_seg, unsigned long seg_len) {
    unsigned long sum = (unsigned long)0;
    // Pseudo-header: src_ip, dst_ip, 0x00, proto=6, tcp_length
    unsafe {
        sum = sum + ((src_ip >> 16) & (unsigned int)0xFFFF);
        sum = sum + (src_ip & (unsigned int)0xFFFF);
        sum = sum + ((dst_ip >> 16) & (unsigned int)0xFFFF);
        sum = sum + (dst_ip & (unsigned int)0xFFFF);
        sum = sum + (unsigned long)6;   // protocol = TCP
        sum = sum + seg_len;
        unsigned long i = (unsigned long)0;
        while (i + (unsigned long)1 < seg_len) {
            unsigned long word = ((unsigned long)tcp_seg[i] << 8)
                               | (unsigned long)tcp_seg[i + (unsigned long)1];
            sum = sum + word;
            i = i + (unsigned long)2;
        }
        if (i < seg_len) {
            sum = sum + ((unsigned long)tcp_seg[i] << 8);
        }
    }
    while ((sum >> 16) != (unsigned long)0) {
        sum = (sum & (unsigned long)0xFFFF) + (sum >> 16);
    }
    return (unsigned short)(~sum & (unsigned long)0xFFFF);
}

int tcp_parse(&stack PacketBuf pkt, unsigned long offset,
              &stack TcpHdr hdr_out) {
    if (pkt.len < offset + (unsigned long)TCP_HDR_MIN_LEN) { return -1; }
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + offset;
        hdr_out.src_port = net_ntohs((unsigned short)(((unsigned short)d[0] << 8)
                                                     | (unsigned short)d[1]));
        hdr_out.dst_port = net_ntohs((unsigned short)(((unsigned short)d[2] << 8)
                                                     | (unsigned short)d[3]));
        hdr_out.seq      = net_ntohl(((unsigned int)d[4] << 24) | ((unsigned int)d[5] << 16)
                                   | ((unsigned int)d[6] << 8)  |  (unsigned int)d[7]);
        hdr_out.ack_num  = net_ntohl(((unsigned int)d[8] << 24) | ((unsigned int)d[9] << 16)
                                   | ((unsigned int)d[10] << 8) |  (unsigned int)d[11]);
        hdr_out.data_off = (unsigned char)((d[12] >> 4) & (unsigned char)0xF);
        hdr_out.flags    = d[13];
        hdr_out.window   = net_ntohs((unsigned short)(((unsigned short)d[14] << 8)
                                                     | (unsigned short)d[15]));
        hdr_out.checksum = (unsigned short)0;
        hdr_out.urgent   = (unsigned short)0;
    }
    return 0;
}

// ── Helper: write TCP header into pkt at offset ───────────────────────────────

static unsigned long tcp_write_hdr_(unsigned char* d,
                                    unsigned short src_port, unsigned short dst_port,
                                    unsigned int seq, unsigned int ack_num,
                                    unsigned char flags, unsigned short window) {
    unsafe {
        unsigned short sp = net_htons(src_port);
        unsigned short dp = net_htons(dst_port);
        d[0] = (unsigned char)((sp >> 8) & (unsigned short)0xFF);
        d[1] = (unsigned char)(sp & (unsigned short)0xFF);
        d[2] = (unsigned char)((dp >> 8) & (unsigned short)0xFF);
        d[3] = (unsigned char)(dp & (unsigned short)0xFF);
        unsigned int seq_n = net_htonl(seq);
        d[4] = (unsigned char)((seq_n >> 24) & (unsigned int)0xFF);
        d[5] = (unsigned char)((seq_n >> 16) & (unsigned int)0xFF);
        d[6] = (unsigned char)((seq_n >>  8) & (unsigned int)0xFF);
        d[7] = (unsigned char)(seq_n & (unsigned int)0xFF);
        unsigned int ack_n = net_htonl(ack_num);
        d[8]  = (unsigned char)((ack_n >> 24) & (unsigned int)0xFF);
        d[9]  = (unsigned char)((ack_n >> 16) & (unsigned int)0xFF);
        d[10] = (unsigned char)((ack_n >>  8) & (unsigned int)0xFF);
        d[11] = (unsigned char)(ack_n & (unsigned int)0xFF);
        d[12] = (unsigned char)0x50;  // data offset = 5 (20 bytes)
        d[13] = flags;
        unsigned short win = net_htons(window);
        d[14] = (unsigned char)((win >> 8) & (unsigned short)0xFF);
        d[15] = (unsigned char)(win & (unsigned short)0xFF);
        d[16] = (unsigned char)0; d[17] = (unsigned char)0;  // checksum placeholder
        d[18] = (unsigned char)0; d[19] = (unsigned char)0;  // urgent
    }
    return (unsigned long)TCP_HDR_MIN_LEN;
}

// ── TcpConn methods ───────────────────────────────────────────────────────────

void TcpConn::connect(unsigned int remote_ip, unsigned short remote_port,
                      unsigned int local_ip, unsigned short local_port,
                      unsigned int isn) {
    self.state       = TCP_SYN_SENT;
    self.remote_ip   = remote_ip;
    self.remote_port = remote_port;
    self.local_ip    = local_ip;
    self.local_port  = local_port;
    self.snd_nxt     = isn + (unsigned int)1;
    self.snd_una     = isn;
    self.rcv_nxt     = (unsigned int)0;
    self.rcv_wnd     = (unsigned int)TCP_RX_BUF;
    self.rx_len      = (unsigned long)0;
    self.tx_len      = (unsigned long)0;
}

void TcpConn::close() {
    if (self.state == TCP_ESTABLISHED) {
        self.state = TCP_FIN_WAIT1;
    } else if (self.state == TCP_CLOSE_WAIT) {
        self.state = TCP_LAST_ACK;
    }
}

int TcpConn::recv(&stack PacketBuf pkt, unsigned long tcp_offset) {
    struct TcpHdr hdr;
    if (tcp_parse(pkt, tcp_offset, hdr) != 0) { return 0; }
    if (hdr.dst_port != self.local_port) { return 0; }

    unsigned char flags = hdr.flags;
    int prev = self.state;

    if (self.state == TCP_SYN_SENT) {
        // Expect SYN+ACK
        if ((flags & (unsigned char)(TCP_SYN | TCP_ACK)) == (unsigned char)(TCP_SYN | TCP_ACK)) {
            self.rcv_nxt = hdr.seq + (unsigned int)1;
            self.snd_una = hdr.ack_num;
            self.state   = TCP_ESTABLISHED;
        }
    } else if (self.state == TCP_ESTABLISHED) {
        if ((flags & (unsigned char)TCP_ACK) != (unsigned char)0) {
            self.snd_una = hdr.ack_num;
        }
        // Receive payload.
        unsigned long tcp_hdr_bytes = (unsigned long)hdr.data_off * (unsigned long)4;
        unsigned long data_start = tcp_offset + tcp_hdr_bytes;
        if (pkt.len > data_start) {
            unsigned long data_len = pkt.len - data_start;
            unsigned long space = (unsigned long)TCP_RX_BUF - self.rx_len;
            if (data_len > space) { data_len = space; }
            unsafe {
                unsigned char* src = (unsigned char*)pkt.data + data_start;
                unsigned long i = (unsigned long)0;
                while (i < data_len) {
                    self.rx_buf[self.rx_len + i] = src[i];
                    i = i + (unsigned long)1;
                }
            }
            self.rx_len = self.rx_len + data_len;
            self.rcv_nxt = self.rcv_nxt + (unsigned int)data_len;
        }
        if ((flags & (unsigned char)TCP_FIN) != (unsigned char)0) {
            self.rcv_nxt = self.rcv_nxt + (unsigned int)1;
            self.state   = TCP_CLOSE_WAIT;
        }
    } else if (self.state == TCP_FIN_WAIT1) {
        if ((flags & (unsigned char)TCP_ACK) != (unsigned char)0) {
            self.state = TCP_FIN_WAIT2;
        }
    } else if (self.state == TCP_FIN_WAIT2) {
        if ((flags & (unsigned char)TCP_FIN) != (unsigned char)0) {
            self.rcv_nxt = self.rcv_nxt + (unsigned int)1;
            self.state   = TCP_TIME_WAIT;
        }
    } else if (self.state == TCP_LAST_ACK) {
        if ((flags & (unsigned char)TCP_ACK) != (unsigned char)0) {
            self.state = TCP_CLOSED;
        }
    }

    if (self.state != prev) { return 1; }
    return 0;
}

unsigned long TcpConn::send(const unsigned char* data, unsigned long len) {
    unsigned long space = (unsigned long)TCP_TX_BUF - self.tx_len;
    if (len > space) { len = space; }
    unsafe {
        unsigned long i = (unsigned long)0;
        while (i < len) {
            self.tx_buf[self.tx_len + i] = data[i];
            i = i + (unsigned long)1;
        }
    }
    self.tx_len = self.tx_len + len;
    return len;
}

int TcpConn::build_segment(&stack PacketBuf pkt,
                            const unsigned char eth_src[NET_MAC_LEN],
                            const unsigned char eth_dst[NET_MAC_LEN]) {
    if (self.state == TCP_CLOSED) { return 0; }

    unsigned char flags = (unsigned char)0;
    unsigned int seq = self.snd_nxt;

    if (self.state == TCP_SYN_SENT) {
        flags = (unsigned char)TCP_SYN;
    } else if (self.state == TCP_ESTABLISHED) {
        flags = (unsigned char)TCP_ACK;
        if (self.tx_len > (unsigned long)0) {
            flags = (unsigned char)(flags | (unsigned char)TCP_PSH);
        }
    } else if (self.state == TCP_FIN_WAIT1 || self.state == TCP_LAST_ACK) {
        flags = (unsigned char)(TCP_FIN | TCP_ACK);
    } else if (self.state == TCP_CLOSE_WAIT) {
        flags = (unsigned char)TCP_ACK;
    } else {
        return 0;
    }

    unsigned long payload_len = self.tx_len;
    unsigned long total_tcp = (unsigned long)TCP_HDR_MIN_LEN + payload_len;
    unsigned short ip_payload = (unsigned short)((unsigned short)IPV4_HDR_LEN
                                                + (unsigned short)total_tcp);

    pkt.reset();
    eth_build(pkt, eth_dst, eth_src, (unsigned short)ETH_TYPE_IPV4);
    unsigned long ip_off  = (unsigned long)ETH_HDR_LEN;
    ipv4_build(pkt, ip_off, (unsigned char)IP_PROTO_TCP,
               self.local_ip, self.remote_ip,
               (unsigned short)((unsigned short)TCP_HDR_MIN_LEN
                                + (unsigned short)payload_len));

    unsigned long tcp_off = ip_off + (unsigned long)IPV4_HDR_LEN;
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + tcp_off;
        tcp_write_hdr_(d, self.local_port, self.remote_port,
                       seq, self.rcv_nxt, flags,
                       (unsigned short)self.rcv_wnd);
        // Copy tx_buf payload.
        unsigned long i = (unsigned long)0;
        while (i < payload_len) {
            d[TCP_HDR_MIN_LEN + i] = self.tx_buf[i];
            i = i + (unsigned long)1;
        }
        // Fill checksum.
        unsigned short csum = tcp_checksum(self.local_ip, self.remote_ip,
                                           (const unsigned char*)d,
                                           total_tcp);
        unsigned short csum_n = net_htons(csum);
        d[16] = (unsigned char)((csum_n >> 8) & (unsigned short)0xFF);
        d[17] = (unsigned char)(csum_n & (unsigned short)0xFF);
    }
    pkt.len = tcp_off + total_tcp;

    // Advance sequence numbers.
    if ((flags & (unsigned char)TCP_SYN) != (unsigned char)0) {
        self.snd_nxt = self.snd_nxt + (unsigned int)1;
    }
    if (payload_len > (unsigned long)0) {
        self.snd_nxt = self.snd_nxt + (unsigned int)payload_len;
        self.tx_len  = (unsigned long)0;
    }
    if ((flags & (unsigned char)TCP_FIN) != (unsigned char)0) {
        self.snd_nxt = self.snd_nxt + (unsigned int)1;
    }

    return 1;
}

int TcpConn::rx_ready() const {
    if (self.rx_len > (unsigned long)0) { return 1; }
    return 0;
}

unsigned long TcpConn::read(unsigned char* out, unsigned long len) {
    if (len > self.rx_len) { len = self.rx_len; }
    unsafe {
        unsigned long i = (unsigned long)0;
        while (i < len) {
            out[i] = self.rx_buf[i];
            i = i + (unsigned long)1;
        }
        // Shift remaining bytes down.
        unsigned long remain = self.rx_len - len;
        i = (unsigned long)0;
        while (i < remain) {
            self.rx_buf[i] = self.rx_buf[len + i];
            i = i + (unsigned long)1;
        }
    }
    self.rx_len = self.rx_len - len;
    return len;
}
