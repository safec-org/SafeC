// SafeC Standard Library — TCP Layer (minimal state machine)
// Freestanding-safe. No dynamic allocation — caller supplies fixed buffers.
#pragma once
#include "ipv4.h"

#define TCP_HDR_MIN_LEN   20

// TCP flags
#define TCP_FIN   0x01
#define TCP_SYN   0x02
#define TCP_RST   0x04
#define TCP_PSH   0x08
#define TCP_ACK   0x10
#define TCP_URG   0x20

// Connection states
#define TCP_CLOSED       0
#define TCP_LISTEN       1
#define TCP_SYN_SENT     2
#define TCP_SYN_RECEIVED 3
#define TCP_ESTABLISHED  4
#define TCP_FIN_WAIT1    5
#define TCP_FIN_WAIT2    6
#define TCP_CLOSE_WAIT   7
#define TCP_CLOSING      8
#define TCP_LAST_ACK     9
#define TCP_TIME_WAIT    10

struct TcpHdr {
    unsigned short src_port;
    unsigned short dst_port;
    unsigned int   seq;
    unsigned int   ack_num;
    unsigned char  data_off;  // header length in 32-bit words (upper nibble)
    unsigned char  flags;
    unsigned short window;
    unsigned short checksum;
    unsigned short urgent;
};

#define TCP_RX_BUF  2048
#define TCP_TX_BUF  2048

struct TcpConn {
    int            state;
    unsigned int   local_ip;
    unsigned int   remote_ip;
    unsigned short local_port;
    unsigned short remote_port;
    unsigned int   snd_nxt;         // next sequence number to send
    unsigned int   snd_una;         // oldest unacknowledged sequence number
    unsigned int   rcv_nxt;         // next expected receive sequence number
    unsigned int   rcv_wnd;         // receive window advertised
    unsigned char  rx_buf[TCP_RX_BUF];
    unsigned long  rx_len;
    unsigned char  tx_buf[TCP_TX_BUF];
    unsigned long  tx_len;

    // Feed a received packet (starting at TCP header offset in pkt).
    // Returns 1 if state changed, 0 otherwise.
    int  recv(&stack PacketBuf pkt, unsigned long tcp_offset);

    // Enqueue payload data to send.  Returns bytes accepted (0 if full).
    unsigned long send(const unsigned char* data, unsigned long len);

    // Build next outgoing segment into pkt; returns 1 if segment produced.
    int  build_segment(&stack PacketBuf pkt,
                       const unsigned char eth_src[NET_MAC_LEN],
                       const unsigned char eth_dst[NET_MAC_LEN]);

    // Is there received data ready?
    int  rx_ready() const;

    // Consume `len` bytes from rx_buf into `out`; returns bytes copied.
    unsigned long read(unsigned char* out, unsigned long len);

    // Initiate active open (send SYN).
    void connect(unsigned int remote_ip, unsigned short remote_port,
                 unsigned int local_ip, unsigned short local_port,
                 unsigned int isn);

    // Close connection (send FIN).
    void close();
};

// Parse TCP header from pkt at byte offset `offset`.
int tcp_parse(&stack PacketBuf pkt, unsigned long offset,
              &stack TcpHdr hdr_out);

// Compute TCP checksum (pseudo-header + segment).
unsigned short tcp_checksum(unsigned int src_ip, unsigned int dst_ip,
                             const unsigned char* tcp_seg, unsigned long seg_len);
