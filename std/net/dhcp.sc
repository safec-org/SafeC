// SafeC Standard Library — DHCP Client Implementation
#pragma once
#include "dhcp.h"
#include "ethernet.h"

// DHCP BOOTP message: 236 bytes fixed + 4-byte magic cookie + options
// op(1) htype(1) hlen(1) hops(1) xid(4) secs(2) flags(2)
// ciaddr(4) yiaddr(4) siaddr(4) giaddr(4)
// chaddr(16) sname(64) file(128)
// magic(4) options(...)

#define DHCP_FIXED_LEN   236
#define DHCP_HDR_LEN     (DHCP_FIXED_LEN + 4)  // + magic cookie

static unsigned char dhcp_bcast_mac_[NET_MAC_LEN] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

// Write `len` bytes of `val` into buf.
static void dhcp_fill_(unsigned char* buf, unsigned char val, unsigned long len) {
    unsigned long i = (unsigned long)0;
    while (i < len) { unsafe { buf[i] = val; } i = i + (unsigned long)1; }
}

// Append a DHCP option: code, len, data.
static unsigned long dhcp_opt_(unsigned char* opts, unsigned long pos,
                                unsigned char code, unsigned char len,
                                const unsigned char* data) {
    unsafe {
        opts[pos] = code;
        opts[pos + (unsigned long)1] = len;
        unsigned long i = (unsigned long)0;
        while (i < (unsigned long)len) {
            opts[pos + (unsigned long)2 + i] = data[i];
            i = i + (unsigned long)1;
        }
    }
    return pos + (unsigned long)2 + (unsigned long)len;
}

static void dhcp_write_fixed_(unsigned char* d, unsigned char op,
                               const unsigned char mac[NET_MAC_LEN],
                               unsigned int xid) {
    unsafe {
        d[0] = op;            // op: 1=BOOTREQUEST
        d[1] = (unsigned char)1;    // htype: Ethernet
        d[2] = (unsigned char)6;    // hlen
        d[3] = (unsigned char)0;    // hops
        unsigned int xid_n = net_htonl(xid);
        d[4] = (unsigned char)((xid_n >> 24) & (unsigned int)0xFF);
        d[5] = (unsigned char)((xid_n >> 16) & (unsigned int)0xFF);
        d[6] = (unsigned char)((xid_n >>  8) & (unsigned int)0xFF);
        d[7] = (unsigned char)(xid_n & (unsigned int)0xFF);
        d[8] = (unsigned char)0; d[9] = (unsigned char)0;    // secs
        d[10]= (unsigned char)0x80; d[11]= (unsigned char)0; // broadcast flag
        // ciaddr/yiaddr/siaddr/giaddr = 0
        dhcp_fill_(d + 12, (unsigned char)0, (unsigned long)16);
        // chaddr (MAC + pad)
        int i = 0;
        while (i < NET_MAC_LEN) { d[28 + i] = mac[i]; i = i + 1; }
        dhcp_fill_(d + 34, (unsigned char)0, (unsigned long)10); // chaddr pad
        dhcp_fill_(d + 44, (unsigned char)0, (unsigned long)192); // sname + file
        // Magic cookie
        d[236]= (unsigned char)0x63; d[237]= (unsigned char)0x82;
        d[238]= (unsigned char)0x53; d[239]= (unsigned char)0x63;
    }
}

void DhcpClient::discover(&stack PacketBuf pkt,
                           const unsigned char eth_dst[NET_MAC_LEN]) {
    unsigned short opts_len = (unsigned short)7; // type(3)+end(1) + pad
    unsigned short dhcp_payload = (unsigned short)((unsigned short)DHCP_HDR_LEN
                                                    + opts_len);
    unsigned long data_off = udp_frame(pkt,
                                       (const unsigned char*)self.mac,
                                       eth_dst,
                                       (unsigned int)0,           // src IP 0.0.0.0
                                       (unsigned int)0xFFFFFFFF,  // dst 255.255.255.255
                                       (unsigned short)DHCP_CLIENT_PORT,
                                       (unsigned short)DHCP_SERVER_PORT,
                                       dhcp_payload);
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + data_off;
        dhcp_write_fixed_(d, (unsigned char)1, (const unsigned char*)self.mac, self.xid);
        unsigned char* opts = d + (unsigned long)DHCP_HDR_LEN;
        unsigned long pos = (unsigned long)0;
        unsigned char msg_type = (unsigned char)DHCP_DISCOVER;
        pos = dhcp_opt_(opts, pos, (unsigned char)53, (unsigned char)1,
                        (const unsigned char*)&msg_type);
        opts[pos] = (unsigned char)255; // END
    }
    pkt.len = data_off + (unsigned long)dhcp_payload;
    self.state = DHCP_STATE_SELECTING;
}

void DhcpClient::request(&stack PacketBuf pkt,
                          const unsigned char eth_dst[NET_MAC_LEN],
                          unsigned int offered_ip, unsigned int server_ip) {
    unsigned short opts_len = (unsigned short)13; // type(3)+reqip(6)+serverid(6)+end(1)
    unsigned short dhcp_payload = (unsigned short)((unsigned short)DHCP_HDR_LEN
                                                    + opts_len);
    unsigned long data_off = udp_frame(pkt,
                                       (const unsigned char*)self.mac,
                                       eth_dst,
                                       (unsigned int)0,
                                       (unsigned int)0xFFFFFFFF,
                                       (unsigned short)DHCP_CLIENT_PORT,
                                       (unsigned short)DHCP_SERVER_PORT,
                                       dhcp_payload);
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + data_off;
        dhcp_write_fixed_(d, (unsigned char)1, (const unsigned char*)self.mac, self.xid);
        unsigned char* opts = d + (unsigned long)DHCP_HDR_LEN;
        unsigned long pos = (unsigned long)0;
        unsigned char msg_type = (unsigned char)DHCP_REQUEST;
        pos = dhcp_opt_(opts, pos, (unsigned char)53, (unsigned char)1,
                        (const unsigned char*)&msg_type);
        unsigned int ip_n = net_htonl(offered_ip);
        pos = dhcp_opt_(opts, pos, (unsigned char)50, (unsigned char)4,
                        (const unsigned char*)&ip_n);
        unsigned int srv_n = net_htonl(server_ip);
        pos = dhcp_opt_(opts, pos, (unsigned char)54, (unsigned char)4,
                        (const unsigned char*)&srv_n);
        opts[pos] = (unsigned char)255;
    }
    pkt.len = data_off + (unsigned long)dhcp_payload;
    self.state = DHCP_STATE_REQUESTING;
}

int DhcpClient::parse_reply(&stack PacketBuf pkt, unsigned long udp_payload_offset) {
    if (pkt.len < udp_payload_offset + (unsigned long)DHCP_HDR_LEN) { return 0; }
    int msg_type = 0;
    unsafe {
        unsigned char* d = (unsigned char*)pkt.data + udp_payload_offset;
        if (d[0] != (unsigned char)2) { return 0; }  // op must be BOOTREPLY
        // Check xid
        unsigned int xid = net_ntohl(((unsigned int)d[4] << 24) | ((unsigned int)d[5] << 16)
                                    | ((unsigned int)d[6] << 8)  |  (unsigned int)d[7]);
        if (xid != self.xid) { return 0; }
        // yiaddr at offset 16
        self.lease.your_ip = net_ntohl(((unsigned int)d[16] << 24)
                                      | ((unsigned int)d[17] << 16)
                                      | ((unsigned int)d[18] << 8)
                                      |  (unsigned int)d[19]);
        // siaddr at offset 20
        self.lease.server_ip = net_ntohl(((unsigned int)d[20] << 24)
                                        | ((unsigned int)d[21] << 16)
                                        | ((unsigned int)d[22] << 8)
                                        |  (unsigned int)d[23]);
        // Parse options
        unsigned long pos = udp_payload_offset + (unsigned long)DHCP_HDR_LEN;
        while (pos + (unsigned long)1 < pkt.len) {
            unsigned char code = ((unsigned char*)pkt.data)[pos];
            if (code == (unsigned char)255) { break; }
            if (code == (unsigned char)0)   { pos = pos + (unsigned long)1; }
            else {
                unsigned char opt_len = ((unsigned char*)pkt.data)[pos + (unsigned long)1];
                unsigned char* v = (unsigned char*)pkt.data + pos + (unsigned long)2;
                if (code == (unsigned char)53 && opt_len == (unsigned char)1) {
                    msg_type = (int)v[0];
                } else if (code == (unsigned char)1 && opt_len == (unsigned char)4) {
                    self.lease.subnet_mask = net_ntohl(((unsigned int)v[0] << 24)
                                                      | ((unsigned int)v[1] << 16)
                                                      | ((unsigned int)v[2] << 8)
                                                      |  (unsigned int)v[3]);
                } else if (code == (unsigned char)3 && opt_len >= (unsigned char)4) {
                    self.lease.gateway = net_ntohl(((unsigned int)v[0] << 24)
                                                  | ((unsigned int)v[1] << 16)
                                                  | ((unsigned int)v[2] << 8)
                                                  |  (unsigned int)v[3]);
                } else if (code == (unsigned char)6 && opt_len >= (unsigned char)4) {
                    self.lease.dns = net_ntohl(((unsigned int)v[0] << 24)
                                              | ((unsigned int)v[1] << 16)
                                              | ((unsigned int)v[2] << 8)
                                              |  (unsigned int)v[3]);
                } else if (code == (unsigned char)51 && opt_len == (unsigned char)4) {
                    self.lease.lease_secs = net_ntohl(((unsigned int)v[0] << 24)
                                                     | ((unsigned int)v[1] << 16)
                                                     | ((unsigned int)v[2] << 8)
                                                     |  (unsigned int)v[3]);
                }
                pos = pos + (unsigned long)2 + (unsigned long)opt_len;
            }
        }
    }
    if (msg_type == DHCP_ACK) { self.state = DHCP_STATE_BOUND; }
    return msg_type;
}

int DhcpClient::is_bound() const {
    if (self.state == DHCP_STATE_BOUND) { return 1; }
    return 0;
}
