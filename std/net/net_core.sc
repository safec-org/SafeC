// SafeC Standard Library — Network Core Implementation
#pragma once
#include <std/net/net_core.h>

// ── Byte-order conversions ────────────────────────────────────────────────────

namespace std {

inline unsigned short net_htons(unsigned short v) {
    return (unsigned short)(((v & (unsigned short)0xFF) << 8) |
                            ((v >> 8) & (unsigned short)0xFF));
}

inline unsigned short net_ntohs(unsigned short v) {
    return net_htons(v);
}

inline unsigned int net_htonl(unsigned int v) {
    return ((v & (unsigned int)0xFF)         << 24)
         | (((v >> 8)  & (unsigned int)0xFF) << 16)
         | (((v >> 16) & (unsigned int)0xFF) <<  8)
         |  ((v >> 24) & (unsigned int)0xFF);
}

inline unsigned int net_ntohl(unsigned int v) {
    return net_htonl(v);
}

// ── IP / MAC utilities ────────────────────────────────────────────────────────

// Write a decimal number (0-255) into buf; return chars written.
static int net_write_dec_(char* buf, unsigned int val) {
    if (val >= 100) {
        unsafe {
            buf[0] = (char)('0' + val / (unsigned int)100);
            buf[1] = (char)('0' + (val / (unsigned int)10) % (unsigned int)10);
            buf[2] = (char)('0' + val % (unsigned int)10);
        }
        return 3;
    }
    if (val >= 10) {
        unsafe {
            buf[0] = (char)('0' + val / (unsigned int)10);
            buf[1] = (char)('0' + val % (unsigned int)10);
        }
        return 2;
    }
    unsafe { buf[0] = (char)('0' + val); }
    return 1;
}

// Write a hex nibble.
static char net_hex_nibble_(unsigned int v) {
    v = v & (unsigned int)0xF;
    if (v < (unsigned int)10) {
        return (char)('0' + v);
    }
    return (char)('a' + v - (unsigned int)10);
}

inline void net_ip4_str(unsigned int ip_be, char* buf) {
    int pos = 0;
    int i = 3;
    while (i >= 0) {
        unsigned int octet;
        unsafe { octet = (ip_be >> ((unsigned int)i * (unsigned int)8)) & (unsigned int)0xFF; }
        int n;
        unsafe { n = net_write_dec_(buf + pos, octet); }
        pos = pos + n;
        if (i > 0) {
            unsafe { buf[pos] = '.'; }
            pos = pos + 1;
        }
        i = i - 1;
    }
    unsafe { buf[pos] = (char)0; }
}

inline void net_mac_str(const unsigned char mac[NET_MAC_LEN], char* buf) {
    int i = 0;
    while (i < NET_MAC_LEN) {
        unsigned int b;
        unsafe { b = (unsigned int)mac[i]; }
        unsafe {
            buf[i*3+0] = net_hex_nibble_(b >> 4);
            buf[i*3+1] = net_hex_nibble_(b);
        }
        if (i < NET_MAC_LEN - 1) {
            unsafe { buf[i*3+2] = ':'; }
        }
        i = i + 1;
    }
    unsafe { buf[NET_MAC_LEN*3 - 1] = (char)0; }
}

unsigned int net_ip4(unsigned char a, unsigned char b,
                     unsigned char c, unsigned char d) {
    return ((unsigned int)a << 24)
         | ((unsigned int)b << 16)
         | ((unsigned int)c <<  8)
         |  (unsigned int)d;
}

// ── PacketBuf ─────────────────────────────────────────────────────────────────

inline void* PacketBuf::at(unsigned long offset) {
    unsafe { return (void*)((unsigned long)(unsigned char*)self.data + offset); }
    return (void*)0;
}

inline void PacketBuf::reset() {
    unsigned long i = (unsigned long)0;
    while (i < (unsigned long)NET_MTU) {
        unsafe { self.data[i] = (unsigned char)0; }
        i = i + (unsigned long)1;
    }
    self.len = (unsigned long)0;
}

// ── NetIf ─────────────────────────────────────────────────────────────────────

inline int NetIf::tx(&stack PacketBuf pkt) {
    if (self.tx_fn == (void*)0) { return -1; }
    unsafe {
        fn int(void*, unsigned char*, unsigned long) func =
            (fn int(void*, unsigned char*, unsigned long))self.tx_fn;
        return func(self.iface_ctx, (unsigned char*)pkt.data, pkt.len);
    }
    return -1;
}

} // namespace std
