// std::net — byte order, ethernet, arp, ipv4 (+checksum), ipv6, udp, dns,
// dhcp, and a full TCP 3-way handshake + data transfer between two TcpConns.
// IPv4 addresses are the numeric value a<<24 | b<<16 | c<<8 | d (what
// net_ip4() returns); assertions check the exact bytes put on the wire.
#include <std/test/test.h>
#include <std/mem.h>
#include <std/panic.h>
#include <std/str.h>
#include <std/net/net_core.h>
#include <std/net/ethernet.h>
#include <std/net/arp.h>
#include <std/net/ipv4.h>
#include <std/net/ipv6.h>
#include <std/net/udp.h>
#include <std/net/dns.h>
#include <std/net/dhcp.h>
#include <std/net/tcp.h>
#include <std/dma.h>

static unsigned char MAC_A[6];
static unsigned char MAC_B[6];

static void init_macs() {
    int i = 0;
    while (i < 6) {
        unsafe {
            MAC_A[i] = (unsigned char)(0x10 + i);
            MAC_B[i] = (unsigned char)(0xA0 + i);
        }
        i = i + 1;
    }
}

static int bytes_eq(const &unsigned char a, const &unsigned char b, unsigned long n) {
    unsigned long i = 0UL;
    while (i < n) {
        unsafe { if (a[i] != b[i]) return 0; }
        i = i + 1UL;
    }
    return 1;
}

void t_core_eth() {
    unsafe {
        ASSERT_EQ(net_htons((unsigned short)0x1234), 0x3412);
        ASSERT_EQ(net_ntohl(net_htonl(0xDEADBEEFu)), 0xDEADBEEFu);
        unsigned int ip = net_ip4((unsigned char)192, (unsigned char)168, (unsigned char)1, (unsigned char)10);
        char buf[32];
        net_ip4_str(ip, buf);
        ASSERT_STR_EQ(buf, "192.168.1.10");
        init_macs();
        net_mac_str(MAC_B, buf);
        ASSERT_STR_EQ(buf, "a0:a1:a2:a3:a4:a5");

        struct PacketBuf pkt;
        pkt.reset();
        eth_build(&pkt, MAC_B, MAC_A, (unsigned short)ETH_TYPE_IPV4);
        ASSERT_EQ(pkt.data[12], 0x08);
        ASSERT_EQ(pkt.data[13], 0x00);
        struct EthernetHdr eh;
        ASSERT_EQ(eth_parse(&pkt, &eh), 0);
        ASSERT_EQ(eh.ethertype, ETH_TYPE_IPV4);
        ASSERT_TRUE(bytes_eq(eh.src, MAC_A, 6UL));
    }
}

void t_ipv4_udp() {
    unsafe {
        init_macs();
        unsigned int src = net_ip4((unsigned char)192, (unsigned char)168, (unsigned char)1, (unsigned char)10);
        unsigned int dst = net_ip4((unsigned char)10, (unsigned char)0, (unsigned char)0, (unsigned char)1);
        struct PacketBuf pkt;
        pkt.reset();
        unsigned long off = udp_frame(&pkt, MAC_A, MAC_B, src, dst,
                                      (unsigned short)5000, (unsigned short)53, (unsigned short)4);
        ASSERT_EQ(off, ETH_HDR_LEN + IPV4_HDR_LEN + UDP_HDR_LEN);
        pkt.data[off] = (unsigned char)'p';
        pkt.data[off + 1UL] = (unsigned char)'i';
        pkt.data[off + 2UL] = (unsigned char)'n';
        pkt.data[off + 3UL] = (unsigned char)'g';
        pkt.len = off + 4UL;
        // Wire bytes of the source address: c0 a8 01 0a.
        ASSERT_EQ(pkt.data[ETH_HDR_LEN + 12], 192);
        ASSERT_EQ(pkt.data[ETH_HDR_LEN + 13], 168);
        ASSERT_EQ(pkt.data[ETH_HDR_LEN + 15], 10);
        ASSERT_EQ(pkt.data[ETH_HDR_LEN + 16], 10);
        // A correct IPv4 header checksums to zero.
        ASSERT_EQ(ip_checksum(pkt.data[(unsigned long)ETH_HDR_LEN..(unsigned long)(ETH_HDR_LEN + IPV4_HDR_LEN)]), 0);
        struct Ipv4Hdr ih;
        ASSERT_EQ(ipv4_parse(&pkt, (unsigned long)ETH_HDR_LEN, &ih), 0);
        ASSERT_EQ(ih.src, src);
        ASSERT_EQ(ih.dst, dst);
        ASSERT_EQ(ih.proto, IP_PROTO_UDP);
        ASSERT_EQ(ih.total_len, IPV4_HDR_LEN + UDP_HDR_LEN + 4);
        struct UdpHdr uh;
        ASSERT_EQ(udp_parse(&pkt, (unsigned long)(ETH_HDR_LEN + IPV4_HDR_LEN), &uh), 0);
        ASSERT_EQ(uh.src_port, 5000);
        ASSERT_EQ(uh.dst_port, 53);
        ASSERT_EQ(uh.length, UDP_HDR_LEN + 4);
    }
}

void t_arp() {
    unsafe {
        init_macs();
        unsigned int ipA = net_ip4((unsigned char)10, (unsigned char)0, (unsigned char)0, (unsigned char)2);
        unsigned int ipB = net_ip4((unsigned char)10, (unsigned char)0, (unsigned char)0, (unsigned char)3);
        struct PacketBuf pkt;
        pkt.reset();
        eth_build(&pkt, MAC_B, MAC_A, (unsigned short)ETH_TYPE_ARP);   // ARP is appended after it
        arp_build_packet(&pkt, (unsigned short)ARP_OP_REQUEST, MAC_A, ipA, MAC_B, ipB);
        // Sender protocol address on the wire: 0a 00 00 02.
        ASSERT_EQ(pkt.data[ETH_HDR_LEN + 14], 10);
        ASSERT_EQ(pkt.data[ETH_HDR_LEN + 17], 2);
        unsigned short op = (unsigned short)0;
        unsigned char sha[6];
        unsigned char tha[6];
        unsigned int spa = 0u;
        unsigned int tpa = 0u;
        ASSERT_EQ(arp_parse_packet(&pkt, (unsigned long)ETH_HDR_LEN, &op, sha, &spa, tha, &tpa), 0);
        ASSERT_EQ(op, ARP_OP_REQUEST);
        ASSERT_EQ(spa, ipA);
        ASSERT_EQ(tpa, ipB);
        ASSERT_TRUE(bytes_eq(sha, MAC_A, 6UL));

        struct ArpTable t;
        t.clear();
        t.update(ipA, MAC_A);
        unsigned char out[6];
        ASSERT_TRUE(t.lookup(ipA, out));
        ASSERT_TRUE(bytes_eq(out, MAC_A, 6UL));
        ASSERT_FALSE(t.lookup(ipB, out));
        t.evict(ipA);
        ASSERT_FALSE(t.lookup(ipA, out));
        // Fill past capacity: oldest entries are replaced, newest is present.
        unsigned int k = 0u;
        while (k < 20u) { t.update(0x0A000100u + k, MAC_B); k = k + 1u; }
        ASSERT_TRUE(t.lookup(0x0A000100u + 19u, out));
    }
}

void t_ipv6() {
    unsafe {
        init_macs();
        struct Ipv6Addr a;
        struct Ipv6Addr b;
        safe_memset_obj(&a, 0);
        safe_memset_obj(&b, 0);
        ASSERT_TRUE(ipv6_addr_is_unspecified(&a));
        a.bytes[15] = (unsigned char)1;
        ASSERT_TRUE(ipv6_addr_is_loopback(&a));
        b.bytes[0] = (unsigned char)0xfe; b.bytes[1] = (unsigned char)0x80; b.bytes[15] = (unsigned char)0x42;
        ASSERT_TRUE(ipv6_addr_is_link_local(&b));
        ASSERT_FALSE(ipv6_addr_eq(&a, &b));
        char s[48];
        ipv6_addr_str(&b, s);
        ASSERT_TRUE(str_find(s, "fe80") != null);
        struct PacketBuf pkt;
        pkt.reset();
        unsigned long off = ipv6_frame(&pkt, MAC_A, MAC_B, (unsigned char)IP_PROTO_UDP, &b, &a, (unsigned short)8);
        ASSERT_EQ(off, ETH_HDR_LEN + IPV6_HDR_LEN);
        struct Ipv6Hdr h;
        ASSERT_EQ(ipv6_parse(&pkt, (unsigned long)ETH_HDR_LEN, &h), 0);
        ASSERT_EQ(h.payload_len, 8);
        ASSERT_EQ(h.next_hdr, IP_PROTO_UDP);
        ASSERT_TRUE(ipv6_addr_eq(&h.src, &b));
    }
}

void t_dns() {
    unsafe {
        init_macs();
        unsigned int me = net_ip4((unsigned char)10, (unsigned char)0, (unsigned char)0, (unsigned char)2);
        unsigned int dnsip = net_ip4((unsigned char)8, (unsigned char)8, (unsigned char)8, (unsigned char)8);
        struct PacketBuf pkt;
        pkt.reset();
        unsigned short qlen = dns_query(&pkt, MAC_A, MAC_B, me, dnsip, (unsigned short)40000,
                                        "example.com", (unsigned short)0xBEEF);
        ASSERT_TRUE(qlen > (unsigned short)0);
        unsigned long q = (unsigned long)(ETH_HDR_LEN + IPV4_HDR_LEN + UDP_HDR_LEN);
        // Question name in label form: 7 'example' 3 'com' 0.
        ASSERT_EQ(pkt.data[q + 12UL], 7);
        ASSERT_EQ(pkt.data[q + 20UL], 3);
        ASSERT_EQ(pkt.data[q], 0xBE);
        // Turn it into a reply: QR=1, ANCOUNT=1, answer = compressed name
        // pointer to the question + A record 93.184.216.34.
        unsigned long end = q + 12UL + 13UL + 4UL;
        pkt.data[q + 2UL] = (unsigned char)0x81;
        pkt.data[q + 3UL] = (unsigned char)0x80;
        pkt.data[q + 7UL] = (unsigned char)1;
        unsigned char ans[16];
        ans[0] = (unsigned char)0xC0; ans[1] = (unsigned char)12;           // name ptr
        ans[2] = (unsigned char)0; ans[3] = (unsigned char)1;               // TYPE A
        ans[4] = (unsigned char)0; ans[5] = (unsigned char)1;               // CLASS IN
        ans[6] = (unsigned char)0; ans[7] = (unsigned char)0; ans[8] = (unsigned char)0x0E; ans[9] = (unsigned char)0x10;
        ans[10] = (unsigned char)0; ans[11] = (unsigned char)4;
        ans[12] = (unsigned char)93; ans[13] = (unsigned char)184; ans[14] = (unsigned char)216; ans[15] = (unsigned char)34;
        safe_memcpy(pkt.data[end..end + 16UL], ans[0..16]);
        pkt.len = end + 16UL;
        unsigned int got = 0u;
        ASSERT_EQ(dns_parse_reply(&pkt, q, (unsigned short)0xBEEF, &got), 1);
        ASSERT_EQ(got, net_ip4((unsigned char)93, (unsigned char)184, (unsigned char)216, (unsigned char)34));
        ASSERT_EQ(dns_parse_reply(&pkt, q, (unsigned short)0x1111, &got), 0);   // wrong txid
    }
}

void t_dhcp() {
    unsafe {
        init_macs();
        struct DhcpClient c;
        safe_memset_obj(&c, 0);
        c.state = DHCP_STATE_IDLE;
        c.xid = 0x12345678u;
        safe_memcpy(c.mac[0..6], MAC_A[0..6]);
        struct PacketBuf pkt;
        pkt.reset();
        c.discover(&pkt, MAC_B);
        ASSERT_EQ(c.state, DHCP_STATE_SELECTING);
        unsigned long d = (unsigned long)(ETH_HDR_LEN + IPV4_HDR_LEN + UDP_HDR_LEN);
        struct UdpHdr uh;
        udp_parse(&pkt, (unsigned long)(ETH_HDR_LEN + IPV4_HDR_LEN), &uh);
        ASSERT_EQ(uh.dst_port, DHCP_SERVER_PORT);
        ASSERT_EQ(pkt.data[d], 1);                         // BOOTREQUEST
        ASSERT_EQ(pkt.data[d + 236UL], 0x63);              // magic cookie
        ASSERT_EQ(pkt.data[d + 240UL], 53);                // option: message type
        ASSERT_EQ(pkt.data[d + 242UL], DHCP_DISCOVER);

        // Server ACK: yiaddr 192.168.1.50, router .1, mask /24, lease 3600s.
        pkt.data[d] = (unsigned char)2;
        pkt.data[d + 16UL] = (unsigned char)192; pkt.data[d + 17UL] = (unsigned char)168;
        pkt.data[d + 18UL] = (unsigned char)1;   pkt.data[d + 19UL] = (unsigned char)50;
        unsigned long o = d + 240UL;
        unsigned char opts[24];
        opts[0] = (unsigned char)53; opts[1] = (unsigned char)1; opts[2] = (unsigned char)DHCP_ACK;
        opts[3] = (unsigned char)1;  opts[4] = (unsigned char)4;
        opts[5] = (unsigned char)255; opts[6] = (unsigned char)255; opts[7] = (unsigned char)255; opts[8] = (unsigned char)0;
        opts[9] = (unsigned char)3;  opts[10] = (unsigned char)4;
        opts[11] = (unsigned char)192; opts[12] = (unsigned char)168; opts[13] = (unsigned char)1; opts[14] = (unsigned char)1;
        opts[15] = (unsigned char)51; opts[16] = (unsigned char)4;
        opts[17] = (unsigned char)0; opts[18] = (unsigned char)0; opts[19] = (unsigned char)0x0E; opts[20] = (unsigned char)0x10;
        opts[21] = (unsigned char)255;
        safe_memcpy(pkt.data[o..o + 22UL], opts[0..22]);
        pkt.len = o + 22UL;
        ASSERT_EQ(c.parse_reply(&pkt, d), DHCP_ACK);
        ASSERT_TRUE(c.is_bound());
        ASSERT_EQ(c.lease.your_ip, net_ip4((unsigned char)192, (unsigned char)168, (unsigned char)1, (unsigned char)50));
        ASSERT_EQ(c.lease.gateway, net_ip4((unsigned char)192, (unsigned char)168, (unsigned char)1, (unsigned char)1));
        ASSERT_EQ(c.lease.subnet_mask, 0xFFFFFF00u);
        ASSERT_EQ(c.lease.lease_secs, 3600);
    }
}

// Moves one segment from 'from' to 'to'; returns 1 if a segment was produced.
static int hop(&TcpConn from, &TcpConn to, const unsigned char fromMac[6], const unsigned char toMac[6]) {
    struct PacketBuf pkt;
    pkt.reset();
    if (!from.build_segment(&pkt, fromMac, toMac)) return 0;
    unsigned long tcpOff = (unsigned long)(ETH_HDR_LEN + IPV4_HDR_LEN);
    // Every segment must carry a valid TCP checksum.
    struct Ipv4Hdr ih;
    ipv4_parse(&pkt, (unsigned long)ETH_HDR_LEN, &ih);
    unsigned long segLen = (unsigned long)ih.total_len - (unsigned long)IPV4_HDR_LEN;
    unsafe {
        if (tcp_checksum(ih.src, ih.dst, pkt.data[tcpOff..tcpOff + segLen]) != (unsigned short)0) return -1;
    }
    to.recv(&pkt, tcpOff);
    return 1;
}

void t_tcp() {
    unsafe {
        init_macs();
        unsigned int ipA = net_ip4((unsigned char)10, (unsigned char)0, (unsigned char)0, (unsigned char)2);
        unsigned int ipB = net_ip4((unsigned char)10, (unsigned char)0, (unsigned char)0, (unsigned char)3);
        ?&heap TcpConn aq = alloc_zeroed(sizeof(struct TcpConn));
        ?&heap TcpConn bq = alloc_zeroed(sizeof(struct TcpConn));
        if (aq.is_null() || bq.is_null()) { ASSERT_TRUE(0); return; }
        &TcpConn a = std::expect_ref(aq, "aq is null");
        &TcpConn b = std::expect_ref(bq, "bq is null");
        b.listen(ipB, (unsigned short)80, 9000u);
        ASSERT_EQ(b.state, TCP_LISTEN);
        a.connect(ipB, (unsigned short)80, ipA, (unsigned short)40001, 1000u);
        ASSERT_EQ(a.state, TCP_SYN_SENT);
        ASSERT_EQ(hop(a, b, MAC_A, MAC_B), 1);    // SYN
        ASSERT_EQ(b.state, TCP_SYN_RECEIVED);
        ASSERT_EQ(b.remote_port, 40001);
        ASSERT_EQ(b.rcv_nxt, 1001);              // peer ISN 1000 + SYN
        ASSERT_EQ(hop(b, a, MAC_B, MAC_A), 1);    // SYN+ACK
        ASSERT_EQ(a.state, TCP_ESTABLISHED);
        ASSERT_EQ(hop(a, b, MAC_A, MAC_B), 1);    // ACK
        ASSERT_EQ(b.state, TCP_ESTABLISHED);
        ASSERT_EQ(a.rcv_nxt, 9001);              // our ISN 9000 + SYN
        ASSERT_EQ(a.send(str_bytes("hello tcp")), 9);
        ASSERT_EQ(hop(a, b, MAC_A, MAC_B), 1);    // data
        ASSERT_TRUE(b.rx_ready());
        unsigned char got[16];
        safe_memset(got[..], 0);
        ASSERT_EQ(b.read(got[0..16]), 9);
        ASSERT_STR_EQ((const char*)got, "hello tcp");
        a.close();
        ASSERT_EQ(hop(a, b, MAC_A, MAC_B), 1);    // FIN
        ASSERT_EQ(b.state, TCP_CLOSE_WAIT);
        dealloc(aq);
        dealloc(bq);
    }
}

// A NetDriver: anything with transmit(frame).
struct LoopDrv {
    int frames;
    unsigned long bytes;
    int transmit([]const unsigned char frame);
};
int LoopDrv::transmit([]const unsigned char frame) {
    self.frames = self.frames + 1;
    self.bytes = self.bytes + frame.len;
    return 0;
}

struct DmaEngine { int polls; unsigned long moved; };
static int dma_start(?&DmaEngine e, unsigned long src, unsigned long dst, unsigned long len) {
    match (e) { case some(x): x.moved = x.moved + len; case null: return -1; }
    return 0;
}
static int dma_poll(?&DmaEngine e) {
    match (e) { case some(x): { x.polls = x.polls + 1; return x.polls >= 3; } case null: { } }
    return 1;
}

void t_drivers() {
    struct LoopDrv drv;
    drv.frames = 0;
    drv.bytes = 0UL;
    struct NetIf nif = NetIf::new(&drv);
    struct PacketBuf pkt;
    pkt.reset();
    pkt.len = 60UL;
    match (pkt.at(0UL)) { case some(b): *b = (unsigned char)0xAB; case null: ASSERT_TRUE(0); }
    ASSERT_TRUE(pkt.at((unsigned long)NET_MTU).is_null());
    ASSERT_EQ(nif.tx(&pkt), 0);
    ASSERT_EQ(drv.frames, 1);
    ASSERT_EQ(drv.bytes, 60);

    struct DmaEngine eng;
    eng.polls = 0;
    eng.moved = 0UL;
    struct DmaChannel<DmaEngine> ch = DmaChannel::new(1U, dma_start, dma_poll, (?&DmaEngine)&eng);
    ASSERT_EQ(ch.start(0x1000UL, 0x2000UL, 256UL), 0);
    ASSERT_TRUE(ch.is_busy());
    ASSERT_EQ(ch.start(0UL, 0UL, 1UL), -1);   // busy
    ASSERT_EQ(ch.wait(10U), 0);
    ASSERT_FALSE(ch.is_busy());
    ASSERT_EQ(eng.polls, 3);
    ASSERT_EQ(eng.moved, 256);
}

void t_build_parse() {
    unsafe {
        struct PacketBuf pkt;
        pkt.reset();
        unsigned int src = net_ip4((unsigned char)10, (unsigned char)1, (unsigned char)2, (unsigned char)3);
        unsigned int dst = net_ip4((unsigned char)10, (unsigned char)9, (unsigned char)8, (unsigned char)7);
        unsigned long p = ipv4_build(&pkt, 0UL, (unsigned char)17, src, dst, (unsigned short)8);
        ASSERT_EQ(p, IPV4_HDR_LEN);
        pkt.len = p + 8UL;
        struct Ipv4Hdr h4;
        ASSERT_EQ(ipv4_parse(&pkt, 0UL, &h4), 0);
        ASSERT_EQ(h4.proto, 17);
        ASSERT_EQ(h4.total_len, IPV4_HDR_LEN + 8);
        ASSERT_TRUE(h4.src == src && h4.dst == dst);

        struct Ipv6Addr a;
        struct Ipv6Addr b;
        safe_memset_obj(&a, 0);
        safe_memset_obj(&b, 0);
        a.bytes[15] = (unsigned char)1;
        b.bytes[0] = (unsigned char)0x20; b.bytes[1] = (unsigned char)0x01;
        pkt.reset();
        p = ipv6_build(&pkt, 0UL, (unsigned char)6, (unsigned char)64, &a, &b, (unsigned short)20);
        ASSERT_EQ(p, 40);
        pkt.len = p + 20UL;
        struct Ipv6Hdr h6;
        ASSERT_EQ(ipv6_parse(&pkt, 0UL, &h6), 0);
        ASSERT_EQ(h6.next_hdr, 6);
        ASSERT_EQ(h6.hop_limit, 64);
        ASSERT_EQ(h6.payload_len, 20);
        ASSERT_TRUE(ipv6_addr_eq(&h6.src, &a));
        ASSERT_TRUE(ipv6_addr_eq(&h6.dst, &b));

        // A 20-byte TCP header written by hand: ports 1234 -> 80, SYN|ACK.
        pkt.reset();
        unsigned char tcp[20];
        int i = 0;
        while (i < 20) { tcp[i] = (unsigned char)0; i = i + 1; }
        tcp[0] = (unsigned char)0x04; tcp[1] = (unsigned char)0xD2;
        tcp[2] = (unsigned char)0x00; tcp[3] = (unsigned char)0x50;
        tcp[4] = (unsigned char)0x00; tcp[5] = (unsigned char)0x00; tcp[6] = (unsigned char)0x01; tcp[7] = (unsigned char)0x00;
        tcp[12] = (unsigned char)0x50;           // data offset 5 words
        tcp[13] = (unsigned char)0x12;           // SYN + ACK
        tcp[14] = (unsigned char)0x10; tcp[15] = (unsigned char)0x00;
        i = 0;
        while (i < 20) { pkt.data[i] = tcp[i]; i = i + 1; }
        pkt.len = 20UL;
        struct TcpHdr th;
        ASSERT_EQ(tcp_parse(&pkt, 0UL, &th), 0);
        ASSERT_EQ(th.src_port, 1234);
        ASSERT_EQ(th.dst_port, 80);
        ASSERT_EQ(th.seq, 256);
        ASSERT_EQ(th.flags, 0x12);
        ASSERT_EQ(th.window, 4096);
        pkt.len = 10UL;
        ASSERT_EQ(tcp_parse(&pkt, 0UL, &th), -1);   // truncated
        pkt.len = 20UL;
        pkt.data[12] = (unsigned char)0x60;      // claims 24 header bytes
        ASSERT_EQ(tcp_parse(&pkt, 0UL, &th), -1);
    }
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("net_core/ethernet", t_core_eth);
        t.add("ipv4/udp", t_ipv4_udp);
        t.add("arp", t_arp);
        t.add("ipv6", t_ipv6);
        t.add("dns", t_dns);
        t.add("dhcp", t_dhcp);
        t.add("tcp handshake+data+fin", t_tcp);
        t.add("netif/dma drivers", t_drivers);
        t.add("ipv4/ipv6/tcp header build+parse", t_build_parse);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
