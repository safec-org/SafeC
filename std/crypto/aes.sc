// SafeC Standard Library — AES Block Cipher
#pragma once
#include <std/stddef.h>
#include <std/crypto/aes.h>
#include <std/mem.h>

namespace std {

extern void* memcpy(void* d, const void* s, size_t n);

// ── AES S-box and inverse S-box ───────────────────────────────────────────────

static unsigned char aes_sbox_[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static unsigned char aes_inv_sbox_[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};

// Rcon (round constants for key expansion): Rcon[i] = 2^(i-1) in GF(2^8)
static unsigned char aes_rcon_[11] = {
    0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36
};

// ── GF(2^8) multiply (xtime) ──────────────────────────────────────────────────
static unsigned char xtime_(unsigned char b) {
    return (unsigned char)(((unsigned int)b << 1) ^
                            (((unsigned int)b >> 7) & 1 ? 0x1b : 0x00));
}

static unsigned char gmul_(unsigned char a, unsigned char b) {
    unsigned char p = (unsigned char)0;
    int i = 0;
    while (i < 8) {
        if ((b & (unsigned char)1) != (unsigned char)0) { p = p ^ a; }
        int hi = (a & (unsigned char)0x80) != (unsigned char)0 ? 1 : 0;
        a = (unsigned char)(a << 1);
        if (hi != 0) { a = a ^ (unsigned char)0x1b; }
        b = b >> 1;
        i = i + 1;
    }
    return p;
}

// ── Key expansion helpers ─────────────────────────────────────────────────────

static unsigned int sub_word_(unsigned int w) {
    return ((unsigned int)aes_sbox_[(w >> 24) & 0xFF] << 24)
         | ((unsigned int)aes_sbox_[(w >> 16) & 0xFF] << 16)
         | ((unsigned int)aes_sbox_[(w >>  8) & 0xFF] <<  8)
         | ((unsigned int)aes_sbox_[w & 0xFF]);
}

static unsigned int rot_word_(unsigned int w) {
    return (w << 8) | (w >> 24);
}

static void expand_key_(const unsigned char* key, int nk, int nr,
                          unsigned int* ks) {
    int i = 0;
    while (i < nk) {
        unsafe {
            ks[i] = ((unsigned int)key[4*i]   << 24)
                  | ((unsigned int)key[4*i+1] << 16)
                  | ((unsigned int)key[4*i+2] <<  8)
                  | ((unsigned int)key[4*i+3]);
        }
        i = i + 1;
    }
    while (i < 4 * (nr + 1)) {
        unsafe {
            unsigned int tmp = ks[i - 1];
            if (i % nk == 0) {
                tmp = sub_word_(rot_word_(tmp)) ^ ((unsigned int)aes_rcon_[i / nk] << 24);
            } else if (nk > 6 && i % nk == 4) {
                tmp = sub_word_(tmp);
            }
            ks[i] = ks[i - nk] ^ tmp;
        }
        i = i + 1;
    }
}

// ── AddRoundKey ───────────────────────────────────────────────────────────────

static void add_round_key_(unsigned char* state, const unsigned int* rk) {
    int i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4) {
            unsafe {
                // state is column-major (state[c*4 + r], FIPS-197 input
                // order): column i takes round-key word i, row j its byte j.
                state[i*4+j] = state[i*4+j] ^
                    (unsigned char)((rk[i] >> (unsigned int)(24 - 8*j)) & 0xFF);
            }
            j = j + 1;
        }
        i = i + 1;
    }
}

// ── SubBytes ──────────────────────────────────────────────────────────────────

static void sub_bytes_(unsigned char* state) {
    int i = 0;
    while (i < 16) {
        unsafe { state[i] = aes_sbox_[state[i]]; }
        i = i + 1;
    }
}

static void inv_sub_bytes_(unsigned char* state) {
    int i = 0;
    while (i < 16) {
        unsafe { state[i] = aes_inv_sbox_[state[i]]; }
        i = i + 1;
    }
}

// ── ShiftRows ─────────────────────────────────────────────────────────────────

static void shift_rows_(unsigned char* s) {
    unsafe {
        unsigned char t;
        // Row 1: shift left 1
        t=s[1]; s[1]=s[5]; s[5]=s[9]; s[9]=s[13]; s[13]=t;
        // Row 2: shift left 2
        t=s[2]; s[2]=s[10]; s[10]=t;
        t=s[6]; s[6]=s[14]; s[14]=t;
        // Row 3: shift left 3 (= right 1)
        t=s[15]; s[15]=s[11]; s[11]=s[7]; s[7]=s[3]; s[3]=t;
    }
}

static void inv_shift_rows_(unsigned char* s) {
    unsafe {
        unsigned char t;
        t=s[13]; s[13]=s[9]; s[9]=s[5]; s[5]=s[1]; s[1]=t;
        t=s[2]; s[2]=s[10]; s[10]=t;
        t=s[6]; s[6]=s[14]; s[14]=t;
        t=s[3]; s[3]=s[7]; s[7]=s[11]; s[11]=s[15]; s[15]=t;
    }
}

// ── MixColumns ────────────────────────────────────────────────────────────────

static void mix_columns_(unsigned char* s) {
    int c = 0;
    while (c < 4) {
        unsafe {
            unsigned char a0=s[c*4+0]; unsigned char a1=s[c*4+1]; unsigned char a2=s[c*4+2]; unsigned char a3=s[c*4+3];
            s[c*4+0] = gmul_(2,a0)^gmul_(3,a1)^a2^a3;
            s[c*4+1] = a0^gmul_(2,a1)^gmul_(3,a2)^a3;
            s[c*4+2] = a0^a1^gmul_(2,a2)^gmul_(3,a3);
            s[c*4+3] = gmul_(3,a0)^a1^a2^gmul_(2,a3);
        }
        c = c + 1;
    }
}

static void inv_mix_columns_(unsigned char* s) {
    int c = 0;
    while (c < 4) {
        unsafe {
            unsigned char a0=s[c*4+0]; unsigned char a1=s[c*4+1]; unsigned char a2=s[c*4+2]; unsigned char a3=s[c*4+3];
            s[c*4+0]=gmul_(14,a0)^gmul_(11,a1)^gmul_(13,a2)^gmul_(9,a3);
            s[c*4+1]=gmul_(9,a0) ^gmul_(14,a1)^gmul_(11,a2)^gmul_(13,a3);
            s[c*4+2]=gmul_(13,a0)^gmul_(9,a1) ^gmul_(14,a2)^gmul_(11,a3);
            s[c*4+3]=gmul_(11,a0)^gmul_(13,a1)^gmul_(9,a2) ^gmul_(14,a3);
        }
        c = c + 1;
    }
}

// ── Core encrypt / decrypt ────────────────────────────────────────────────────

void AesCtx::encrypt_block(&unsigned char block) {
    // Copy input into the column-major state (byte order is already column-major)
    unsigned char state[16];
    int i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4) {
            unsafe { state[i*4+j] = block[i*4+j]; }
            j = j + 1;
        }
        i = i + 1;
    }

    add_round_key_(state, self.ks);

    int r = 1;
    while (r < self.rounds) {
        sub_bytes_(state);
        shift_rows_(state);
        mix_columns_(state);
        unsafe { add_round_key_(state, self.ks + r * 4); }
        r = r + 1;
    }
    sub_bytes_(state);
    shift_rows_(state);
    unsafe { add_round_key_(state, self.ks + self.rounds * 4); }

    // Write back (same column-major byte order)
    i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4) {
            unsafe { block[i*4+j] = state[i*4+j]; }
            j = j + 1;
        }
        i = i + 1;
    }
}

void AesCtx::decrypt_block(&unsigned char block) {
    unsigned char state[16];
    int i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4) {
            unsafe { state[i*4+j] = block[i*4+j]; }
            j = j + 1;
        }
        i = i + 1;
    }

    unsafe { add_round_key_(state, self.ks + self.rounds * 4); }

    int r = self.rounds - 1;
    while (r >= 1) {
        inv_shift_rows_(state);
        inv_sub_bytes_(state);
        unsafe { add_round_key_(state, self.ks + r * 4); }
        inv_mix_columns_(state);
        r = r - 1;
    }
    inv_shift_rows_(state);
    inv_sub_bytes_(state);
    add_round_key_(state, self.ks);

    i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4) {
            unsafe { block[i*4+j] = state[i*4+j]; }
            j = j + 1;
        }
        i = i + 1;
    }
}

inline void AesCtx::set_iv(const &unsigned char iv) {
    unsafe { memcpy((void*)self.iv, (const void*)iv, (unsigned long)AES_BLOCK_SIZE); }
}

inline void AesCtx::destroy() { self.drop(); }

inline void AesCtx::drop() {
    unsafe {
        mem_zero_secure_raw_((void*)self.ks, (unsigned long)AES_KS_WORDS * (unsigned long)sizeof(unsigned int));
        mem_zero_secure_raw_((void*)self.iv, (unsigned long)AES_BLOCK_SIZE);
    }
    self.rounds = 0;
}

inline void AesCtx::cbc_encrypt(&unsigned char data, unsigned long len) {
    unsigned long blocks = len / (unsigned long)AES_BLOCK_SIZE;
    unsigned long b = (unsigned long)0;
    while (b < blocks) {
        unsafe {
            unsigned char* blk = (unsigned char*)data + b * (unsigned long)AES_BLOCK_SIZE;
            int k = 0;
            while (k < AES_BLOCK_SIZE) {
                blk[k] = blk[k] ^ self.iv[k];
                k = k + 1;
            }
            self.encrypt_block((&unsigned char)blk);
            memcpy((void*)self.iv, (const void*)blk, (unsigned long)AES_BLOCK_SIZE);
        }
        b = b + (unsigned long)1;
    }
}

inline void AesCtx::cbc_decrypt(&unsigned char data, unsigned long len) {
    unsigned long blocks = len / (unsigned long)AES_BLOCK_SIZE;
    unsigned long b = (unsigned long)0;
    while (b < blocks) {
        unsafe {
            unsigned char* blk = (unsigned char*)data + b * (unsigned long)AES_BLOCK_SIZE;
            unsigned char prev[16];
            memcpy((void*)prev, (const void*)blk, (unsigned long)AES_BLOCK_SIZE);
            self.decrypt_block((&unsigned char)blk);
            int k = 0;
            while (k < AES_BLOCK_SIZE) {
                blk[k] = blk[k] ^ self.iv[k];
                k = k + 1;
            }
            memcpy((void*)self.iv, (const void*)prev, (unsigned long)AES_BLOCK_SIZE);
        }
        b = b + (unsigned long)1;
    }
}


// ── GCM (NIST SP 800-38D) ─────────────────────────────────────────────────────
// GHASH multiplies in GF(2^128) with the bit-reflected convention of the
// spec (bit 0 = the MSB of byte 0). Blocks are held as two big-endian
// 64-bit halves. The multiply is branch-free in the data (masks, not ifs),
// so its timing doesn't depend on H or the message.

static unsigned long long gcm_load64_(const unsigned char* p) {
    unsigned long long v = 0ULL;
    int i = 0;
    unsafe { while (i < 8) { v = (v << 8ULL) | (unsigned long long)p[i]; i = i + 1; } }
    return v;
}

static void gcm_store64_(unsigned char* p, unsigned long long v) {
    int i = 7;
    unsafe { while (i >= 0) { p[i] = (unsigned char)(v & 0xFFULL); v = v >> 8ULL; i = i - 1; } }
}

// x := x * h in GF(2^128).
static void gcm_mul_(unsigned long long* x, unsigned long long hh, unsigned long long hl) {
    unsigned long long zh = 0ULL;
    unsigned long long zl = 0ULL;
    unsigned long long vh = hh;
    unsigned long long vl = hl;
    unsafe {
        int i = 0;
        while (i < 128) {
            unsigned long long word = i < 64 ? x[0] : x[1];
            unsigned long long bit = (word >> (unsigned long long)(63 - (i & 63))) & 1ULL;
            unsigned long long m = 0ULL - bit;
            zh = zh ^ (vh & m);
            zl = zl ^ (vl & m);
            unsigned long long lsb = 0ULL - (vl & 1ULL);
            vl = (vl >> 1ULL) | (vh << 63ULL);
            vh = (vh >> 1ULL) ^ (0xE100000000000000ULL & lsb);
            i = i + 1;
        }
        x[0] = zh;
        x[1] = zl;
    }
}

// Absorb len bytes (zero-padded to whole blocks) into the GHASH state y.
static void gcm_ghash_(unsigned long long* y, unsigned long long hh, unsigned long long hl,
                       const unsigned char* p, unsigned long len) {
    unsigned long off = 0UL;
    unsafe {
        while (off < len) {
            unsigned char blk[16];
            unsigned long n = len - off < 16UL ? len - off : 16UL;
            unsigned long k = 0UL;
            while (k < 16UL) { blk[k] = k < n ? p[off + k] : (unsigned char)0; k = k + 1UL; }
            y[0] = y[0] ^ gcm_load64_((const unsigned char*)&blk[0]);
            y[1] = y[1] ^ gcm_load64_((const unsigned char*)&blk[8]);
            gcm_mul_(y, hh, hl);
            off = off + n;
        }
    }
}

// CTR mode starting at counter block 'ctr' (its last 32 bits incremented
// before each block, as GCM's inc32 does), XORed into data in place.
static void gcm_ctr_(&AesCtx ctx, const unsigned char* j0, unsigned char* data, unsigned long len) {
    unsafe {
        unsigned char ctr[16];
        memcpy((void*)ctr, (const void*)j0, 16UL);
        unsigned long off = 0UL;
        while (off < len) {
            unsigned int c = ((unsigned int)ctr[12] << 24) | ((unsigned int)ctr[13] << 16)
                           | ((unsigned int)ctr[14] << 8) | (unsigned int)ctr[15];
            c = c + 1u;
            ctr[12] = (unsigned char)(c >> 24); ctr[13] = (unsigned char)(c >> 16);
            ctr[14] = (unsigned char)(c >> 8);  ctr[15] = (unsigned char)c;
            unsigned char ks[16];
            memcpy((void*)ks, (const void*)ctr, 16UL);
            ctx.encrypt_block((&unsigned char)&ks[0]);
            unsigned long k = 0UL;
            while (k < 16UL && off + k < len) { data[off + k] = data[off + k] ^ ks[k]; k = k + 1UL; }
            off = off + 16UL;
        }
    }
}

static void gcm_j0_(const unsigned char* iv, unsigned char* j0) {
    unsafe {
        memcpy((void*)j0, (const void*)iv, 12UL);
        j0[12] = (unsigned char)0; j0[13] = (unsigned char)0; j0[14] = (unsigned char)0; j0[15] = (unsigned char)1;
    }
}

// The tag over (aad, ciphertext): GHASH, then XOR with E(K, J0).
static void gcm_tag_(&AesCtx ctx, const unsigned char* iv, const unsigned char* aad, unsigned long aad_len,
                     const unsigned char* ct, unsigned long len, unsigned char* tag) {
    unsafe {
        unsigned char h[16];
        int i = 0;
        while (i < 16) { h[i] = (unsigned char)0; i = i + 1; }
        ctx.encrypt_block((&unsigned char)&h[0]);
        unsigned long long hh = gcm_load64_((const unsigned char*)&h[0]);
        unsigned long long hl = gcm_load64_((const unsigned char*)&h[8]);
        unsigned long long y[2];
        y[0] = 0ULL;
        y[1] = 0ULL;
        if (aad_len > 0UL) { gcm_ghash_((unsigned long long*)y, hh, hl, aad, aad_len); }
        gcm_ghash_((unsigned long long*)y, hh, hl, ct, len);
        y[0] = y[0] ^ ((unsigned long long)aad_len * 8ULL);
        y[1] = y[1] ^ ((unsigned long long)len * 8ULL);
        gcm_mul_((unsigned long long*)y, hh, hl);
        unsigned char j0[16];
        gcm_j0_(iv, (unsigned char*)j0);
        ctx.encrypt_block((&unsigned char)&j0[0]);
        gcm_store64_(tag, y[0]);
        gcm_store64_(tag + 8, y[1]);
        i = 0;
        while (i < 16) { tag[i] = tag[i] ^ j0[i]; i = i + 1; }
        mem_zero_secure_raw_((void*)h, 16UL);
    }
}

void AesCtx::gcm_encrypt(const &unsigned char iv, const ?&unsigned char aad, unsigned long aad_len,
                         &unsigned char data, unsigned long len, &unsigned char tag) {
    unsafe {
        unsigned char j0[16];
        gcm_j0_((const unsigned char*)iv, (unsigned char*)j0);
        gcm_ctr_(self, (const unsigned char*)j0, (unsigned char*)data, len);
        gcm_tag_(self, (const unsigned char*)iv, (const unsigned char*)aad, aad_len,
                 (const unsigned char*)data, len, (unsigned char*)tag);
    }
}

int AesCtx::gcm_decrypt(const &unsigned char iv, const ?&unsigned char aad, unsigned long aad_len,
                        &unsigned char data, unsigned long len, const &unsigned char tag) {
    unsigned char want[16];
    unsigned int diff = 0u;
    unsafe {
        gcm_tag_(self, (const unsigned char*)iv, (const unsigned char*)aad, aad_len,
                 (const unsigned char*)data, len, (unsigned char*)want);
        int i = 0;
        while (i < 16) { diff = diff | (unsigned int)(want[i] ^ tag[i]); i = i + 1; }
    }
    if (diff != 0u) { return 0; }
    unsafe {
        unsigned char j0[16];
        gcm_j0_((const unsigned char*)iv, (unsigned char*)j0);
        gcm_ctr_(self, (const unsigned char*)j0, (unsigned char*)data, len);
    }
    return 1;
}

// ── Constructors ──────────────────────────────────────────────────────────────

inline struct AesCtx AesCtx::new_128(const &unsigned char key) {
    struct AesCtx ctx;
    ctx.rounds = AES128_ROUNDS;
    unsafe { expand_key_((const unsigned char*)key, 4, AES128_ROUNDS, ctx.ks); }
    unsafe { memset_byte_((unsigned char*)ctx.iv, 0, AES_BLOCK_SIZE); }
    return ctx;
}

inline struct AesCtx AesCtx::new_256(const &unsigned char key) {
    struct AesCtx ctx;
    ctx.rounds = AES256_ROUNDS;
    unsafe { expand_key_((const unsigned char*)key, 8, AES256_ROUNDS, ctx.ks); }
    unsafe { memset_byte_((unsigned char*)ctx.iv, 0, AES_BLOCK_SIZE); }
    return ctx;
}

// Simple zero-init helper for IV
static void memset_byte_(unsigned char* p, unsigned char v, int n) {
    int i = 0;
    while (i < n) {
        unsafe { p[i] = v; }
        i = i + 1;
    }
}

} // namespace std
