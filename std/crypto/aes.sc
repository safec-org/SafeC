// SafeC Standard Library — AES Block Cipher
#pragma once
#include "aes.h"

extern void* memcpy(void* d, const void* s, unsigned long n);

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
         | ((unsigned int)aes_sbox_[(w      ) & 0xFF]      );
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
        unsigned int tmp = ks[i - 1];
        if (i % nk == 0) {
            tmp = sub_word_(rot_word_(tmp)) ^ ((unsigned int)aes_rcon_[i / nk] << 24);
        } else if (nk > 6 && i % nk == 4) {
            tmp = sub_word_(tmp);
        }
        ks[i] = ks[i - nk] ^ tmp;
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
                state[i*4+j] = state[i*4+j] ^
                    (unsigned char)((rk[j] >> (unsigned int)(24 - 8*i)) & 0xFF);
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
        state[i] = aes_sbox_[state[i]];
        i = i + 1;
    }
}

static void inv_sub_bytes_(unsigned char* state) {
    int i = 0;
    while (i < 16) {
        state[i] = aes_inv_sbox_[state[i]];
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
            unsigned char a0=s[c*4+0], a1=s[c*4+1], a2=s[c*4+2], a3=s[c*4+3];
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
            unsigned char a0=s[c*4+0],a1=s[c*4+1],a2=s[c*4+2],a3=s[c*4+3];
            s[c*4+0]=gmul_(14,a0)^gmul_(11,a1)^gmul_(13,a2)^gmul_(9,a3);
            s[c*4+1]=gmul_(9,a0) ^gmul_(14,a1)^gmul_(11,a2)^gmul_(13,a3);
            s[c*4+2]=gmul_(13,a0)^gmul_(9,a1) ^gmul_(14,a2)^gmul_(11,a3);
            s[c*4+3]=gmul_(11,a0)^gmul_(13,a1)^gmul_(9,a2) ^gmul_(14,a3);
        }
        c = c + 1;
    }
}

// ── Core encrypt / decrypt ────────────────────────────────────────────────────

void AesCtx::encrypt_block(&stack unsigned char block) {
    // Copy input into column-major state
    unsigned char state[16];
    int i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4) {
            unsafe { state[i*4+j] = block[j*4+i]; }
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

    // Write back row-major
    i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4) {
            unsafe { block[j*4+i] = state[i*4+j]; }
            j = j + 1;
        }
        i = i + 1;
    }
}

void AesCtx::decrypt_block(&stack unsigned char block) {
    unsigned char state[16];
    int i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4) {
            unsafe { state[i*4+j] = block[j*4+i]; }
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
            unsafe { block[j*4+i] = state[i*4+j]; }
            j = j + 1;
        }
        i = i + 1;
    }
}

void AesCtx::set_iv(const &stack unsigned char iv) {
    unsafe { memcpy((void*)self.iv, (const void*)iv, (unsigned long)AES_BLOCK_SIZE); }
}

void AesCtx::cbc_encrypt(&stack unsigned char data, unsigned long len) {
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
            self.encrypt_block(blk);
            memcpy((void*)self.iv, (const void*)blk, (unsigned long)AES_BLOCK_SIZE);
        }
        b = b + (unsigned long)1;
    }
}

void AesCtx::cbc_decrypt(&stack unsigned char data, unsigned long len) {
    unsigned long blocks = len / (unsigned long)AES_BLOCK_SIZE;
    unsigned long b = (unsigned long)0;
    while (b < blocks) {
        unsafe {
            unsigned char* blk = (unsigned char*)data + b * (unsigned long)AES_BLOCK_SIZE;
            unsigned char prev[16];
            memcpy((void*)prev, (const void*)blk, (unsigned long)AES_BLOCK_SIZE);
            self.decrypt_block(blk);
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

// ── Constructors ──────────────────────────────────────────────────────────────

struct AesCtx aes128_init(const &stack unsigned char key) {
    struct AesCtx ctx;
    ctx.rounds = AES128_ROUNDS;
    unsafe { expand_key_((const unsigned char*)key, 4, AES128_ROUNDS, ctx.ks); }
    unsafe { memset_byte_((unsigned char*)ctx.iv, 0, AES_BLOCK_SIZE); }
    return ctx;
}

struct AesCtx aes256_init(const &stack unsigned char key) {
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
        p[i] = v;
        i = i + 1;
    }
}
