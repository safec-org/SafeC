// SafeC Standard Library — I2C HAL chips (see i2c.h)
#pragma once
#include <std/hal/i2c.h>
#include <std/hal/reg.h>

namespace std {

#define I2C_SPINS_ 100000

// ── I2cGeneric ───────────────────────────────────────────────────────────────
// DATA +0, STATUS +4 (bit0 busy, bit1 ACK, bit2 error), CTRL +8, ADDR +0xC, SPEED +0x10.
void I2cGeneric::init(unsigned long base, unsigned int speed, unsigned int clock_hz) const {
    hal_wr32_(base, 0x10UL, speed);
}

static int i2c_gen_wait_(unsigned long base) {
    int spins = I2C_SPINS_;
    while (spins > 0) {
        unsigned int st = hal_rd32_(base, 0x04UL);
        if ((st & 1u) == 0u) {
            if ((st & 4u) != 0u) { return I2C_ERROR; }
            if ((st & 2u) == 0u) { return I2C_NACK; }
            return I2C_OK;
        }
        spins = spins - 1;
    }
    return I2C_ERROR;                                   // timeout
}
// START with the address byte; on failure sends STOP.
static int i2c_gen_start_(unsigned long base, unsigned int addrByte) {
    hal_wr32_(base, 0x0CUL, addrByte);
    hal_wr32_(base, 0x08UL, 1u);                        // START
    int r = i2c_gen_wait_(base);
    if (r != I2C_OK) { hal_wr32_(base, 0x08UL, 2u); }   // STOP: release the bus
    return r;
}

int I2cGeneric::write(unsigned long base, unsigned char addr, const &unsigned char data, unsigned long len) const {
    int r = i2c_gen_start_(base, (unsigned int)addr << 1);
    if (r != I2C_OK) { return r; }
    unsigned long i = 0UL;
    while (i < len) {
        unsafe { hal_wr32_(base, 0x00UL, (unsigned int)data[i]); }
        hal_wr32_(base, 0x08UL, 4u);                    // SEND
        r = i2c_gen_wait_(base);
        if (r != I2C_OK) { hal_wr32_(base, 0x08UL, 2u); return r; }
        i = i + 1UL;
    }
    hal_wr32_(base, 0x08UL, 2u);                        // STOP
    return I2C_OK;
}

int I2cGeneric::read(unsigned long base, unsigned char addr, &unsigned char data, unsigned long len) const {
    int r = i2c_gen_start_(base, ((unsigned int)addr << 1) | 1u);
    if (r != I2C_OK) { return r; }
    unsigned long i = 0UL;
    while (i < len) {
        hal_wr32_(base, 0x08UL, i + 1UL < len ? 8u : 16u);   // ACK all but the last byte
        r = i2c_gen_wait_(base);
        if (r != I2C_OK) { hal_wr32_(base, 0x08UL, 2u); return r; }
        unsafe { data[i] = (unsigned char)hal_rd32_(base, 0x00UL); }
        i = i + 1UL;
    }
    hal_wr32_(base, 0x08UL, 2u);
    return I2C_OK;
}

int I2cGeneric::probe(unsigned long base, unsigned char addr) const {
    int r = i2c_gen_start_(base, (unsigned int)addr << 1);
    if (r != I2C_OK) { return 0; }
    hal_wr32_(base, 0x08UL, 2u);
    return 1;
}

// ── STM32F4 I2C (v1) ─────────────────────────────────────────────────────────
// CR1 +0 (PE bit0, START bit8, STOP bit9, ACK bit10), CR2 +4 (FREQ = MHz),
// DR +0x10, SR1 +0x14 (SB bit0, ADDR bit1, BTF bit2, RXNE bit6, TXE bit7,
// AF bit10), SR2 +0x18, CCR +0x1C, TRISE +0x20.
#define ST_CR1_  0x00UL
#define ST_CR2_  0x04UL
#define ST_DR_   0x10UL
#define ST_SR1_  0x14UL
#define ST_SR2_  0x18UL
#define ST_CCR_  0x1CUL
#define ST_TRISE_ 0x20UL

void Stm32F4I2c::init(unsigned long base, unsigned int speed, unsigned int clock_hz) const {
    if (speed == 0u) { speed = 100000u; }
    unsigned int mhz = clock_hz / 1000000u;
    hal_wr32_(base, ST_CR1_, 0u);                       // PE off while configuring
    hal_wr32_(base, ST_CR2_, mhz & 0x3Fu);
    if (speed <= 100000u) {
        unsigned int ccr = clock_hz / (2u * speed);
        if (ccr < 4u) { ccr = 4u; }
        hal_wr32_(base, ST_CCR_, ccr & 0xFFFu);
        hal_wr32_(base, ST_TRISE_, mhz + 1u);           // 1000 ns max rise time
    } else {
        unsigned int ccr = clock_hz / (3u * speed);     // DUTY = 0: Tlow/Thigh = 2
        if (ccr < 1u) { ccr = 1u; }
        hal_wr32_(base, ST_CCR_, 0x8000u | (ccr & 0xFFFu));
        hal_wr32_(base, ST_TRISE_, mhz * 300u / 1000u + 1u);   // 300 ns
    }
    hal_wr32_(base, ST_CR1_, 1u);                       // PE
}

// Waits for 'mask' in SR1. NACK (AF) or a timeout sends STOP.
static int st_i2c_wait_(unsigned long base, unsigned int mask) {
    int spins = I2C_SPINS_;
    while (spins > 0) {
        unsigned int sr1 = hal_rd32_(base, ST_SR1_);
        if ((sr1 & 0x400u) != 0u) {                     // AF: not acknowledged
            hal_wr32_(base, ST_SR1_, sr1 & ~0x400u);
            hal_set32_(base, ST_CR1_, 0x200u);          // STOP
            return I2C_NACK;
        }
        if ((sr1 & mask) != 0u) { return I2C_OK; }
        spins = spins - 1;
    }
    hal_set32_(base, ST_CR1_, 0x200u);
    return I2C_ERROR;
}

// START + address byte, through ADDR; ADDR is left set (cleared by the
// caller reading SR1 then SR2, after it has set up ACK/STOP).
static int st_i2c_start_(unsigned long base, unsigned int addrByte) {
    hal_set32_(base, ST_CR1_, 0x100u);                  // START
    int r = st_i2c_wait_(base, 0x01u);                  // SB
    if (r != I2C_OK) { return r; }
    hal_wr32_(base, ST_DR_, addrByte);
    return st_i2c_wait_(base, 0x02u);                   // ADDR (or AF -> NACK)
}

static void st_i2c_clear_addr_(unsigned long base) {
    hal_rd32_(base, ST_SR1_);
    hal_rd32_(base, ST_SR2_);
}

int Stm32F4I2c::write(unsigned long base, unsigned char addr, const &unsigned char data, unsigned long len) const {
    int r = st_i2c_start_(base, (unsigned int)addr << 1);
    if (r != I2C_OK) { return r; }
    st_i2c_clear_addr_(base);
    unsigned long i = 0UL;
    while (i < len) {
        r = st_i2c_wait_(base, 0x80u);                  // TXE
        if (r != I2C_OK) { return r; }
        unsafe { hal_wr32_(base, ST_DR_, (unsigned int)data[i]); }
        i = i + 1UL;
    }
    if (len > 0UL) {
        r = st_i2c_wait_(base, 0x04u);                  // BTF: last byte out
        if (r != I2C_OK) { return r; }
    }
    hal_set32_(base, ST_CR1_, 0x200u);                  // STOP
    return I2C_OK;
}

int Stm32F4I2c::read(unsigned long base, unsigned char addr, &unsigned char data, unsigned long len) const {
    if (len > 1UL) { hal_set32_(base, ST_CR1_, 0x400u); }      // ACK all but the last byte
    else { hal_clr32_(base, ST_CR1_, 0x400u); }
    int r = st_i2c_start_(base, ((unsigned int)addr << 1) | 1u);
    if (r != I2C_OK) { return r; }
    st_i2c_clear_addr_(base);
    if (len <= 1UL) { hal_set32_(base, ST_CR1_, 0x200u); }     // STOP right after this byte
    unsigned long i = 0UL;
    while (i < len) {
        r = st_i2c_wait_(base, 0x40u);                  // RXNE
        if (r != I2C_OK) { return r; }
        if (len > 1UL && i + 2UL == len) {              // next byte is the last: NACK + STOP
            hal_clr32_(base, ST_CR1_, 0x400u);
            hal_set32_(base, ST_CR1_, 0x200u);
        }
        unsafe { data[i] = (unsigned char)hal_rd32_(base, ST_DR_); }
        i = i + 1UL;
    }
    return I2C_OK;
}

int Stm32F4I2c::probe(unsigned long base, unsigned char addr) const {
    int r = st_i2c_start_(base, (unsigned int)addr << 1);
    if (r != I2C_OK) { return 0; }
    st_i2c_clear_addr_(base);
    hal_set32_(base, ST_CR1_, 0x200u);
    return 1;
}

} // namespace std
