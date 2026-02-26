// SafeC Standard Library — SPI HAL
// Generic MMIO layout:
//   base + 0x00 = data register (write TX, read RX)
//   base + 0x04 = status register (bit 0 = busy, bit 1 = TX empty)
//   base + 0x08 = control register (mode, enable)
//   base + 0x0C = chip select register
#pragma once
#include "spi.h"

struct SpiDevice spi_init(void* base, int mode) {
    struct SpiDevice s;
    s.base = base;
    s.mode = mode;
    unsafe {
        unsigned int* ctrl = (unsigned int*)((unsigned long)base + (unsigned long)8);
        unsigned int val = ((unsigned int)mode & (unsigned int)3) | (unsigned int)0x80;
        volatile_store(ctrl, val);
    }
    return s;
}

unsigned char SpiDevice::transfer(unsigned char tx) {
    unsafe {
        unsigned int* status   = (unsigned int*)((unsigned long)self.base + (unsigned long)4);
        unsigned int* data_reg = (unsigned int*)self.base;

        // Wait until TX empty
        while ((volatile_load(status) & (unsigned int)2) == (unsigned int)0) { }

        // Write TX byte
        volatile_store(data_reg, (unsigned int)tx);

        // Wait until not busy
        while ((volatile_load(status) & (unsigned int)1) != (unsigned int)0) { }

        // Read RX byte
        return (unsigned char)volatile_load(data_reg);
    }
}

void SpiDevice::cs_assert() {
    unsafe {
        unsigned int* cs = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        volatile_store(cs, (unsigned int)0);
    }
}

void SpiDevice::cs_deassert() {
    unsafe {
        unsigned int* cs = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        volatile_store(cs, (unsigned int)1);
    }
}

void SpiDevice::write(const unsigned char* tx, unsigned char* rx, unsigned long len) {
    unsigned long i = (unsigned long)0;
    while (i < len) {
        unsafe {
            unsigned char b = self.transfer(tx[i]);
            if (rx != (unsigned char*)0) { rx[i] = b; }
        }
        i = i + (unsigned long)1;
    }
}

void SpiDevice::read(unsigned char* rx, unsigned long len) {
    unsigned long i = (unsigned long)0;
    while (i < len) {
        unsafe { rx[i] = self.transfer((unsigned char)0xFF); }
        i = i + (unsigned long)1;
    }
}
