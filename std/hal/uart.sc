// SafeC Standard Library — UART HAL
// Generic MMIO layout:
//   base + 0x00 = data register (TX/RX)
//   base + 0x04 = status register (bit 0 = RX ready, bit 1 = TX ready)
//   base + 0x08 = control/baud register
#pragma once
#include "uart.h"

struct Uart uart_init(void* base, unsigned int baud) {
    struct Uart u;
    u.base = base;
    u.baud = baud;
    unsafe {
        unsigned int* ctrl = (unsigned int*)((unsigned long)base + (unsigned long)8);
        volatile_store(ctrl, baud);
    }
    return u;
}

int Uart::tx_ready() const {
    unsafe {
        unsigned int* status = (unsigned int*)((unsigned long)self.base + (unsigned long)4);
        unsigned int val = volatile_load(status);
        if ((val & (unsigned int)2) != (unsigned int)0) { return 1; }
        return 0;
    }
}

int Uart::rx_ready() const {
    unsafe {
        unsigned int* status = (unsigned int*)((unsigned long)self.base + (unsigned long)4);
        unsigned int val = volatile_load(status);
        if ((val & (unsigned int)1) != (unsigned int)0) { return 1; }
        return 0;
    }
}

void Uart::write_byte(unsigned char c) {
    while (self.tx_ready() == 0) { }
    unsafe {
        unsigned int* data_reg = (unsigned int*)self.base;
        volatile_store(data_reg, (unsigned int)c);
    }
}

unsigned char Uart::read_byte() {
    while (self.rx_ready() == 0) { }
    unsafe {
        unsigned int* data_reg = (unsigned int*)self.base;
        return (unsigned char)volatile_load(data_reg);
    }
}

void Uart::write_str(const char* s) {
    unsafe {
        int i = 0;
        while (s[i] != (char)0) {
            self.write_byte((unsigned char)s[i]);
            i = i + 1;
        }
    }
}
