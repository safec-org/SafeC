// SafeC Standard Library — I2C HAL
// Generic MMIO layout:
//   base + 0x00 = data register
//   base + 0x04 = status register (bit 0 = busy, bit 1 = ACK, bit 2 = error)
//   base + 0x08 = control register (start/stop/ack bits)
//   base + 0x0C = address register
//   base + 0x10 = speed register
#pragma once
#include "i2c.h"

struct I2cBus i2c_init(void* base, unsigned int speed) {
    struct I2cBus bus;
    bus.base  = base;
    bus.speed = speed;
    unsafe {
        unsigned int* spd_reg = (unsigned int*)((unsigned long)base + (unsigned long)16);
        volatile_store(spd_reg, speed);
    }
    return bus;
}

// Wait until bus is not busy. Returns 0 on success, I2C_ERROR on timeout.
int I2cBus::wait_() {
    unsafe {
        unsigned int* status = (unsigned int*)((unsigned long)self.base + (unsigned long)4);
        int timeout = 100000;
        while (timeout > 0) {
            unsigned int val = volatile_load(status);
            if ((val & (unsigned int)1) == (unsigned int)0) {
                if ((val & (unsigned int)4) != (unsigned int)0) { return 2; } // I2C_ERROR
                if ((val & (unsigned int)2) == (unsigned int)0) { return 1; } // I2C_NACK
                return 0; // I2C_OK
            }
            timeout = timeout - 1;
        }
    }
    return 2; // I2C_ERROR (timeout)
}

int I2cBus::write(unsigned char addr, const unsigned char* data, unsigned long len) {
    unsafe {
        unsigned int* addr_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        unsigned int* ctrl     = (unsigned int*)((unsigned long)self.base + (unsigned long)8);
        unsigned int* data_reg = (unsigned int*)self.base;

        volatile_store(addr_reg, (unsigned int)addr << 1);
        volatile_store(ctrl, (unsigned int)1); // START

        int result = self.wait_();
        if (result != 0) { return result; }

        unsigned long i = (unsigned long)0;
        while (i < len) {
            volatile_store(data_reg, (unsigned int)data[i]);
            volatile_store(ctrl, (unsigned int)4); // SEND
            result = self.wait_();
            if (result != 0) { return result; }
            i = i + (unsigned long)1;
        }

        volatile_store(ctrl, (unsigned int)2); // STOP
    }
    return 0;
}

int I2cBus::read(unsigned char addr, unsigned char* data, unsigned long len) {
    unsafe {
        unsigned int* addr_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        unsigned int* ctrl     = (unsigned int*)((unsigned long)self.base + (unsigned long)8);
        unsigned int* data_reg = (unsigned int*)self.base;

        volatile_store(addr_reg, ((unsigned int)addr << 1) | (unsigned int)1);
        volatile_store(ctrl, (unsigned int)1); // START

        int result = self.wait_();
        if (result != 0) { return result; }

        unsigned long i = (unsigned long)0;
        while (i < len) {
            if (i < len - (unsigned long)1) {
                volatile_store(ctrl, (unsigned int)8);  // READ + ACK
            } else {
                volatile_store(ctrl, (unsigned int)16); // READ + NACK
            }
            result = self.wait_();
            if (result != 0) { return result; }
            data[i] = (unsigned char)volatile_load(data_reg);
            i = i + (unsigned long)1;
        }

        volatile_store(ctrl, (unsigned int)2); // STOP
    }
    return 0;
}

int I2cBus::write_read(unsigned char addr,
                       const unsigned char* wr, unsigned long wr_len,
                       unsigned char* rd,       unsigned long rd_len) {
    int result = self.write(addr, wr, wr_len);
    if (result != 0) { return result; }
    return self.read(addr, rd, rd_len);
}

int I2cBus::probe(unsigned char addr) {
    unsafe {
        unsigned int* addr_reg = (unsigned int*)((unsigned long)self.base + (unsigned long)12);
        unsigned int* ctrl     = (unsigned int*)((unsigned long)self.base + (unsigned long)8);

        volatile_store(addr_reg, (unsigned int)addr << 1);
        volatile_store(ctrl, (unsigned int)1); // START
        int result = self.wait_();
        volatile_store(ctrl, (unsigned int)2); // STOP

        if (result == 0) { return 1; }
        return 0;
    }
}
