// SafeC Standard Library — GPIO HAL
// Memory-mapped GPIO pin abstraction. Freestanding-safe.
//
// GpioPin<C> is generic over the chip's GPIO block, a compile-time type
// whose methods do the register work:
//   GpioGeneric   reference layout: DIR +0x00, OUT +0x04, IN +0x08,
//                 PULL_EN +0x0C, PULL_SEL +0x10 (1 = up), one bit per pin
//   Stm32F4Gpio   STM32F2/F4/F7/L4/G0/H7-style port: MODER, PUPDR, IDR,
//                 ODR, BSRR (atomic set/reset)
//
//   struct GpioPin<Stm32F4Gpio> led = std::gpio_init(gpioa, 5, GPIO_OUTPUT);
//   led.write(1);
#pragma once

// Pin direction
#define GPIO_INPUT  0
#define GPIO_OUTPUT 1

// Pin pull configuration
#define GPIO_PULL_NONE 0
#define GPIO_PULL_UP   1
#define GPIO_PULL_DOWN 2

namespace std {

struct GpioGeneric {
    int  unused_;
    void set_direction(unsigned long base, int pin, int direction) const;
    void write(unsigned long base, int pin, int value) const;
    int  read(unsigned long base, int pin) const;
    void toggle(unsigned long base, int pin) const;
    void set_pull(unsigned long base, int pin, int pull) const;
};

struct Stm32F4Gpio {
    int  unused_;
    void set_direction(unsigned long base, int pin, int direction) const;
    void write(unsigned long base, int pin, int value) const;
    int  read(unsigned long base, int pin) const;
    void toggle(unsigned long base, int pin) const;
    void set_pull(unsigned long base, int pin, int pull) const;
};

generic<C>
struct GpioPin {
    &volatile unsigned int base; // MMIO base address of the GPIO port
    int          pin;        // pin number within the port
    unsigned int pin_mask;   // 1 << pin
    int          direction;  // GPIO_INPUT or GPIO_OUTPUT
    C            chip;

    // Set pin direction (GPIO_INPUT or GPIO_OUTPUT).
    void         set_direction(int direction);
    // Write 1 (high) or 0 (low) to an output pin.
    void         write(int value);
    // Read pin state. Returns 0 or 1.
    int          read() const;
    // Toggle output pin.
    void         toggle();
    // GPIO_PULL_NONE, GPIO_PULL_UP or GPIO_PULL_DOWN.
    void         set_pull(int pull);
    unsigned long addr_() const;
};

// Initialize a GPIO pin: `base` = port base, `pin` = pin number (0-31; 0-15
// on STM32). The chip type comes from the target:
//   struct GpioPin<GpioGeneric> p = std::gpio_init(base, 3, GPIO_INPUT);
generic<C>
struct GpioPin<C> gpio_init(&volatile unsigned int base, int pin, int direction) {
    struct GpioPin<C> p;
    p.base = base;
    p.pin = pin;
    p.pin_mask = (unsigned int)1 << pin;
    p.direction = direction;
    p.chip.set_direction(p.addr_(), pin, direction);
    return p;
}

generic<C>
unsigned long GpioPin::addr_() const { unsafe { return (unsigned long)self.base; } }
generic<C>
void GpioPin::set_direction(int direction) {
    self.direction = direction;
    self.chip.set_direction(self.addr_(), self.pin, direction);
}
generic<C>
void GpioPin::write(int value) { self.chip.write(self.addr_(), self.pin, value); }
generic<C>
int GpioPin::read() const { return self.chip.read(self.addr_(), self.pin); }
generic<C>
void GpioPin::toggle() { self.chip.toggle(self.addr_(), self.pin); }
generic<C>
void GpioPin::set_pull(int pull) { self.chip.set_pull(self.addr_(), self.pin, pull); }

} // namespace std
