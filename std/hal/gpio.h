// SafeC Standard Library — GPIO HAL
// Memory-mapped GPIO pin abstraction. Freestanding-safe.
#pragma once

// Pin direction
#define GPIO_INPUT  0
#define GPIO_OUTPUT 1

// Pin pull configuration
#define GPIO_PULL_NONE 0
#define GPIO_PULL_UP   1
#define GPIO_PULL_DOWN 2

struct GpioPin {
    void*        base;       // MMIO base address of GPIO port
    unsigned int pin_mask;   // bitmask for this pin (1 << pin_number)
    int          direction;  // GPIO_INPUT or GPIO_OUTPUT

    // Set pin direction (GPIO_INPUT or GPIO_OUTPUT).
    void         set_direction(int direction);

    // Write 1 (high) or 0 (low) to an output pin.
    void         write(int value);

    // Read pin state. Returns 0 or 1.
    int          read() const;

    // Toggle output pin.
    void         toggle();

    // Configure pull-up/pull-down. Requires pull register at base + 0x0C.
    void         set_pull(int pull);
};

// Initialize a GPIO pin. `base` = MMIO port base, `pin` = pin number (0-31).
struct GpioPin gpio_init(void* base, int pin, int direction);
