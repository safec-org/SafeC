// SafeC Standard Library — Timer HAL
// Hardware timer control. Freestanding-safe.
#pragma once

struct Timer {
    void*        base;       // MMIO base address
    unsigned int prescaler;  // clock divider

    // Set the auto-reload (period) value.
    void         set_period(unsigned int period);

    // Start the timer.
    void         start();

    // Stop the timer.
    void         stop();

    // Read the current counter value.
    unsigned int read() const;

    // Clear the interrupt/overflow flag.
    void         clear_flag();

    // Check if the overflow/compare flag is set. Returns 1 if set.
    int          flag_set() const;
};

// Initialize a hardware timer at `base` with the given prescaler.
struct Timer timer_init(void* base, unsigned int prescaler);
