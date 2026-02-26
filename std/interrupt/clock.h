// SafeC Standard Library — System Clock Configuration
// PLL and clock source management. Freestanding-safe.
#pragma once

// Clock sources
#define CLK_SRC_HSI   0   // High-speed internal oscillator
#define CLK_SRC_HSE   1   // High-speed external oscillator
#define CLK_SRC_PLL   2   // PLL output

struct ClockConfig {
    void*        base;      // clock control MMIO base
    int          source;    // CLK_SRC_HSI, CLK_SRC_HSE, or CLK_SRC_PLL
    unsigned int sys_freq;  // system clock frequency in Hz
    unsigned int ahb_div;   // AHB prescaler divider
    unsigned int apb1_div;  // APB1 prescaler divider
    unsigned int apb2_div;  // APB2 prescaler divider

    // Select clock source.
    void         set_source(int source);

    // Configure PLL multiplier and divider. Returns 1 on success.
    int          configure_pll(unsigned int mul, unsigned int div);

    // Set AHB bus prescaler.
    void         set_ahb_div(unsigned int div);

    // Set APB1 bus prescaler.
    void         set_apb1_div(unsigned int div);

    // Set APB2 bus prescaler.
    void         set_apb2_div(unsigned int div);

    // Get current system clock frequency in Hz.
    unsigned int get_freq() const;
};

// Initialize clock configuration at `base`.
struct ClockConfig clock_init(void* base);
