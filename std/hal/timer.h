// SafeC Standard Library — Timer HAL
// Hardware timer control. Freestanding-safe.
//
// Timer<C> is generic over the timer block, a compile-time chip type:
//   TimerGeneric  reference layout: CTRL +0x00 (bit 0 enable), PRESCALER
//                 +0x04, PERIOD +0x08, COUNT +0x0C, STATUS +0x10 (bit 0 flag)
//   Stm32F4Tim    STM32 general-purpose timer (TIM2-TIM5 style): PSC, ARR,
//                 CNT, update flag (UIF)
#pragma once

namespace std {

// What a timer block type provides; Timer<C> requires it.
trait TimerChip {
    void         init(unsigned long base, unsigned int prescaler) const;
    void         set_period(unsigned long base, unsigned int period) const;
    void         run(unsigned long base, int on) const;
    unsigned int count(unsigned long base) const;
    int          flag(unsigned long base) const;
    void         clear_flag(unsigned long base) const;
}

struct TimerGeneric {
    int          unused_;
    void         init(unsigned long base, unsigned int prescaler) const;
    void         set_period(unsigned long base, unsigned int period) const;
    void         run(unsigned long base, int on) const;
    unsigned int count(unsigned long base) const;
    int          flag(unsigned long base) const;
    void         clear_flag(unsigned long base) const;
};
struct Stm32F4Tim {
    int          unused_;
    void         init(unsigned long base, unsigned int prescaler) const;
    void         set_period(unsigned long base, unsigned int period) const;
    void         run(unsigned long base, int on) const;
    unsigned int count(unsigned long base) const;
    int          flag(unsigned long base) const;
    void         clear_flag(unsigned long base) const;
};

generic<C: TimerChip>
struct Timer {
    &volatile unsigned int base; // MMIO base address
    unsigned int prescaler;  // clock divider
    C            chip;

    // Initialize a hardware timer at `base` with the given prescaler (STM32: the
    // counter runs at clock / (prescaler + 1)). The chip type comes from the target.
    static struct Timer<C> new(&volatile unsigned int base, unsigned int prescaler);

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
    unsigned long addr_() const;
};

generic<C: TimerChip>
struct Timer<C> Timer::new(&volatile unsigned int base, unsigned int prescaler) {
    struct Timer<C> t;
    t.base = base;
    t.prescaler = prescaler;
    t.chip.init(t.addr_(), prescaler);
    return t;
}

generic<C: TimerChip>
unsigned long Timer::addr_() const { unsafe { return (unsigned long)self.base; } }
generic<C: TimerChip>
void Timer::set_period(unsigned int period) { self.chip.set_period(self.addr_(), period); }
generic<C: TimerChip>
void Timer::start() { self.chip.run(self.addr_(), 1); }
generic<C: TimerChip>
void Timer::stop() { self.chip.run(self.addr_(), 0); }
generic<C: TimerChip>
unsigned int Timer::read() const { return self.chip.count(self.addr_()); }
generic<C: TimerChip>
void Timer::clear_flag() { self.chip.clear_flag(self.addr_()); }
generic<C: TimerChip>
int Timer::flag_set() const { return self.chip.flag(self.addr_()); }

} // namespace std
