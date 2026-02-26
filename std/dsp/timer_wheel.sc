#pragma once
#include "timer_wheel.h"

// ── timer_wheel_init ──────────────────────────────────────────────────────────
// Zero-initialise the wheel.  Must be called before any other operation.
void timer_wheel_init(struct TimerWheel* tw) {
    tw->current_tick = 0;
    int i = 0;
    while (i < WHEEL_TIMERS) {
        unsafe {
            tw->timers[i].used     = 0;
            tw->timers[i].periodic = 0;
            tw->timers[i].period   = 0;
            tw->timers[i].expires  = 0;
            tw->timers[i].callback = NULL;
            tw->timers[i].ctx      = NULL;
        }
        i = i + 1;
    }
}

// ── Internal helper: allocate a free timer slot ───────────────────────────────
// Returns 0-based index or -1 if full.
static int alloc_slot(struct TimerWheel* tw) {
    int i = 0;
    while (i < WHEEL_TIMERS) {
        unsafe {
            if (tw->timers[i].used == 0) {
                return i;
            }
        }
        i = i + 1;
    }
    return -1;
}

// ── TimerWheel::add ───────────────────────────────────────────────────────────
int TimerWheel::add(unsigned long tick, void* callback, void* ctx) {
    int idx = alloc_slot(self);
    if (idx < 0) {
        return 0;  // wheel full
    }
    unsafe {
        self->timers[idx].expires  = tick;
        self->timers[idx].callback = callback;
        self->timers[idx].ctx      = ctx;
        self->timers[idx].used     = 1;
        self->timers[idx].periodic = 0;
        self->timers[idx].period   = 0;
    }
    // Return 1-based ID.
    return idx + 1;
}

// ── TimerWheel::add_periodic ──────────────────────────────────────────────────
int TimerWheel::add_periodic(unsigned long period, void* callback, void* ctx) {
    if (period == 0) {
        return 0;
    }
    int idx = alloc_slot(self);
    if (idx < 0) {
        return 0;
    }
    unsafe {
        self->timers[idx].expires  = self->current_tick + period;
        self->timers[idx].callback = callback;
        self->timers[idx].ctx      = ctx;
        self->timers[idx].used     = 1;
        self->timers[idx].periodic = 1;
        self->timers[idx].period   = period;
    }
    return idx + 1;
}

// ── TimerWheel::cancel ────────────────────────────────────────────────────────
void TimerWheel::cancel(int id) {
    if (id <= 0 || id > WHEEL_TIMERS) {
        return;
    }
    int idx = id - 1;
    unsafe {
        self->timers[idx].used = 0;
    }
}

// ── TimerWheel::tick ──────────────────────────────────────────────────────────
// Advance by one tick.  Fire all timers whose expires == new current_tick.
// Periodic timers are rescheduled by incrementing expires by their period.
int TimerWheel::tick() {
    self->current_tick = self->current_tick + 1;
    unsigned long now = self->current_tick;

    int fired = 0;
    int i = 0;
    while (i < WHEEL_TIMERS) {
        int do_fire = 0;
        unsafe {
            if (self->timers[i].used == 1 && self->timers[i].expires == now) {
                do_fire = 1;
            }
        }
        if (do_fire == 1) {
            // Invoke the callback.
            void* cb;
            void* ctx;
            unsafe {
                cb  = self->timers[i].callback;
                ctx = self->timers[i].ctx;
            }
            // Call the callback through a void(*)(void*) function pointer.
            unsafe {
                typedef void (*TimerCB)(void*);
                TimerCB fn = (TimerCB)cb;
                fn(ctx);
            }
            fired = fired + 1;

            // Reschedule periodic timers; free one-shot timers.
            int is_periodic;
            unsigned long period;
            unsafe {
                is_periodic = self->timers[i].periodic;
                period      = self->timers[i].period;
            }
            if (is_periodic == 1) {
                unsafe {
                    self->timers[i].expires = now + period;
                }
            } else {
                unsafe {
                    self->timers[i].used = 0;
                }
            }
        }
        i = i + 1;
    }
    return fired;
}

// ── TimerWheel::now ───────────────────────────────────────────────────────────
unsigned long TimerWheel::now() const {
    return self->current_tick;
}
