// SafeC Standard Library — Priority Round-Robin Scheduler
// Manages an array of PCBs and selects the next process to run.
// Freestanding-safe.
#pragma once

#include "process.h"

#define SCHED_MAX_PROCS 256

struct Scheduler {
    struct PCB    procs[SCHED_MAX_PROCS];
    int           count;    // total number of processes
    int           current;  // index of currently running process (-1 if none)

    // Add a process. Returns the process index, or -1 if full.
    int           spawn(&stack PCB proc);

    // Select the next READY process with highest priority.
    // Returns the index of the selected process, or -1 if none are ready.
    int           next();

    // Mark current process as READY and select the next one.
    int           yield();

    // Block the current process (waiting on I/O, etc.).
    void          block_current();

    // Unblock a process by index (set to READY).
    void          unblock(int idx);

    // Remove a zombie process by index.
    void          remove(int idx);

    // Return the number of READY processes.
    int           ready_count() const;
};

// Initialize the scheduler.
struct Scheduler sched_init();
