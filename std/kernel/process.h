// SafeC Standard Library — Process Control Block
// Minimal PCB for kernel-level process management. Freestanding-safe.
#pragma once

// Process states
#define PROC_READY    0
#define PROC_RUNNING  1
#define PROC_BLOCKED  2
#define PROC_ZOMBIE   3

struct PCB {
    int           pid;         // process ID
    int           state;       // PROC_READY, PROC_RUNNING, etc.
    int           priority;    // scheduling priority (higher = more important)
    unsigned long stack_ptr;   // saved stack pointer
    unsigned long pc;          // saved program counter
    unsigned long page_table;  // physical address of page table root
    int           parent_pid;  // parent process ID (-1 if none)
    int           exit_code;   // exit code (valid when ZOMBIE)

    // Set process state.
    void          set_state(int state);

    // Set process priority.
    void          set_priority(int priority);

    // Save context (stack pointer + program counter).
    void          save_context(unsigned long sp, unsigned long pc);

    // Mark process as zombie with the given exit code.
    void          exit(int exit_code);
};

// Initialize a PCB with the given pid, entry point, and stack.
struct PCB pcb_init(int pid, unsigned long entry, unsigned long stack, unsigned long page_table);
