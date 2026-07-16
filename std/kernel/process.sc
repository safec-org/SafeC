// SafeC Standard Library — Process Control Block
#pragma once
#include <std/kernel/process.h>

namespace std {

struct PCB pcb_init(int pid, unsigned long entry, unsigned long sp, unsigned long page_table) {
    struct PCB p;
    p.pid        = pid;
    p.state      = 0; // PROC_READY
    p.priority   = 0;
    p.stack_ptr  = sp;
    p.pc         = entry;
    p.page_table = page_table;
    p.parent_pid = -1;
    p.exit_code  = 0;
    return p;
}

void PCB::set_state(int state) {
    self.state = state;
}

void PCB::set_priority(int priority) {
    self.priority = priority;
}

void PCB::save_context(unsigned long sp, unsigned long pc) {
    self.stack_ptr = sp;
    self.pc        = pc;
}

void PCB::exit(int exit_code) {
    self.state     = 3; // PROC_ZOMBIE
    self.exit_code = exit_code;
}

} // namespace std
