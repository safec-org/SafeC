// SafeC Standard Library — IPC Mailbox
#pragma once
#include "ipc.h"

extern void* memcpy(void* dst, const void* src, unsigned long n);

struct Mailbox mailbox_init(int owner_pid) {
    struct Mailbox mb;
    mb.head      = 0;
    mb.tail      = 0;
    mb.count     = 0;
    mb.owner_pid = owner_pid;
    return mb;
}

int Mailbox::send(int sender_pid, int type, const void* payload, unsigned long size) {
    if (self.count >= 64) { return 0; } // full
    if (size > (unsigned long)256) { size = (unsigned long)256; } // truncate

    int idx = self.head;
    self.messages[idx].sender_pid = sender_pid;
    self.messages[idx].type       = type;
    self.messages[idx].size       = size;
    if (size > (unsigned long)0 && payload != (void*)0) {
        unsafe { memcpy((void*)self.messages[idx].payload, payload, size); }
    }

    self.head  = (self.head + 1) & 63;
    self.count = self.count + 1;
    return 1;
}

int Mailbox::recv(&stack Message out) {
    if (self.count <= 0) { return 0; } // empty

    int idx = self.tail;
    out.sender_pid = self.messages[idx].sender_pid;
    out.type       = self.messages[idx].type;
    out.size       = self.messages[idx].size;
    if (out.size > (unsigned long)0) {
        unsafe { memcpy((void*)out.payload, (const void*)self.messages[idx].payload, out.size); }
    }

    self.tail  = (self.tail + 1) & 63;
    self.count = self.count - 1;
    return 1;
}

int Mailbox::peek(&stack Message out) const {
    if (self.count <= 0) { return 0; }

    int idx = self.tail;
    out.sender_pid = self.messages[idx].sender_pid;
    out.type       = self.messages[idx].type;
    out.size       = self.messages[idx].size;
    if (out.size > (unsigned long)0) {
        unsafe { memcpy((void*)out.payload, (const void*)self.messages[idx].payload, out.size); }
    }
    return 1;
}

int Mailbox::has_msg() const {
    if (self.count > 0) { return 1; }
    return 0;
}

int Mailbox::length() const { return self.count; }

void Mailbox::clear() {
    self.head  = 0;
    self.tail  = 0;
    self.count = 0;
}
