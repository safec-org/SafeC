// SafeC Standard Library — IPC Mailbox
// Mailbox-based message passing between processes. Freestanding-safe.
#pragma once

#define MAILBOX_CAPACITY 64
#define MSG_MAX_SIZE     256

struct Message {
    int           sender_pid;            // sender process ID
    int           type;                  // user-defined message type
    unsigned long size;                  // payload size in bytes
    unsigned char payload[MSG_MAX_SIZE]; // message payload
};

struct Mailbox {
    struct Message messages[MAILBOX_CAPACITY];
    int            head;       // write index
    int            tail;       // read index
    int            count;      // current message count
    int            owner_pid;  // owning process ID

    // Send a message. Returns 1 on success, 0 if full.
    int            send(int sender_pid, int type, const void* payload, unsigned long size);

    // Receive next message. Copies into `out`. Returns 1 on success, 0 if empty.
    int            recv(&stack Message out);

    // Peek at next message without removing it. Returns 1 if available.
    int            peek(&stack Message out) const;

    // Check if the mailbox has messages. Returns 1 if non-empty.
    int            has_msg() const;

    // Return the number of pending messages.
    int            length() const;

    // Clear all messages.
    void           clear();
};

// Initialize a mailbox for the given process.
struct Mailbox mailbox_init(int owner_pid);
