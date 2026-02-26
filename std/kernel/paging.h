// SafeC Standard Library — Page Table Management
// Virtual memory page table entries and mapping. Freestanding-safe.
#pragma once

// Page table flags
#define PAGE_PRESENT    1
#define PAGE_WRITABLE   2
#define PAGE_USER       4
#define PAGE_WRITE_THRU 8
#define PAGE_NO_CACHE   16
#define PAGE_ACCESSED   32
#define PAGE_DIRTY      64
#define PAGE_HUGE       128

#define PAGE_SIZE       4096
#define PAGE_TABLE_SIZE 512

struct PageEntry {
    unsigned long value;  // raw page table entry (flags + physical address)
};

struct PageTable {
    struct PageEntry entries[PAGE_TABLE_SIZE];

    // Initialize all entries to zero.
    void          init();

    // Map entry at `idx` to `phys_addr` with `flags`.
    void          map(int idx, unsigned long phys_addr, unsigned int flags);

    // Unmap entry at `idx`.
    void          unmap(int idx);

    // Check if entry at `idx` is present. Returns 1 if present.
    int           is_present(int idx) const;

    // Get physical address from entry at `idx`. Returns 0 if not present.
    unsigned long get_phys(int idx) const;

    // Get flags from entry at `idx`.
    unsigned int  get_flags(int idx) const;

    // Set flags on existing mapping (preserves physical address).
    void          set_flags(int idx, unsigned int flags);
};
