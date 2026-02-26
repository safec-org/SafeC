// SafeC Standard Library — TLSF Allocator
#pragma once
#include "tlsf.h"

extern void* malloc(unsigned long size);
extern void  free(void* ptr);
extern void* memset(void* ptr, int val, unsigned long n);

// Block header size (32 bytes: size + prev_phys + next_free + prev_free)
unsigned long tlsf_hdr_size_() {
    return (unsigned long)32;
}

// Minimum block size (must fit header pointers when free)
unsigned long tlsf_min_block_() {
    return (unsigned long)32;
}

// ── TlsfBlock methods ────────────────────────────────────────────────────────

int TlsfBlock::is_free_() const {
    unsafe { return (int)(self.size & (unsigned long)1); }
}

unsigned long TlsfBlock::block_size_() const {
    unsafe { return self.size & ~(unsigned long)1; }
}

void TlsfBlock::set_free_() {
    unsafe { self.size = self.size | (unsigned long)1; }
}

void TlsfBlock::set_used_() {
    unsafe { self.size = self.size & ~(unsigned long)1; }
}

// ── Utility free function ────────────────────────────────────────────────────

void tlsf_mapping_(unsigned long size, int* fl, int* sl) {
    if (size < (unsigned long)256) {
        unsafe {
            *fl = 0;
            *sl = (int)(size / (unsigned long)16);
        }
    } else {
        int f = 0;
        unsigned long tmp = size;
        while (tmp >= (unsigned long)2) {
            tmp = tmp / (unsigned long)2;
            f = f + 1;
        }
        unsafe {
            *fl = f;
            int shift = f - 4;
            if (shift < 0) { shift = 0; }
            unsigned long s = size;
            int i = 0;
            while (i < shift) {
                s = s / (unsigned long)2;
                i = i + 1;
            }
            int sli = (int)s - 16;
            if (sli < 0) { sli = 0; }
            if (sli > 15) { sli = 15; }
            *sl = sli;
        }
    }
}

// ── TlsfAllocator methods ────────────────────────────────────────────────────

void TlsfAllocator::insert_(&stack TlsfBlock blk) {
    int fl = 0;
    int sl = 0;
    unsigned long sz = blk.block_size_();
    tlsf_mapping_(sz, &fl, &sl);
    blk.set_free_();

    unsafe {
        int idx = fl * 16 + sl;
        void* head = self.blocks[idx];
        blk.next_free = head;
        blk.prev_free = (void*)0;
        if (head != (void*)0) {
            struct TlsfBlock* h = (struct TlsfBlock*)head;
            h->prev_free = (void*)blk;
        }
        self.blocks[idx] = (void*)blk;
        self.fl_bitmap = self.fl_bitmap | ((unsigned int)1 << fl);
        self.sl_bitmap[fl] = self.sl_bitmap[fl] | ((unsigned int)1 << sl);
    }
}

void TlsfAllocator::remove_(&stack TlsfBlock blk) {
    int fl = 0;
    int sl = 0;
    unsigned long sz = blk.block_size_();
    tlsf_mapping_(sz, &fl, &sl);

    unsafe {
        int idx = fl * 16 + sl;
        struct TlsfBlock* prev = (struct TlsfBlock*)blk.prev_free;
        struct TlsfBlock* next = (struct TlsfBlock*)blk.next_free;
        if (prev != (struct TlsfBlock*)0) {
            prev->next_free = (void*)next;
        } else {
            self.blocks[idx] = (void*)next;
        }
        if (next != (struct TlsfBlock*)0) {
            next->prev_free = (void*)prev;
        }
        if (self.blocks[idx] == (void*)0) {
            self.sl_bitmap[fl] = self.sl_bitmap[fl] & ~((unsigned int)1 << sl);
            if (self.sl_bitmap[fl] == (unsigned int)0) {
                self.fl_bitmap = self.fl_bitmap & ~((unsigned int)1 << fl);
            }
        }
    }
}

void* TlsfAllocator::find_(unsigned long size) {
    int fl = 0;
    int sl = 0;
    tlsf_mapping_(size, &fl, &sl);

    unsafe {
        unsigned int sl_map = self.sl_bitmap[fl] & ~(((unsigned int)1 << (sl + 1)) - (unsigned int)1);
        if (sl_map == (unsigned int)0) {
            unsigned int fl_map = self.fl_bitmap & ~(((unsigned int)1 << (fl + 1)) - (unsigned int)1);
            if (fl_map == (unsigned int)0) {
                return (void*)0;
            }
            fl = 0;
            unsigned int ftmp = fl_map;
            while ((ftmp & (unsigned int)1) == (unsigned int)0) {
                ftmp = ftmp >> 1;
                fl = fl + 1;
            }
            sl_map = self.sl_bitmap[fl];
        }
        sl = 0;
        unsigned int stmp = sl_map;
        while ((stmp & (unsigned int)1) == (unsigned int)0) {
            stmp = stmp >> 1;
            sl = sl + 1;
        }
        int idx = fl * 16 + sl;
        return self.blocks[idx];
    }
}

struct TlsfAllocator tlsf_init(&heap void buffer, unsigned long cap) {
    struct TlsfAllocator a;
    a.base      = buffer;
    a.cap       = cap;
    a.fl_bitmap = (unsigned int)0;
    unsafe {
        memset((void*)a.sl_bitmap, 0, (unsigned long)128);
        memset((void*)a.blocks, 0, (unsigned long)(32 * 16 * 8));
    }
    unsigned long hdr = tlsf_hdr_size_();
    if (cap > hdr) {
        unsafe {
            struct TlsfBlock* blk = (struct TlsfBlock*)buffer;
            blk->size      = cap - hdr;
            blk->prev_phys = (void*)0;
            blk->next_free = (void*)0;
            blk->prev_free = (void*)0;
            a.insert_(*blk);
        }
    }
    return a;
}

struct TlsfAllocator tlsf_new(unsigned long cap) {
    struct TlsfAllocator a;
    unsafe { a.base = (&heap void)malloc(cap); }
    a.cap       = cap;
    a.fl_bitmap = (unsigned int)0;
    unsafe {
        memset((void*)a.sl_bitmap, 0, (unsigned long)128);
        memset((void*)a.blocks, 0, (unsigned long)(32 * 16 * 8));
    }
    unsigned long hdr = tlsf_hdr_size_();
    if (cap > hdr) {
        unsafe {
            struct TlsfBlock* blk = (struct TlsfBlock*)a.base;
            blk->size      = cap - hdr;
            blk->prev_phys = (void*)0;
            blk->next_free = (void*)0;
            blk->prev_free = (void*)0;
            a.insert_(*blk);
        }
    }
    return a;
}

&heap void TlsfAllocator::alloc(unsigned long size) {
    unsigned long hdr  = tlsf_hdr_size_();
    unsigned long min  = tlsf_min_block_();
    unsigned long aligned = (size + (unsigned long)7) & ~(unsigned long)7;
    if (aligned < min) { aligned = min; }

    void* found = self.find_(aligned);
    if (found == (void*)0) {
        return (&heap void)0;
    }

    unsafe {
        struct TlsfBlock* blk = (struct TlsfBlock*)found;
        self.remove_(*blk);
        unsigned long bsz = blk->block_size_();

        if (bsz >= aligned + hdr + min) {
            struct TlsfBlock* rest = (struct TlsfBlock*)((unsigned long)blk + hdr + aligned);
            rest->size      = bsz - aligned - hdr;
            rest->prev_phys = (void*)blk;
            rest->next_free = (void*)0;
            rest->prev_free = (void*)0;
            blk->size = aligned;
            self.insert_(*rest);
        }

        blk->set_used_();
        return (&heap void)((unsigned long)blk + hdr);
    }
}

void TlsfAllocator::free(&heap void ptr) {
    unsigned long hdr = tlsf_hdr_size_();
    unsafe {
        struct TlsfBlock* blk = (struct TlsfBlock*)((unsigned long)ptr - hdr);

        // Coalesce with next physical block if free
        unsigned long blk_end = (unsigned long)blk + hdr + blk->block_size_();
        unsigned long buf_end = (unsigned long)self.base + self.cap;
        if (blk_end < buf_end) {
            struct TlsfBlock* next = (struct TlsfBlock*)blk_end;
            if (next->is_free_() == 1) {
                self.remove_(*next);
                blk->size = blk->block_size_() + hdr + next->block_size_();
            }
        }

        // Coalesce with previous physical block if free
        if (blk->prev_phys != (void*)0) {
            struct TlsfBlock* prev = (struct TlsfBlock*)blk->prev_phys;
            if (prev->is_free_() == 1) {
                self.remove_(*prev);
                prev->size = prev->block_size_() + hdr + blk->block_size_();
                blk = prev;
            }
        }

        self.insert_(*blk);
    }
}

void TlsfAllocator::destroy() {
    unsafe { free((void*)self.base); }
    self.cap = (unsigned long)0;
}
