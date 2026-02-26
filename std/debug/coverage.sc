// SafeC Standard Library — Code Coverage Tracker Implementation
#pragma once
#include "coverage.h"

extern int printf(const char* fmt, ...);

// ── Global coverage instance ──────────────────────────────────────────────────

struct Coverage coverage;

// ── coverage_init ─────────────────────────────────────────────────────────────

void coverage_init() {
    coverage.count = 0;
    int i = 0;
    while (i < COV_MAX_SITES) {
        unsafe {
            coverage.sites[i].file  = (const char*)0;
            coverage.sites[i].line  = 0;
            coverage.sites[i].count = (unsigned long)0;
        }
        i = i + 1;
    }
}

// ── Coverage::register_site ───────────────────────────────────────────────────

int Coverage::register_site(const char* file, int line) {
    if (self.count >= COV_MAX_SITES) { return -1; }
    int idx = self.count;
    unsafe {
        self.sites[idx].file  = file;
        self.sites[idx].line  = line;
        self.sites[idx].count = (unsigned long)0;
    }
    self.count = self.count + 1;
    return idx;
}

// ── Coverage::hit ─────────────────────────────────────────────────────────────

void Coverage::hit(int idx) {
    if (idx < 0 || idx >= self.count) { return; }
    unsafe {
        self.sites[idx].count = self.sites[idx].count + (unsigned long)1;
    }
}

// ── Coverage::get ─────────────────────────────────────────────────────────────

unsigned long Coverage::get(int idx) const {
    if (idx < 0 || idx >= self.count) { return (unsigned long)0; }
    unsafe { return self.sites[idx].count; }
    return (unsigned long)0;
}

// ── Coverage::report ─────────────────────────────────────────────────────────

void Coverage::report() const {
#ifndef __SAFEC_FREESTANDING__
    int total = self.count;
    int covered = self.covered_count();
    int pct = self.coverage_pct();
    unsafe {
        printf("[coverage] %d/%d sites hit (%d%%)\n", covered, total, pct);
    }
    int i = 0;
    while (i < self.count) {
        unsafe {
            const char* fname = self.sites[i].file;
            int line = self.sites[i].line;
            unsigned long cnt = self.sites[i].count;
            if (cnt > (unsigned long)0) {
                printf("  [HIT]  %s:%d  count=%lu\n", fname, line, cnt);
            } else {
                printf("  [MISS] %s:%d\n", fname, line);
            }
        }
        i = i + 1;
    }
#endif
}

// ── Coverage::covered_count ───────────────────────────────────────────────────

int Coverage::covered_count() const {
    int n = 0;
    int i = 0;
    while (i < self.count) {
        unsafe {
            if (self.sites[i].count > (unsigned long)0) { n = n + 1; }
        }
        i = i + 1;
    }
    return n;
}

// ── Coverage::coverage_pct ────────────────────────────────────────────────────

int Coverage::coverage_pct() const {
    if (self.count == 0) { return 0; }
    int covered = self.covered_count();
    // Integer percentage: (covered * 100) / total.
    return (covered * 100) / self.count;
}

// ── Coverage::reset ───────────────────────────────────────────────────────────

void Coverage::reset() {
    int i = 0;
    while (i < self.count) {
        unsafe { self.sites[i].count = (unsigned long)0; }
        i = i + 1;
    }
}
