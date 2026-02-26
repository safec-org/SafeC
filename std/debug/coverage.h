// SafeC Standard Library — Basic Code Coverage Tracker
// Instruments execution counts for up to COV_MAX_SITES sites.
// Designed for test/fuzz integration; not a compiler plugin.
// Freestanding-safe.
#pragma once

#define COV_MAX_SITES  1024

struct CovSite {
    const char*   file;
    int           line;
    unsigned long count;
};

struct Coverage {
    struct CovSite sites[COV_MAX_SITES];
    int            count;

    // Register a new site; returns site index (for COV_HIT).
    int  register_site(const char* file, int line);

    // Increment hit count for site `idx`.
    void hit(int idx);

    // Return hit count for site `idx`.
    unsigned long get(int idx) const;

    // Print all sites with hit counts to stdout (hosted only).
    void report() const;

    // Return total number of sites with count > 0.
    int  covered_count() const;

    // Return percentage of covered sites (0-100).
    int  coverage_pct() const;

    // Reset all counts to zero.
    void reset();
};

extern struct Coverage coverage;
void coverage_init();

// Place COV_SITE() once per code region you want to track.
// Registers on first execution; subsequent calls increment the count.
// Requires a static int site_id_ guard per call site (macro handles this).
#define COV_SITE()  do {                                          \
    static int _cov_id = -1;                                      \
    if (_cov_id < 0) {                                            \
        _cov_id = coverage.register_site(__FILE__, __LINE__);     \
    }                                                             \
    if (_cov_id >= 0) { coverage.hit(_cov_id); }                  \
} while(0)
