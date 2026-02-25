// SafeC Standard Library — Memory declarations
#pragma once

void*         alloc(unsigned long size);
void*         alloc_zeroed(unsigned long size);
void          dealloc(void* ptr);
void*         realloc_buf(void* ptr, unsigned long new_size);
void          safe_memcpy(void* dst, const void* src, unsigned long n);
void          safe_memmove(void* dst, const void* src, unsigned long n);
void          safe_memset(void* ptr, int val, unsigned long n);
int           safe_memcmp(const void* a, const void* b, unsigned long n);
