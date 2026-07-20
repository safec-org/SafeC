#pragma once
// SafeC Standard Library — List (doubly linked list)

namespace std {

struct ListNode {
    void*            data;
    ?&heap ListNode  next;
    ?&heap ListNode  prev;
};

struct List {
    ?&heap ListNode  head;
    ?&heap ListNode  tail;
    unsigned long    len;
    unsigned long    elem_size;
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct List list_new(unsigned long elem_size);
void        list_free(&List l);

// ── Core operations ───────────────────────────────────────────────────────────
int           list_push_front(&List l, const void* elem);
int           list_push_back(&List l, const void* elem);
int           list_pop_front(&List l, void* out);
int           list_pop_back(&List l, void* out);
void*         list_front(&List l);    // NULL if empty
void*         list_back(&List l);     // NULL if empty
unsigned long list_len(&List l);
int           list_is_empty(&List l);
void          list_clear(&List l);

// ── Search & iteration ────────────────────────────────────────────────────────
// cmp: int(*)(const void*, const void*) — passed as void*
?&heap ListNode list_find(&List l, const void* val, void* cmp);
int             list_contains(&List l, const void* val, void* cmp);
void            list_remove_node(&List l, &heap ListNode node);
int             list_remove(&List l, const void* val, void* cmp);  // remove first match
void            list_foreach(&List l, void* func);  // func: void(*)(void* data)

// ── Reorder ───────────────────────────────────────────────────────────────────
void list_reverse(&List l);

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int list_push_front_t(&List l, T val);

generic<T>
int list_push_back_t(&List l, T val);

generic<T>
T* list_front_t(&List l);

generic<T>
T* list_back_t(&List l);

} // namespace std
