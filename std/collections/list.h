#pragma once
// SafeC Standard Library — List (doubly linked list)

struct ListNode {
    void*            data;
    struct ListNode* next;
    struct ListNode* prev;
};

struct List {
    struct ListNode* head;
    struct ListNode* tail;
    unsigned long    len;
    unsigned long    elem_size;
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
struct List list_new(unsigned long elem_size);
void        list_free(struct List* l);

// ── Core operations ───────────────────────────────────────────────────────────
int           list_push_front(struct List* l, const void* elem);
int           list_push_back(struct List* l, const void* elem);
int           list_pop_front(struct List* l, void* out);
int           list_pop_back(struct List* l, void* out);
void*         list_front(struct List* l);    // NULL if empty
void*         list_back(struct List* l);     // NULL if empty
unsigned long list_len(struct List* l);
int           list_is_empty(struct List* l);
void          list_clear(struct List* l);

// ── Search & iteration ────────────────────────────────────────────────────────
// cmp: int(*)(const void*, const void*) — passed as void*
struct ListNode* list_find(struct List* l, const void* val, void* cmp);
int              list_contains(struct List* l, const void* val, void* cmp);
void             list_remove_node(struct List* l, struct ListNode* node);
int              list_remove(struct List* l, const void* val, void* cmp);  // remove first match
void             list_foreach(struct List* l, void* fn);  // fn: void(*)(void* data)

// ── Reorder ───────────────────────────────────────────────────────────────────
void list_reverse(struct List* l);

// ── Typed generic wrappers ────────────────────────────────────────────────────
generic<T>
int list_push_back_t(struct List* l, T val);

generic<T>
T* list_front_t(struct List* l);

generic<T>
T* list_back_t(struct List* l);
