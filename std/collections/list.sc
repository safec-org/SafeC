// SafeC Standard Library — List (doubly linked list) implementation
#include "list.h"
#include "../mem.h"

struct List list_new(unsigned long elem_size) {
    struct List l;
    l.head = (struct ListNode*)0;
    l.tail = (struct ListNode*)0;
    l.len = 0UL;
    l.elem_size = elem_size;
    return l;
}

// Allocate a new node with a copy of elem
struct ListNode* list_alloc_node_(unsigned long elem_size, const void* elem) {
    unsafe {
        struct ListNode* n = (struct ListNode*)alloc(sizeof(struct ListNode));
        if (n == (struct ListNode*)0) return (struct ListNode*)0;
        n->data = alloc(elem_size);
        if (n->data == (void*)0) { dealloc((void*)n); return (struct ListNode*)0; }
        safe_memcpy(n->data, elem, elem_size);
        n->next = (struct ListNode*)0;
        n->prev = (struct ListNode*)0;
        return n;
    }
}

void list_free_node_(struct ListNode* n) {
    unsafe {
        if (n->data != (void*)0) dealloc(n->data);
        dealloc((void*)n);
    }
}

void list_free(struct List* l) {
    struct ListNode* n = l->head;
    while (n != (struct ListNode*)0) {
        struct ListNode* next = n->next;
        list_free_node_(n);
        n = next;
    }
    l->head = (struct ListNode*)0;
    l->tail = (struct ListNode*)0;
    l->len = 0UL;
}

unsigned long list_len(struct List* l)      { return l->len; }
int           list_is_empty(struct List* l) { return l->len == 0UL; }
void*         list_front(struct List* l)    { return l->head != (struct ListNode*)0 ? l->head->data : (void*)0; }
void*         list_back(struct List* l)     { return l->tail != (struct ListNode*)0 ? l->tail->data : (void*)0; }

void list_clear(struct List* l) { list_free(l); }

int list_push_front(struct List* l, const void* elem) {
    struct ListNode* n = list_alloc_node_(l->elem_size, elem);
    if (n == (struct ListNode*)0) return 0;
    n->next = l->head;
    n->prev = (struct ListNode*)0;
    if (l->head != (struct ListNode*)0) l->head->prev = n;
    else l->tail = n;
    l->head = n;
    l->len = l->len + 1UL;
    return 1;
}

int list_push_back(struct List* l, const void* elem) {
    struct ListNode* n = list_alloc_node_(l->elem_size, elem);
    if (n == (struct ListNode*)0) return 0;
    n->prev = l->tail;
    n->next = (struct ListNode*)0;
    if (l->tail != (struct ListNode*)0) l->tail->next = n;
    else l->head = n;
    l->tail = n;
    l->len = l->len + 1UL;
    return 1;
}

int list_pop_front(struct List* l, void* out) {
    if (l->head == (struct ListNode*)0) return 0;
    struct ListNode* n = l->head;
    if (out != (void*)0) unsafe { safe_memcpy(out, (const void*)n->data, l->elem_size); }
    l->head = n->next;
    if (l->head != (struct ListNode*)0) l->head->prev = (struct ListNode*)0;
    else l->tail = (struct ListNode*)0;
    list_free_node_(n);
    l->len = l->len - 1UL;
    return 1;
}

int list_pop_back(struct List* l, void* out) {
    if (l->tail == (struct ListNode*)0) return 0;
    struct ListNode* n = l->tail;
    if (out != (void*)0) unsafe { safe_memcpy(out, (const void*)n->data, l->elem_size); }
    l->tail = n->prev;
    if (l->tail != (struct ListNode*)0) l->tail->next = (struct ListNode*)0;
    else l->head = (struct ListNode*)0;
    list_free_node_(n);
    l->len = l->len - 1UL;
    return 1;
}

struct ListNode* list_find(struct List* l, const void* val, void* cmp) {
    unsafe {
        int (*cmpfn)(const void*, const void*) = (int (*)(const void*, const void*))cmp;
        struct ListNode* n = l->head;
        while (n != (struct ListNode*)0) {
            if (cmpfn((const void*)n->data, val) == 0) return n;
            n = n->next;
        }
    }
    return (struct ListNode*)0;
}

int list_contains(struct List* l, const void* val, void* cmp) {
    return list_find(l, val, cmp) != (struct ListNode*)0;
}

void list_remove_node(struct List* l, struct ListNode* node) {
    if (node->prev != (struct ListNode*)0) node->prev->next = node->next;
    else l->head = node->next;
    if (node->next != (struct ListNode*)0) node->next->prev = node->prev;
    else l->tail = node->prev;
    list_free_node_(node);
    l->len = l->len - 1UL;
}

int list_remove(struct List* l, const void* val, void* cmp) {
    struct ListNode* n = list_find(l, val, cmp);
    if (n == (struct ListNode*)0) return 0;
    list_remove_node(l, n);
    return 1;
}

void list_foreach(struct List* l, void* fn) {
    unsafe {
        void (*f)(void*) = (void (*)(void*))fn;
        struct ListNode* n = l->head;
        while (n != (struct ListNode*)0) {
            f(n->data);
            n = n->next;
        }
    }
}

void list_reverse(struct List* l) {
    struct ListNode* n = l->head;
    l->tail = l->head;
    while (n != (struct ListNode*)0) {
        struct ListNode* tmp = n->next;
        n->next = n->prev;
        n->prev = tmp;
        if (tmp == (struct ListNode*)0) l->head = n;
        n = tmp;
    }
}

generic<T>
int list_push_back_t(struct List* l, T val) {
    return list_push_back(l, (const void*)&val);
}

generic<T>
T* list_front_t(struct List* l) {
    return (T*)list_front(l);
}

generic<T>
T* list_back_t(struct List* l) {
    return (T*)list_back(l);
}
