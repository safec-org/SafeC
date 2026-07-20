// SafeC Standard Library — List (doubly linked list) implementation
#pragma once
#include <std/collections/list.h>
#include <std/mem.h>

namespace std {

inline struct List list_new(unsigned long elem_size) {
    struct List l;
    unsafe {
        l.head = (struct ListNode*)0;
        l.tail = (struct ListNode*)0;
    }
    l.len = 0UL;
    l.elem_size = elem_size;
    return l;
}

// Allocate a new node with a copy of elem. The node and its data used to be
// two separate alloc()s -- every push/pop paid std::alloc's 16-byte header
// and quarantine-ring bookkeeping twice per element instead of once. Since
// elem_size is fixed for the list's lifetime, the data bytes are stored
// inline right after the node in one allocation instead (same
// alloc()/dealloc() pairing and double-free/UAF tracking on the node, just
// nothing separate left to double-free for data).
inline struct ListNode* list_alloc_node_(unsigned long elem_size, const void* elem) {
    unsafe {
        unsigned long total = checked_add_size(sizeof(struct ListNode), elem_size);
        struct ListNode* n = (struct ListNode*)alloc(total);
        if (n == (struct ListNode*)0) return (struct ListNode*)0;
        n->data = (void*)((char*)n + sizeof(struct ListNode));
        safe_memcpy(n->data, elem, elem_size);
        n->next = (struct ListNode*)0;
        n->prev = (struct ListNode*)0;
        return n;
    }
}

inline void list_free_node_(struct ListNode* n) {
    unsafe {
        dealloc((void*)n);
    }
}

inline void list_free(&List l) {
    unsafe {
        struct ListNode* n = l.head;
        while (n != (struct ListNode*)0) {
            struct ListNode* next = n->next;
            list_free_node_(n);
            n = next;
        }
        l.head = (struct ListNode*)0;
        l.tail = (struct ListNode*)0;
    }
    l.len = 0UL;
}

unsigned long list_len(&List l)      { return l.len; }
int           list_is_empty(&List l) { return l.len == 0UL; }
void*         list_front(&List l)    { unsafe { struct ListNode* n = l.head; return n != (struct ListNode*)0 ? n->data : (void*)0; } }
void*         list_back(&List l)     { unsafe { struct ListNode* n = l.tail; return n != (struct ListNode*)0 ? n->data : (void*)0; } }

void list_clear(&List l) { list_free(l); }

inline int list_push_front(&List l, const void* elem) {
    unsafe {
        struct ListNode* n = list_alloc_node_(l.elem_size, elem);
        if (n == (struct ListNode*)0) return 0;
        n->next = l.head;
        n->prev = (struct ListNode*)0;
        struct ListNode* head = l.head;
        if (head != (struct ListNode*)0) head->prev = n;
        else l.tail = n;
        l.head = n;
    }
    l.len = l.len + 1UL;
    return 1;
}

inline int list_push_back(&List l, const void* elem) {
    unsafe {
        struct ListNode* n = list_alloc_node_(l.elem_size, elem);
        if (n == (struct ListNode*)0) return 0;
        n->prev = l.tail;
        n->next = (struct ListNode*)0;
        struct ListNode* tail = l.tail;
        if (tail != (struct ListNode*)0) tail->next = n;
        else l.head = n;
        l.tail = n;
    }
    l.len = l.len + 1UL;
    return 1;
}

inline int list_pop_front(&List l, void* out) {
    unsafe {
        struct ListNode* n = l.head;
        if (n == (struct ListNode*)0) return 0;
        if (out != (void*)0) safe_memcpy(out, (const void*)n->data, l.elem_size);
        l.head = n->next;
        struct ListNode* newHead = l.head;
        if (newHead != (struct ListNode*)0) newHead->prev = (struct ListNode*)0;
        else l.tail = (struct ListNode*)0;
        list_free_node_(n);
    }
    l.len = l.len - 1UL;
    return 1;
}

inline int list_pop_back(&List l, void* out) {
    unsafe {
        struct ListNode* n = l.tail;
        if (n == (struct ListNode*)0) return 0;
        if (out != (void*)0) safe_memcpy(out, (const void*)n->data, l.elem_size);
        l.tail = n->prev;
        struct ListNode* newTail = l.tail;
        if (newTail != (struct ListNode*)0) newTail->next = (struct ListNode*)0;
        else l.head = (struct ListNode*)0;
        list_free_node_(n);
    }
    l.len = l.len - 1UL;
    return 1;
}

?&heap ListNode list_find(&List l, const void* val, void* cmp) {
    unsafe {
        fn int(const void*, const void*) cmpfn = (fn int(const void*, const void*))cmp;
        struct ListNode* n = l.head;
        while (n != (struct ListNode*)0) {
            if (cmpfn((const void*)n->data, val) == 0) return (?&heap ListNode)n;
            n = n->next;
        }
        return (?&heap ListNode)(struct ListNode*)0;
    }
}

inline int list_contains(&List l, const void* val, void* cmp) {
    unsafe { return list_find(l, val, cmp) != (?&heap ListNode)(struct ListNode*)0; }
}

inline void list_remove_node(&List l, &heap ListNode node) {
    unsafe {
        struct ListNode* n = node;
        struct ListNode* prev = n->prev;
        struct ListNode* next = n->next;
        if (prev != (struct ListNode*)0) prev->next = next;
        else l.head = next;
        if (next != (struct ListNode*)0) next->prev = prev;
        else l.tail = prev;
        list_free_node_(n);
    }
    l.len = l.len - 1UL;
}

inline int list_remove(&List l, const void* val, void* cmp) {
    unsafe {
        ?&heap ListNode n = list_find(l, val, cmp);
        if (n == (?&heap ListNode)(struct ListNode*)0) return 0;
        list_remove_node(l, (&heap ListNode)n);
    }
    return 1;
}

inline void list_foreach(&List l, void* func) {
    unsafe {
        fn void(void*) f = (fn void(void*))func;
        struct ListNode* n = l.head;
        while (n != (struct ListNode*)0) {
            f(n->data);
            n = n->next;
        }
    }
}

inline void list_reverse(&List l) {
    unsafe {
        struct ListNode* n = l.head;
        l.tail = l.head;
        while (n != (struct ListNode*)0) {
            struct ListNode* tmp = n->next;
            n->next = n->prev;
            n->prev = tmp;
            if (tmp == (struct ListNode*)0) l.head = n;
            n = tmp;
        }
    }
}

generic<T>
int list_push_front_t(&List l, T val) {
    unsafe { return list_push_front(l, (const void*)&val); }
}

generic<T>
int list_push_back_t(&List l, T val) {
    unsafe { return list_push_back(l, (const void*)&val); }
}

generic<T>
T* list_front_t(&List l) {
    return (T*)list_front(l);
}

generic<T>
T* list_back_t(&List l) {
    return (T*)list_back(l);
}

} // namespace std
