#pragma once
// SafeC Standard Library — container capabilities as traits.
//
// std's containers spell what they have in common with the same method
// names. The traits below name those capabilities, one each, and compose
// into the roles generic code asks for (trait composition: 'trait Fifo :
// Enqueue + Dequeue {}' requires both). The generic functions after them
// are written once and work on every container with those capabilities;
// each call is resolved at compile time.
//
//   Len        length() / is_empty()
//   Clear      clear()
//   Get        get(i) -> ?&T, null when absent (maps: get(key))
//   Push       push(x): 1, or 0 when full or out of memory
//   Pop        pop() -> ?T, null when empty
//   Enqueue    enqueue(x): 1, or 0 when full (Chan: blocks; 0 once closed)
//   Dequeue    dequeue() -> ?T, null when empty (Chan: blocks; null once
//              closed and drained)
//   Insert, Contains, Remove   keyed: insert(k, v), contains(k), remove(k) -> ?V
//
//   Sequence     Len + Get + Push: Vec, StaticVec
//   Lifo         Len + Push + Pop: Vec, Stack, StaticVec
//   Fifo         Enqueue + Dequeue: Queue, MpscQueue, LFQueue, Chan,
//                RingBuffer (one byte at a time)
//   CountedFifo  Fifo + Len: Queue, MpscQueue, LFQueue, RingBuffer
//   Map          Len + Insert + Get + Contains + Remove: HashMap, StrMap,
//                BST, BTree (StaticMap has no remove: Len + Insert + Get +
//                Contains)
//
// Len and Clear alone also cover String, List, Slice (Len) and StaticMap.
//
//   struct Queue<int> q = std::Queue::new();
//   int xs[3] = {1, 2, 3};
//   enqueue_all(&q, &xs[0], 3UL);            // the same call on a Chan<int>,
//   struct Vec<int> got = std::Vec::new();   // an LFQueue<int>, a MpscQueue<int>
//   drain_into(&q, &got);                    // Fifo -> Sequence
//
// A trait checks each method's name, static-ness and parameter count; the
// parameter types are checked where a generic body is instantiated. So
// Vec's contains(key, cmp) keeps Vec out of Contains and Map, while
// String's push(cstr) still counts as Push (push_all over a String fails
// where it passes a char).
//
// Not a trait: iteration. foreach exists on most containers but its
// callback shape differs per container (Vec: (&T, index), List: (&T),
// maps: (const &K, &V)), so there is no one protocol to name.

#include <std/cmp.h>

namespace std {

// Elem_ (cmp.h), Key_ and Val_ stand for the element, key and value types
// in the signatures below; each container has its own.
typedef void* Key_;
typedef void* Val_;

trait Len {
    unsigned long length() const;
    int is_empty() const;
}

trait Clear {
    void clear();
}

trait Get {
    ?&Elem_ get(unsigned long idx) const;   // null when out of range (maps: by key)
}

trait Push {
    int push(Elem_ value);   // 1, or 0 when full / out of memory
}

trait Pop {
    ?Elem_ pop();            // null when empty
}

trait Enqueue {
    int enqueue(Elem_ value);   // 1, or 0 when full (or closed)
}

trait Dequeue {
    ?Elem_ dequeue();           // null when empty (or closed and drained)
}

trait Insert {
    int insert(Key_ key, Val_ val);   // insert or overwrite; 0 when full / out of memory
}

trait Contains {
    int contains(Key_ key) const;
}

trait Remove {
    ?Val_ remove(Key_ key);           // the removed value, or null
}

trait Sequence : Len + Get + Push {}

trait Lifo : Len + Push + Pop {}

trait Fifo : Enqueue + Dequeue {}

trait CountedFifo : Fifo + Len {}

trait Map : Len + Insert + Get + Contains + Remove {}

// Pushes items[0..n) in order, stopping at the first one refused (full or
// out of memory). Returns how many went in; the rest stay in items.
generic<S: Push, T>
unsigned long push_all(&S s, const &T items, unsigned long n) {
    unsigned long i = 0UL;
    unsafe { while (i < n && s.push(items[i])) { i = i + 1UL; } }
    return i;
}

// Enqueues items[0..n) in order, stopping at the first one refused (full or
// closed). Returns how many went in; the rest stay in items.
generic<Q: Enqueue, T>
unsigned long enqueue_all(&Q q, const &T items, unsigned long n) {
    unsigned long i = 0UL;
    unsafe { while (i < n && q.enqueue(items[i])) { i = i + 1UL; } }
    return i;
}

// Dequeues until q is empty (a Chan: until it is closed and drained) and
// pushes each element onto out, in order. Returns how many moved. Meant for
// a growable out (Vec): an element out refuses is dropped and draining
// stops there.
generic<Q: Dequeue, S: Push>
unsigned long drain_into(&Q q, &S out) {
    unsigned long n = 0UL;
    while (1) {
        match (q.dequeue()) {
            case none: return n;
            case some(x): {
                if (!out.push(x)) return n;
                n = n + 1UL;
            }
        }
    }
    return n;
}

} // namespace std
