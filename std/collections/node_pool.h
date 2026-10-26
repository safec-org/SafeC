#pragma once
// SafeC Standard Library — NodePool<N>: per-container node allocator
//
// Node-based containers (List, BST) take their nodes from a NodePool
// instead of calling alloc() per node. Nodes are carved from chunks that
// double in size (8, 16, ... 1024 nodes), recycled through a free list
// when removed, and released all at once by free_all(). Compared with one
// alloc() per node this drops the per-block allocator header and size-
// class padding (a 24-byte node used to occupy a 48-byte heap chunk), and
// a container's free() costs O(chunks) instead of O(nodes).
//
// A removed node's memory stays owned by the pool and is reused by the
// next take(): references into a removed element dangle exactly as they
// would with dealloc(), they just don't fault.
#include <std/mem.h>

// Free-list and chunk links live in the first word of memory that is (or
// will be) a node of type N, so they're read and written with memcpy:
// under strict aliasing a pointer-typed access there could otherwise be
// reordered against the node's own fields.
#include <std/stddef.h>
extern void* memcpy(void* dst, const void* src, size_t n);   // size_t: 32-bit on wasm32 / 32-bit targets

namespace std {

#define NODE_POOL_FIRST_CHUNK 8UL
#define NODE_POOL_MAX_CHUNK   1024UL
#define NODE_POOL_CHUNK_HDR   16UL      // keeps nodes 16-byte aligned

generic<N>
struct NodePool {
    ?&heap unsigned char chunks;      // chunk list; first word of each chunk links the next
    ?&heap unsigned char bump;        // next uncarved node in the newest chunk
    unsigned long        bump_left;   // nodes left to carve there
    ?&heap unsigned char free_list;   // recycled nodes; first word links the next
    unsigned long        next_chunk;  // node count of the next chunk

    ?&heap N take();                  // uninitialised node; null on OOM
    void     give(&heap N node);      // return a node for reuse
    void     free_all();              // release every chunk (all nodes)

    static struct NodePool<N> new();  // empty; first chunk allocated on the first take()
};

generic<N>
struct NodePool<N> NodePool::new() {
    struct NodePool<N> p;
    p.chunks = null;
    p.bump = null;
    p.bump_left = 0UL;
    p.free_list = null;
    p.next_chunk = NODE_POOL_FIRST_CHUNK;
    return p;
}

generic<N>
?&heap N NodePool::take() {
    static_assert(sizeof(N) >= sizeof(void*), "NodePool: node must hold a pointer");
    unsafe {
        if (!self.free_list.is_null()) {
            unsigned char* n = (unsigned char*)self.free_list;
            void* link = (void*)0;
            memcpy((void*)&link, (const void*)n, (size_t)sizeof(void*));
            self.free_list = (?&heap unsigned char)link;
            return (?&heap N)n;
        }
        if (self.bump_left == 0UL) {
            unsigned long count = self.next_chunk;
            unsigned long bytes = checked_add_size(NODE_POOL_CHUNK_HDR, checked_mul_size(count, sizeof(N)));
            unsigned char* c = (unsigned char*)alloc_raw_(bytes);
            if (c == (unsigned char*)0) return null;
            void* prevChunk = (void*)self.chunks;
            memcpy((void*)c, (const void*)&prevChunk, (size_t)sizeof(void*));
            self.chunks = (?&heap unsigned char)c;
            self.bump = (?&heap unsigned char)(c + NODE_POOL_CHUNK_HDR);
            self.bump_left = count;
            if (count < NODE_POOL_MAX_CHUNK) self.next_chunk = count * 2UL;
        }
        unsigned char* n = (unsigned char*)self.bump;
        self.bump = (?&heap unsigned char)(n + sizeof(N));
        self.bump_left = self.bump_left - 1UL;
        return (?&heap N)n;
    }
}

generic<N>
void NodePool::give(&heap N node) {
    unsafe {
        void* head = (void*)self.free_list;
        memcpy((void*)node, (const void*)&head, (size_t)sizeof(void*));
        self.free_list = (?&heap unsigned char)node;
    }
}

generic<N>
void NodePool::free_all() {
    unsafe {
        unsigned char* c = (unsigned char*)self.chunks;
        while (c != (unsigned char*)0) {
            void* nextRaw = (void*)0;
            memcpy((void*)&nextRaw, (const void*)c, (size_t)sizeof(void*));
            unsigned char* next = (unsigned char*)nextRaw;
            dealloc(c);
            c = next;
        }
    }
    self.chunks = null;
    self.bump = null;
    self.bump_left = 0UL;
    self.free_list = null;
    self.next_chunk = NODE_POOL_FIRST_CHUNK;
}

} // namespace std
