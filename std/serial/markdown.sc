// SafeC Standard Library — Markdown (see markdown.h for the syntax covered).
//
// Two passes, as in CommonMark: a line-based block parser builds the
// block tree (recursing into the de-indented contents of block quotes,
// list items and containers), then each leaf block's text goes through
// the inline parser — a token list plus CommonMark's delimiter algorithm
// for emphasis, so '*' / '_' / '~~' runs pair up by the flanking rules
// (no intraword '_', '**a*' leaves a literal '*').
#pragma once
#include <std/serial/markdown.h>
#include <std/serial/html.h>
#include <std/mem.h>
#include <std/str.h>
#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

// ── options ──────────────────────────────────────────────────────────────────

struct MdOptions MdOptions::commonmark() {
    struct MdOptions o;
    o.tables = 0; o.html = 0; o.containers = 0;
    o.heading_ids = 0; o.strikethrough = 0; o.autolinks = 0;
    return o;
}

struct MdOptions MdOptions::gfm() {
    struct MdOptions o = MdOptions::commonmark();
    o.tables = 1; o.strikethrough = 1; o.autolinks = 1;
    return o;
}

// ── nodes ────────────────────────────────────────────────────────────────────

static struct MdNode* md_new_(int kind) {
    struct MdNode* n;
    unsafe {
        n = (struct MdNode*)alloc_raw_((unsigned long)sizeof(struct MdNode));
        n->kind = kind;
        n->level = 0;
        n->ordered = 0;
        n->start = 1;
        n->tight = 1;
        n->align = MD_ALIGN_NONE;
        n->header = 0;
        n->text = String::new();
        n->info = String::new();
        n->id = String::new();
        n->title = String::new();
        n->children = Vec::new();
    }
    return n;
}

static void md_add_(struct MdNode* parent, struct MdNode* child) {
    unsafe { parent->children.push(child); }
}

// Appends an inline node, merging adjacent text nodes.
static void md_add_inline_(struct MdNode* parent, struct MdNode* child) {
    unsafe {
        unsigned long n = parent->children.length();
        if (child->kind == MD_TEXT && n > 0UL) {
            struct MdNode* last = *parent->children.at(n - 1UL);
            if (last->kind == MD_TEXT) {
                last->text.push_str(&child->text);
                child->text.free();
                child->children.free();
                child->info.free(); child->id.free(); child->title.free();
                dealloc_raw_((void*)child);
                return;
            }
        }
        parent->children.push(child);
    }
}

void MdNode::free() {
    unsigned long cn = self.children.length();
    for (unsigned long ci = 0UL; ci < cn; ci = ci + 1UL) {
        &heap MdNode child;
        unsafe { child = *self.children.at(ci); }
        child.free();
    }
    self.children.free();
    self.text.free();
    self.info.free();
    self.id.free();
    self.title.free();
    unsafe { dealloc_raw_((void*)self); }
}

// ── character classes ────────────────────────────────────────────────────────

static int md_is_space_(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

// ASCII punctuation (CommonMark's definition).
static int md_is_punct_(char c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

static int md_is_alnum_(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

static unsigned long md_strlen_(const char* s) {
    unsigned long n = 0UL;
    unsafe { while (s[n] != '\0') { n = n + 1UL; } }
    return n;
}

// ── parser state ─────────────────────────────────────────────────────────────

struct MdParser_ {
    struct MdOptions opts;
    struct Vec<struct String> ids;   // heading ids handed out (heading_ids)
};

static void md_parse_inlines_(struct MdParser_* st, struct MdNode* parent, const char* s, unsigned long n);

// ── inline parsing ───────────────────────────────────────────────────────────

#define MD_TOK_TEXT  0
#define MD_TOK_NODE  1
#define MD_TOK_DELIM 2

struct MdTok_ {
    int kind;
    struct String text;     // TEXT, and a DELIM's remaining run
    struct MdNode* node;    // NODE
    char dch;               // DELIM: '*', '_' or '~'
    int count;              // DELIM: characters left in the run
    int orig;               // DELIM: original run length
    int can_open;
    int can_close;
    int prev;
    int next;
};

struct MdInline_ {
    struct Vec<struct MdTok_> toks;
    int head;
    int tail;
};

static struct MdTok_* md_tok_(struct MdInline_* il, int i) {
    unsafe { return il->toks.at((unsigned long)i); }
}

static int md_push_tok_(struct MdInline_* il, struct MdTok_ t) {
    unsafe {
        t.prev = il->tail;
        t.next = -1;
        int idx = (int)il->toks.length();
        il->toks.push(t);
        if (il->tail >= 0) { md_tok_(il, il->tail)->next = idx; } else { il->head = idx; }
        il->tail = idx;
        return idx;
    }
}

static void md_push_node_(struct MdInline_* il, struct MdNode* node) {
    struct MdTok_ t;
    t.kind = MD_TOK_NODE; t.text = String::new(); t.node = node;
    t.dch = '\0'; t.count = 0; t.orig = 0; t.can_open = 0; t.can_close = 0;
    md_push_tok_(il, t);
}

static void md_flush_text_(struct MdInline_* il, struct String* buf) {
    unsafe {
        if (buf->len == 0UL) { return; }
        struct MdTok_ t;
        t.kind = MD_TOK_TEXT; t.text = buf->clone(); t.node = (struct MdNode*)0;
        t.dch = '\0'; t.count = 0; t.orig = 0; t.can_open = 0; t.can_close = 0;
        md_push_tok_(il, t);
        buf->clear();
    }
}

static void md_unlink_tok_(struct MdInline_* il, int i) {
    unsafe {
        struct MdTok_* t = md_tok_(il, i);
        if (t->prev >= 0) { md_tok_(il, t->prev)->next = t->next; } else { il->head = t->next; }
        if (t->next >= 0) { md_tok_(il, t->next)->prev = t->prev; } else { il->tail = t->prev; }
        t->prev = -1; t->next = -1;
    }
}

// The node a token stands for (text and delimiter leftovers become MD_TEXT).
static struct MdNode* md_tok_node_(struct MdTok_* t) {
    unsafe {
        if (t->kind == MD_TOK_NODE) { struct MdNode* n = t->node; t->node = (struct MdNode*)0; return n; }
        struct MdNode* n = md_new_(MD_TEXT);
        n->text.push_str(&t->text);
        return n;
    }
}

// CommonMark's "process emphasis" over the whole token list.
static void md_process_emphasis_(struct MdInline_* il) {
    unsafe {
        int c = il->head;
        while (c >= 0) {
            struct MdTok_* ct = md_tok_(il, c);
            if (ct->kind != MD_TOK_DELIM || !ct->can_close || ct->count == 0) { c = ct->next; continue; }
            int o = ct->prev;
            int found = -1;
            while (o >= 0) {
                struct MdTok_* ot = md_tok_(il, o);
                if (ot->kind == MD_TOK_DELIM && ot->dch == ct->dch && ot->can_open && ot->count > 0) {
                    if (ct->dch == '~') {
                        if (ot->count == ct->count) { found = o; break; }
                    } else {
                        int odd = (ot->can_close || ct->can_open) && (ot->orig + ct->orig) % 3 == 0 &&
                                  !(ot->orig % 3 == 0 && ct->orig % 3 == 0);
                        if (!odd) { found = o; break; }
                    }
                }
                o = ot->prev;
            }
            if (found < 0) {
                int nx = ct->next;
                if (!ct->can_open) { ct->kind = MD_TOK_TEXT; }   // can never match: plain text
                c = nx;
                continue;
            }
            struct MdTok_* ot = md_tok_(il, found);
            int use = 1;
            int kind = MD_ITALIC;
            if (ct->dch == '~') { use = ct->count; kind = MD_STRIKE; }
            else if (ot->count >= 2 && ct->count >= 2) { use = 2; kind = MD_BOLD; }
            struct MdNode* em = md_new_(kind);
            int t = ot->next;
            while (t >= 0 && t != c) {
                int nx = md_tok_(il, t)->next;
                md_add_inline_(em, md_tok_node_(md_tok_(il, t)));
                md_unlink_tok_(il, t);
                t = nx;
            }
            // Insert the emphasis node between opener and closer.
            struct MdTok_ et;
            et.kind = MD_TOK_NODE; et.text = String::new(); et.node = em;
            et.dch = '\0'; et.count = 0; et.orig = 0; et.can_open = 0; et.can_close = 0;
            et.prev = found; et.next = c;
            int ei = (int)il->toks.length();
            il->toks.push(et);
            ot = md_tok_(il, found);
            ct = md_tok_(il, c);
            ot->next = ei;
            ct->prev = ei;
            ot->count = ot->count - use;
            ct->count = ct->count - use;
            ot->text.truncate((unsigned long)ot->count);
            ct->text.truncate((unsigned long)ct->count);
            if (ot->count == 0) { md_unlink_tok_(il, found); }
            if (ct->count == 0) {
                int nx = ct->next;
                md_unlink_tok_(il, c);
                c = nx;
            }
        }
    }
}

// Matches an inline HTML construct at s[i] ('<' ...): an open tag with
// attributes, a closing tag, a comment, a processing instruction or a
// declaration. Returns the index just past it, or 0.
static unsigned long md_match_html_tag_(const char* s, unsigned long i, unsigned long n) {
    unsafe {
        unsigned long k = i + 1UL;
        if (k >= n) { return 0UL; }
        if (s[k] == '!' && k + 2UL < n && s[k + 1UL] == '-' && s[k + 2UL] == '-') {
            unsigned long e = k + 3UL;
            while (e + 2UL < n + 0UL && !(s[e] == '-' && s[e + 1UL] == '-' && s[e + 2UL] == '>')) { e = e + 1UL; }
            return e + 2UL < n ? e + 3UL : 0UL;
        }
        if (s[k] == '?' || s[k] == '!') {
            unsigned long e = k + 1UL;
            while (e < n && s[e] != '>') { e = e + 1UL; }
            return e < n ? e + 1UL : 0UL;
        }
        int closing = 0;
        if (s[k] == '/') { closing = 1; k = k + 1UL; }
        if (k >= n || !((s[k] >= 'a' && s[k] <= 'z') || (s[k] >= 'A' && s[k] <= 'Z'))) { return 0UL; }
        while (k < n && (md_is_alnum_(s[k]) || s[k] == '-')) { k = k + 1UL; }
        if (closing) {
            while (k < n && md_is_space_(s[k])) { k = k + 1UL; }
            return (k < n && s[k] == '>') ? k + 1UL : 0UL;
        }
        // Attributes.
        for (;;) {
            unsigned long ws = k;
            while (k < n && md_is_space_(s[k])) { k = k + 1UL; }
            if (k >= n) { return 0UL; }
            if (s[k] == '>') { return k + 1UL; }
            if (s[k] == '/' && k + 1UL < n && s[k + 1UL] == '>') { return k + 2UL; }
            if (k == ws) { return 0UL; }   // attributes need leading whitespace
            if (!((s[k] >= 'a' && s[k] <= 'z') || (s[k] >= 'A' && s[k] <= 'Z') || s[k] == '_' || s[k] == ':')) { return 0UL; }
            while (k < n && (md_is_alnum_(s[k]) || s[k] == '_' || s[k] == '.' || s[k] == ':' || s[k] == '-')) { k = k + 1UL; }
            unsigned long save = k;
            while (k < n && md_is_space_(s[k])) { k = k + 1UL; }
            if (k < n && s[k] == '=') {
                k = k + 1UL;
                while (k < n && md_is_space_(s[k])) { k = k + 1UL; }
                if (k >= n) { return 0UL; }
                if (s[k] == '"' || s[k] == '\'') {
                    char q = s[k];
                    k = k + 1UL;
                    while (k < n && s[k] != q) { k = k + 1UL; }
                    if (k >= n) { return 0UL; }
                    k = k + 1UL;
                } else {
                    unsigned long vs = k;
                    while (k < n && !md_is_space_(s[k]) && s[k] != '"' && s[k] != '\'' && s[k] != '=' &&
                           s[k] != '<' && s[k] != '>' && s[k] != '`') { k = k + 1UL; }
                    if (k == vs) { return 0UL; }
                }
            } else {
                k = save;
            }
        }
    }
    return 0UL;
}

// Decodes the character reference at s[i] ('&...;') into 'out'; returns
// the index past it, or 0 if it isn't one.
static unsigned long md_entity_(const char* s, unsigned long i, unsigned long n, struct String* out) {
    unsafe {
        unsigned long k = i + 1UL;
        while (k < n && k - i <= 32UL && (md_is_alnum_(s[k]) || s[k] == '#')) { k = k + 1UL; }
        if (k >= n || s[k] != ';' || k == i + 1UL) { return 0UL; }
        struct String ref = String::from_n(s + i, k + 1UL - i);
        struct String dec = String::new();
        html_unescape(&dec, ref.as_ptr());
        int ok = !dec.eq(&ref);
        if (ok) { out->push_str(&dec); }
        ref.free(); dec.free();
        return ok ? k + 1UL : 0UL;
    }
}

// Unescapes a link destination/title: backslash escapes and references.
static struct String md_unescape_(const char* s, unsigned long n) {
    struct String out = String::new();
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            char c = s[i];
            if (c == '\\' && i + 1UL < n && md_is_punct_(s[i + 1UL])) { out.push_char(s[i + 1UL]); i = i + 2UL; continue; }
            if (c == '&') {
                unsigned long e = md_entity_(s, i, n, &out);
                if (e > 0UL) { i = e; continue; }
            }
            out.push_char(c);
            i = i + 1UL;
        }
    }
    return out;
}

// From s[i] == '[': the index of the matching ']' (nested brackets,
// code spans and escapes respected), or 0.
static unsigned long md_match_bracket_(const char* s, unsigned long i, unsigned long n) {
    unsafe {
        int depth = 0;
        unsigned long k = i;
        while (k < n) {
            char c = s[k];
            if (c == '\\' && k + 1UL < n) { k = k + 2UL; continue; }
            if (c == '`') {
                unsigned long run = 0UL;
                while (k + run < n && s[k + run] == '`') { run = run + 1UL; }
                unsigned long j = k + run;
                unsigned long close = 0UL;
                while (j < n) {
                    if (s[j] == '`') {
                        unsigned long r2 = 0UL;
                        while (j + r2 < n && s[j + r2] == '`') { r2 = r2 + 1UL; }
                        if (r2 == run) { close = j + r2; break; }
                        j = j + r2;
                    } else { j = j + 1UL; }
                }
                k = close > 0UL ? close : k + run;
                continue;
            }
            if (c == '[') { depth = depth + 1; }
            else if (c == ']') {
                depth = depth - 1;
                if (depth == 0) { return k; }
            }
            k = k + 1UL;
        }
    }
    return 0UL;
}

// Parses '(dest "title")' at s[i] == '('. On success fills dest/title and
// returns the index past ')', else 0.
static unsigned long md_link_tail_(const char* s, unsigned long i, unsigned long n, struct String* dest, struct String* title) {
    unsafe {
        unsigned long k = i + 1UL;
        while (k < n && md_is_space_(s[k])) { k = k + 1UL; }
        unsigned long ds = k;
        unsigned long de = k;
        if (k < n && s[k] == '<') {
            k = k + 1UL;
            ds = k;
            while (k < n && s[k] != '>' && s[k] != '\n') {
                if (s[k] == '\\' && k + 1UL < n) { k = k + 1UL; }
                k = k + 1UL;
            }
            if (k >= n || s[k] != '>') { return 0UL; }
            de = k;
            k = k + 1UL;
        } else {
            int parens = 0;
            while (k < n && !md_is_space_(s[k])) {
                if (s[k] == '\\' && k + 1UL < n) { k = k + 2UL; continue; }
                if (s[k] == '(') { parens = parens + 1; }
                else if (s[k] == ')') {
                    if (parens == 0) { break; }
                    parens = parens - 1;
                }
                k = k + 1UL;
            }
            de = k;
        }
        struct String d = md_unescape_(s + ds, de - ds);
        while (k < n && md_is_space_(s[k])) { k = k + 1UL; }
        struct String t = String::new();
        if (k < n && (s[k] == '"' || s[k] == '\'' || s[k] == '(')) {
            char q = s[k] == '(' ? ')' : s[k];
            unsigned long ts = k + 1UL;
            unsigned long te = ts;
            while (te < n && s[te] != q) {
                if (s[te] == '\\' && te + 1UL < n) { te = te + 1UL; }
                te = te + 1UL;
            }
            if (te >= n) { d.free(); t.free(); return 0UL; }
            t.free();
            t = md_unescape_(s + ts, te - ts);
            k = te + 1UL;
            while (k < n && md_is_space_(s[k])) { k = k + 1UL; }
        }
        if (k >= n || s[k] != ')') { d.free(); t.free(); return 0UL; }
        dest->push_str(&d);
        title->push_str(&t);
        d.free(); t.free();
        return k + 1UL;
    }
}

// <scheme:...> or <user@host> at s[i] == '<': the index past '>', or 0;
// 'href' gets the target ('mailto:' added for an email).
static unsigned long md_autolink_(const char* s, unsigned long i, unsigned long n, struct String* href, struct String* label) {
    unsafe {
        unsigned long k = i + 1UL;
        unsigned long st = k;
        // URI: scheme of 2-32 chars, ':', then no spaces / '<' / '>'.
        while (k < n && (md_is_alnum_(s[k]) || s[k] == '+' || s[k] == '.' || s[k] == '-')) { k = k + 1UL; }
        if (k < n && s[k] == ':' && k - st >= 2UL && k - st <= 32UL &&
            ((s[st] >= 'a' && s[st] <= 'z') || (s[st] >= 'A' && s[st] <= 'Z'))) {
            while (k < n && s[k] != '>' && s[k] != '<' && !md_is_space_(s[k])) { k = k + 1UL; }
            if (k < n && s[k] == '>') {
                href->push_n(s + st, k - st);
                label->push_n(s + st, k - st);
                return k + 1UL;
            }
            return 0UL;
        }
        // Email.
        k = st;
        int at = 0;
        while (k < n && s[k] != '>' && !md_is_space_(s[k]) && s[k] != '<') {
            if (s[k] == '@') { at = at + 1; }
            k = k + 1UL;
        }
        if (k < n && s[k] == '>' && at == 1 && s[st] != '@' && s[k - 1UL] != '@') {
            href->push("mailto:");
            href->push_n(s + st, k - st);
            label->push_n(s + st, k - st);
            return k + 1UL;
        }
    }
    return 0UL;
}

// GFM bare URL at s[i] (autolinks option): the index past it, or 0.
static unsigned long md_bare_url_(const char* s, unsigned long i, unsigned long n, struct String* href, struct String* label) {
    unsafe {
        int www = 0;
        if (n - i > 8UL && str_ncmp(s + i, "https://", 8UL) == 0) { }
        else if (n - i > 7UL && str_ncmp(s + i, "http://", 7UL) == 0) { }
        else if (n - i > 4UL && str_ncmp(s + i, "www.", 4UL) == 0) { www = 1; }
        else { return 0UL; }
        unsigned long k = i;
        while (k < n && !md_is_space_(s[k]) && s[k] != '<') { k = k + 1UL; }
        // Trailing punctuation isn't part of the URL; nor is an unbalanced ')'.
        for (;;) {
            if (k <= i) { break; }
            char c = s[k - 1UL];
            if (c == '.' || c == ',' || c == ':' || c == ';' || c == '!' || c == '?' || c == '"' ||
                c == '\'' || c == '*' || c == '_' || c == '~') { k = k - 1UL; continue; }
            if (c == ')') {
                int open = 0;
                int close = 0;
                for (unsigned long j = i; j < k; j = j + 1UL) {
                    if (s[j] == '(') { open = open + 1; }
                    if (s[j] == ')') { close = close + 1; }
                }
                if (close > open) { k = k - 1UL; continue; }
            }
            break;
        }
        unsigned long minlen = www ? 5UL : 9UL;
        if (k - i < minlen) { return 0UL; }
        if (www) { href->push("http://"); }
        href->push_n(s + i, k - i);
        label->push_n(s + i, k - i);
        return k;
    }
    return 0UL;
}

static void md_parse_inlines_(struct MdParser_* st, struct MdNode* parent, const char* s, unsigned long n) {
    struct MdInline_ il;
    il.toks = Vec::new();
    il.head = -1;
    il.tail = -1;
    struct String buf = String::new();
    unsafe {
        unsigned long i = 0UL;
        while (i < n) {
            char c = s[i];
            // Backslash escapes and hard breaks.
            if (c == '\\') {
                if (i + 1UL < n && md_is_punct_(s[i + 1UL])) { buf.push_char(s[i + 1UL]); i = i + 2UL; continue; }
                if (i + 1UL < n && s[i + 1UL] == '\n') {
                    md_flush_text_(&il, &buf);
                    md_push_node_(&il, md_new_(MD_BREAK));
                    i = i + 2UL;
                    while (i < n && (s[i] == ' ' || s[i] == '\t')) { i = i + 1UL; }
                    continue;
                }
                buf.push_char(c); i = i + 1UL; continue;
            }
            // Code spans.
            if (c == '`') {
                unsigned long run = 0UL;
                while (i + run < n && s[i + run] == '`') { run = run + 1UL; }
                unsigned long j = i + run;
                unsigned long close = 0UL;
                while (j < n) {
                    if (s[j] == '`') {
                        unsigned long r2 = 0UL;
                        while (j + r2 < n && s[j + r2] == '`') { r2 = r2 + 1UL; }
                        if (r2 == run) { close = j; break; }
                        j = j + r2;
                    } else { j = j + 1UL; }
                }
                if (close == 0UL) { buf.push_n(s + i, run); i = i + run; continue; }
                struct String code = String::new();
                for (unsigned long k = i + run; k < close; k = k + 1UL) { code.push_char(s[k] == '\n' ? ' ' : s[k]); }
                // One space of padding on both sides is stripped (unless all spaces).
                if (code.len >= 2UL && code.char_at(0UL) == (int)' ' && code.char_at(code.len - 1UL) == (int)' ') {
                    int all = 1;
                    for (unsigned long k = 0UL; k < code.len; k = k + 1UL) { if (code.char_at(k) != (int)' ') { all = 0; } }
                    if (!all) {
                        struct String inner = code.substr(1UL, code.len - 1UL);
                        code.free();
                        code = inner;
                    }
                }
                md_flush_text_(&il, &buf);
                struct MdNode* cs = md_new_(MD_CODE_SPAN);
                cs->text = code;
                md_push_node_(&il, cs);
                i = close + run;
                continue;
            }
            if (c == '&') {
                unsigned long e = md_entity_(s, i, n, &buf);
                if (e > 0UL) { i = e; continue; }
                buf.push_char(c); i = i + 1UL; continue;
            }
            if (c == '<') {
                struct String href = String::new();
                struct String label = String::new();
                unsigned long e = md_autolink_(s, i, n, &href, &label);
                if (e > 0UL) {
                    md_flush_text_(&il, &buf);
                    struct MdNode* a = md_new_(MD_LINK);
                    a->text.push_str(&href);
                    struct MdNode* t = md_new_(MD_TEXT);
                    t->text.push_str(&label);
                    md_add_(a, t);
                    md_push_node_(&il, a);
                    href.free(); label.free();
                    i = e;
                    continue;
                }
                href.free(); label.free();
                if (st->opts.html) {
                    unsigned long h = md_match_html_tag_(s, i, n);
                    if (h > 0UL) {
                        md_flush_text_(&il, &buf);
                        struct MdNode* raw = md_new_(MD_HTML_INLINE);
                        raw->text.push_n(s + i, h - i);
                        md_push_node_(&il, raw);
                        i = h;
                        continue;
                    }
                }
                buf.push_char(c); i = i + 1UL; continue;
            }
            // Links and images.
            if (c == '[' || (c == '!' && i + 1UL < n && s[i + 1UL] == '[')) {
                int image = c == '!';
                unsigned long lb = image ? i + 1UL : i;
                unsigned long rb = md_match_bracket_(s, lb, n);
                if (rb > 0UL && rb + 1UL < n && s[rb + 1UL] == '(') {
                    struct String dest = String::new();
                    struct String title = String::new();
                    unsigned long e = md_link_tail_(s, rb + 1UL, n, &dest, &title);
                    if (e > 0UL) {
                        md_flush_text_(&il, &buf);
                        struct MdNode* a = md_new_(image ? MD_IMAGE : MD_LINK);
                        a->text.push_str(&dest);
                        a->title.push_str(&title);
                        md_parse_inlines_(st, a, s + lb + 1UL, rb - lb - 1UL);
                        md_push_node_(&il, a);
                        dest.free(); title.free();
                        i = e;
                        continue;
                    }
                    dest.free(); title.free();
                }
                buf.push_char(c); i = i + 1UL; continue;
            }
            // Emphasis / strikethrough delimiter runs.
            if (c == '*' || c == '_' || (c == '~' && st->opts.strikethrough)) {
                unsigned long run = 0UL;
                while (i + run < n && s[i + run] == c) { run = run + 1UL; }
                if (c == '~' && run > 2UL) { buf.push_n(s + i, run); i = i + run; continue; }
                char before = i > 0UL ? s[i - 1UL] : ' ';
                char after = i + run < n ? s[i + run] : ' ';
                int left = !md_is_space_(after) && (!md_is_punct_(after) || md_is_space_(before) || md_is_punct_(before));
                int right = !md_is_space_(before) && (!md_is_punct_(before) || md_is_space_(after) || md_is_punct_(after));
                int can_open = left;
                int can_close = right;
                if (c == '_') {
                    can_open = left && (!right || md_is_punct_(before));
                    can_close = right && (!left || md_is_punct_(after));
                }
                md_flush_text_(&il, &buf);
                struct MdTok_ t;
                t.kind = MD_TOK_DELIM;
                t.text = String::from_n(s + i, run);
                t.node = (struct MdNode*)0;
                t.dch = c; t.count = (int)run; t.orig = (int)run;
                t.can_open = can_open; t.can_close = can_close;
                md_push_tok_(&il, t);
                i = i + run;
                continue;
            }
            // Line breaks: two trailing spaces = hard break, else soft.
            if (c == '\n') {
                unsigned long spaces = 0UL;
                while (spaces < buf.len && buf.char_at(buf.len - 1UL - spaces) == (int)' ') { spaces = spaces + 1UL; }
                buf.truncate(buf.len - spaces);
                if (spaces >= 2UL) {
                    md_flush_text_(&il, &buf);
                    md_push_node_(&il, md_new_(MD_BREAK));
                } else {
                    buf.push_char('\n');
                }
                i = i + 1UL;
                while (i < n && (s[i] == ' ' || s[i] == '\t')) { i = i + 1UL; }
                continue;
            }
            // GFM bare URLs.
            if (st->opts.autolinks && (c == 'h' || c == 'w') &&
                (i == 0UL || md_is_space_(s[i - 1UL]) || s[i - 1UL] == '(' || s[i - 1UL] == '*' || s[i - 1UL] == '_' || s[i - 1UL] == '~')) {
                struct String href = String::new();
                struct String label = String::new();
                unsigned long e = md_bare_url_(s, i, n, &href, &label);
                if (e > 0UL) {
                    md_flush_text_(&il, &buf);
                    struct MdNode* a = md_new_(MD_LINK);
                    a->text.push_str(&href);
                    struct MdNode* t = md_new_(MD_TEXT);
                    t->text.push_str(&label);
                    md_add_(a, t);
                    md_push_node_(&il, a);
                    href.free(); label.free();
                    i = e;
                    continue;
                }
                href.free(); label.free();
            }
            buf.push_char(c);
            i = i + 1UL;
        }
        md_flush_text_(&il, &buf);
        md_process_emphasis_(&il);
        // Token list -> children.
        int t = il.head;
        while (t >= 0) {
            struct MdTok_* tk = md_tok_(&il, t);
            md_add_inline_(parent, md_tok_node_(tk));
            t = tk->next;
        }
        // Trim the paragraph's final trailing whitespace.
        unsigned long nc = parent->children.length();
        if (nc > 0UL) {
            struct MdNode* last = *parent->children.at(nc - 1UL);
            if (last->kind == MD_TEXT) {
                struct String tr = last->text.trim_right();
                last->text.free();
                last->text = tr;
            }
        }
        for (unsigned long k = 0UL; k < il.toks.length(); k = k + 1UL) {
            struct MdTok_* tk = il.toks.at(k);
            tk->text.free();
            if (tk->node != (struct MdNode*)0) { tk->node->free(); }
        }
        il.toks.free();
        buf.free();
    }
}

// ── block helpers ────────────────────────────────────────────────────────────

static unsigned long md_indent_(const char* l) {
    unsigned long k = 0UL;
    unsafe { while (l[k] == ' ') { k = k + 1UL; } }
    return k;
}

static int md_blank_(const char* l) {
    unsafe {
        for (unsigned long k = 0UL; l[k] != '\0'; k = k + 1UL) {
            if (l[k] != ' ' && l[k] != '\t') { return 0; }
        }
    }
    return 1;
}

// Line 'l' with up to 'k' leading spaces removed.
static struct String md_dedent_(const char* l, unsigned long k) {
    unsigned long i = 0UL;
    unsafe {
        while (i < k && l[i] == ' ') { i = i + 1UL; }
        return String::from(l + i);
    }
}

// Opening code fence: fills char, length and indent.
static int md_fence_open_(const char* l, char* fch, unsigned long* flen, unsigned long* find) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        char c = l[ind];
        if (c != '`' && c != '~') { return 0; }
        unsigned long k = ind;
        while (l[k] == c) { k = k + 1UL; }
        if (k - ind < 3UL) { return 0; }
        if (c == '`') {
            for (unsigned long j = k; l[j] != '\0'; j = j + 1UL) { if (l[j] == '`') { return 0; } }
        }
        *fch = c; *flen = k - ind; *find = ind;
    }
    return 1;
}

static int md_fence_close_(const char* l, char fch, unsigned long flen) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        unsigned long k = ind;
        while (l[k] == fch) { k = k + 1UL; }
        if (k - ind < flen) { return 0; }
        while (l[k] == ' ' || l[k] == '\t') { k = k + 1UL; }
        return l[k] == '\0';
    }
}

static int md_atx_level_(const char* l) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        int lvl = 0;
        while (l[ind + (unsigned long)lvl] == '#') { lvl = lvl + 1; }
        if (lvl == 0 || lvl > 6) { return 0; }
        char c = l[ind + (unsigned long)lvl];
        return (c == ' ' || c == '\t' || c == '\0') ? lvl : 0;
    }
}

static int md_hr_(const char* l) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        char c = l[ind];
        if (c != '-' && c != '*' && c != '_') { return 0; }
        int count = 0;
        for (unsigned long k = ind; l[k] != '\0'; k = k + 1UL) {
            if (l[k] == c) { count = count + 1; }
            else if (l[k] != ' ' && l[k] != '\t') { return 0; }
        }
        return count >= 3;
    }
}

// Setext underline: 1 for '===', 2 for '---' (only that character).
static int md_setext_(const char* l) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        char c = l[ind];
        if (c != '=' && c != '-') { return 0; }
        unsigned long k = ind;
        while (l[k] == c) { k = k + 1UL; }
        while (l[k] == ' ' || l[k] == '\t') { k = k + 1UL; }
        if (l[k] != '\0') { return 0; }
        return c == '=' ? 1 : 2;
    }
}

static int md_quote_(const char* l) {
    unsigned long ind = md_indent_(l);
    unsafe { return ind <= 3UL && l[ind] == '>'; }
}

struct MdMarker_ {
    int ordered;
    char ch;              // bullet char, or '.' / ')' for ordered
    int start;
    unsigned long width;  // content column (spaces to strip for continuation lines)
    int empty;            // nothing after the marker
};

static int md_list_marker_(const char* l, struct MdMarker_* m) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        unsigned long k = ind;
        char c = l[k];
        if (c == '-' || c == '*' || c == '+') {
            m->ordered = 0; m->ch = c; m->start = 1;
            k = k + 1UL;
        } else if (c >= '0' && c <= '9') {
            int v = 0;
            unsigned long d = 0UL;
            while (l[k] >= '0' && l[k] <= '9' && d < 9UL) { v = v * 10 + (int)(l[k] - '0'); k = k + 1UL; d = d + 1UL; }
            if (l[k] != '.' && l[k] != ')') { return 0; }
            m->ordered = 1; m->ch = l[k]; m->start = v;
            k = k + 1UL;
        } else {
            return 0;
        }
        if (l[k] != ' ' && l[k] != '\t' && l[k] != '\0') { return 0; }
        unsigned long sp = 0UL;
        while (l[k + sp] == ' ') { sp = sp + 1UL; }
        m->empty = l[k + sp] == '\0';
        if (m->empty || sp > 4UL) { sp = 1UL; }
        m->width = k + sp;
    }
    return 1;
}

static int md_container_open_(const char* l, struct String* kind, struct String* title) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        unsigned long k = ind;
        while (l[k] == ':') { k = k + 1UL; }
        if (k - ind < 3UL) { return 0; }
        while (l[k] == ' ') { k = k + 1UL; }
        unsigned long ks = k;
        while (l[k] != '\0' && l[k] != ' ' && l[k] != '\t') { k = k + 1UL; }
        if (k == ks) { return 0; }
        kind->push_n(l + ks, k - ks);
        while (l[k] == ' ' || l[k] == '\t') { k = k + 1UL; }
        struct String t = String::from(l + k);
        struct String tt = t.trim_right();
        title->push_str(&tt);
        t.free(); tt.free();
    }
    return 1;
}

static int md_container_close_(const char* l) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        unsigned long k = ind;
        while (l[k] == ':') { k = k + 1UL; }
        if (k - ind < 3UL) { return 0; }
        while (l[k] == ' ' || l[k] == '\t') { k = k + 1UL; }
        return l[k] == '\0';
    }
}

// HTML block start: 1 = <script|pre|style|textarea (ends at its close tag),
// 2 = comment (ends at '-->'), 6 = block-level tag (ends at a blank line),
// 7 = any other complete tag alone on the line (ends at a blank line;
// can't interrupt a paragraph). 0 = not an HTML block.
static int md_html_start_(const char* l) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        const char* p = l + ind;
        if (p[0] != '<') { return 0; }
        if (p[1] == '!' && p[2] == '-' && p[3] == '-') { return 2; }
        unsigned long k = 1UL;
        if (p[k] == '/') { k = k + 1UL; }
        unsigned long ns = k;
        while (md_is_alnum_(p[k]) || p[k] == '-') { k = k + 1UL; }
        if (k == ns) { return 0; }
        struct String name = String::from_n(p + ns, k - ns);
        struct String lower = name.to_lower();
        name.free();
        int t = 0;
        if (ns == 1UL && (lower.eq_cstr("script") || lower.eq_cstr("pre") || lower.eq_cstr("style") || lower.eq_cstr("textarea"))) { t = 1; }
        else {
            const char* blocks = " address article aside base basefont blockquote body caption center col colgroup dd details dialog dir div dl dt fieldset figcaption figure footer form frame frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li link main menu menuitem nav noframes ol optgroup option p param search section summary table tbody td tfoot th thead title tr track ul ";
            struct String needle = String::from(" ");
            needle.push_str(&lower);
            needle.push(" ");
            if (!str_find(blocks, needle.as_ptr()).is_null()) {
                char c = p[k];
                if (c == ' ' || c == '\t' || c == '\0' || c == '>' || (c == '/' && p[k + 1UL] == '>')) { t = 6; }
            }
            needle.free();
            if (t == 0) {
                unsigned long n = md_strlen_(p);
                unsigned long e = md_match_html_tag_(p, 0UL, n);
                if (e > 0UL && md_blank_(p + e)) { t = 7; }
            }
        }
        lower.free();
        return t;
    }
}

// A GFM delimiter row ('| --- | :-: |'): the number of columns, with
// their alignments pushed to 'aligns'; 0 if it isn't one.
static int md_table_delim_(const char* l, struct Vec<int>* aligns) {
    unsigned long ind = md_indent_(l);
    if (ind > 3UL) { return 0; }
    unsafe {
        unsigned long k = ind;
        int cols = 0;
        int pipes = 0;
        if (l[k] == '|') { k = k + 1UL; pipes = 1; }
        for (;;) {
            while (l[k] == ' ' || l[k] == '\t') { k = k + 1UL; }
            if (l[k] == '\0') { break; }
            int left = 0;
            int right = 0;
            if (l[k] == ':') { left = 1; k = k + 1UL; }
            unsigned long dashes = 0UL;
            while (l[k] == '-') { dashes = dashes + 1UL; k = k + 1UL; }
            if (dashes == 0UL) { return 0; }
            if (l[k] == ':') { right = 1; k = k + 1UL; }
            while (l[k] == ' ' || l[k] == '\t') { k = k + 1UL; }
            aligns->push(left && right ? MD_ALIGN_CENTER : (right ? MD_ALIGN_RIGHT : (left ? MD_ALIGN_LEFT : MD_ALIGN_NONE)));
            cols = cols + 1;
            if (l[k] == '|') { k = k + 1UL; pipes = pipes + 1; continue; }
            if (l[k] == '\0') { break; }
            return 0;
        }
        if (cols == 1 && pipes == 0) { return 0; }   // '---' alone is a setext/hr line
        return cols;
    }
}

// Splits a table row into cell texts (unescaped '|' outside code spans
// separate cells; '\|' becomes '|').
static void md_table_cells_(const char* l, struct Vec<struct String>* cells) {
    unsafe {
        unsigned long n = md_strlen_(l);
        unsigned long a = md_indent_(l);
        unsigned long e = n;
        while (e > a && (l[e - 1UL] == ' ' || l[e - 1UL] == '\t')) { e = e - 1UL; }
        if (a < e && l[a] == '|') { a = a + 1UL; }
        if (e > a && l[e - 1UL] == '|' && !(e >= a + 2UL && l[e - 2UL] == '\\')) { e = e - 1UL; }
        struct String cur = String::new();
        unsigned long k = a;
        while (k < e) {
            char c = l[k];
            if (c == '`') {
                unsigned long run = 0UL;
                while (k + run < e && l[k + run] == '`') { run = run + 1UL; }
                unsigned long j = k + run;
                unsigned long close = 0UL;
                while (j < e) {
                    if (l[j] == '`') {
                        unsigned long r2 = 0UL;
                        while (j + r2 < e && l[j + r2] == '`') { r2 = r2 + 1UL; }
                        if (r2 == run) { close = j + r2; break; }
                        j = j + r2;
                    } else { j = j + 1UL; }
                }
                unsigned long upto = close > 0UL ? close : k + run;
                for (unsigned long q = k; q < upto; q = q + 1UL) {
                    if (l[q] == '\\' && q + 1UL < upto && l[q + 1UL] == '|') { continue; }
                    cur.push_char(l[q]);
                }
                k = upto;
                continue;
            }
            if (c == '\\' && k + 1UL < e && l[k + 1UL] == '|') { cur.push_char('|'); k = k + 2UL; continue; }
            if (c == '\\' && k + 1UL < e) { cur.push_char(c); cur.push_char(l[k + 1UL]); k = k + 2UL; continue; }
            if (c == '|') {
                cells->push(cur.trim());
                cur.clear();
                k = k + 1UL;
                continue;
            }
            cur.push_char(c);
            k = k + 1UL;
        }
        cells->push(cur.trim());
        cur.free();
    }
}

static void md_free_strings_(struct Vec<struct String>* v) {
    unsafe {
        for (unsigned long i = 0UL; i < v->length(); i = i + 1UL) { v->at(i)->free(); }
        v->free();
    }
}

// Could line 'l' start a table header row (has a '|')?
static int md_has_pipe_(const char* l) {
    unsafe { return !str_find(l, "|").is_null(); }
}

// Does 'l' start a block that interrupts a paragraph?
static int md_interrupts_(struct MdParser_* st, const char* l) {
    struct MdOptions opt;
    unsafe { opt = st->opts; }
    if (md_blank_(l)) { return 1; }
    if (md_indent_(l) > 3UL) { return 0; }
    char fc; unsigned long fl; unsigned long fi;
    if (md_fence_open_(l, &fc, &fl, &fi)) { return 1; }
    if (md_atx_level_(l) > 0) { return 1; }
    if (md_hr_(l)) { return 1; }
    if (md_quote_(l)) { return 1; }
    struct MdMarker_ m;
    if (md_list_marker_(l, &m) && !m.empty && (!m.ordered || m.start == 1)) { return 1; }
    if (opt.containers) {
        struct String k = String::new();
        struct String t = String::new();
        int open = md_container_open_(l, &k, &t) || md_container_close_(l);
        k.free(); t.free();
        if (open) { return 1; }
    }
    if (opt.html) {
        int h = md_html_start_(l);
        if (h >= 1 && h <= 6) { return 1; }
    }
    return 0;
}

// ── headings ─────────────────────────────────────────────────────────────────

static int md_id_taken_(struct MdParser_* st, const &String id) {
    unsafe {
        for (unsigned long i = 0UL; i < st->ids.length(); i = i + 1UL) {
            if (st->ids.at(i)->eq(id)) { return 1; }
        }
    }
    return 0;
}

static void md_heading_(struct MdParser_* st, struct MdNode* parent, int level, const char* raw) {
    struct MdNode* h = md_new_(MD_HEADING);
    unsafe {
        h->level = level;
        struct String text = String::from(raw);
        struct String t = text.trim();
        text.free();
        if (st->opts.heading_ids && t.ends_with("}")) {
            long long open = t.last_index_of("{#");
            if (open >= 0LL) {
                h->id = t.substr((unsigned long)open + 2UL, t.len - 1UL);
                struct String rest = t.substr(0UL, (unsigned long)open);
                t.free();
                t = rest.trim();
                rest.free();
            }
        }
        md_parse_inlines_(st, h, t.as_ptr(), t.len);
        t.free();
        if (st->opts.heading_ids) {
            if (h->id.len == 0UL) {
                struct String plain = String::new();
                h->plain_text(&plain);
                struct String base = String::new();
                md_slugify(plain.as_ptr(), &base);
                plain.free();
                h->id.push_str(&base);
                for (long long dup = 1LL; md_id_taken_(st, &h->id); dup = dup + 1LL) {
                    h->id.clear();
                    h->id.push_str(&base);
                    h->id.push_char('-');
                    h->id.push_int(dup);
                }
                base.free();
            }
            st->ids.push(h->id.clone());
        }
        md_add_(parent, h);
    }
}

// ATX heading content: the text after the '#'s, closing '#'s removed.
static struct String md_atx_text_(const char* l, int level) {
    unsigned long ind = md_indent_(l);
    unsafe {
        struct String s = String::from(l + ind + (unsigned long)level);
        struct String t = s.trim();
        s.free();
        // Closing sequence: spaces then only '#'s at the end.
        unsigned long e = t.len;
        while (e > 0UL && t.char_at(e - 1UL) == (int)'#') { e = e - 1UL; }
        if (e < t.len && (e == 0UL || t.char_at(e - 1UL) == (int)' ' || t.char_at(e - 1UL) == (int)'\t')) {
            t.truncate(e);
            struct String r = t.trim_right();
            t.free();
            return r;
        }
        return t;
    }
}

// ── block parsing ────────────────────────────────────────────────────────────

static void md_parse_blocks_(struct MdParser_* st, struct MdNode* parent, struct Vec<struct String>* L,
                             unsigned long lo, unsigned long hi);

static const char* md_line_(struct Vec<struct String>* L, unsigned long i) {
    unsafe { return L->at(i)->as_ptr(); }
}

// Parses lines [i, ...) of a list starting at 'i'; returns the next line.
static unsigned long md_parse_list_(struct MdParser_* st, struct MdNode* parent, struct Vec<struct String>* L,
                                    unsigned long i, unsigned long hi) {
    struct MdMarker_ first;
    md_list_marker_(md_line_(L, i), &first);
    struct MdNode* list = md_new_(MD_LIST);
    unsafe {
        list->ordered = first.ordered;
        list->start = first.start;
        list->tight = 1;
    }
    while (i < hi) {
        struct MdMarker_ m;
        const char* l = md_line_(L, i);
        if (md_hr_(l) || !md_list_marker_(l, &m) || m.ordered != first.ordered || m.ch != first.ch) { break; }
        struct Vec<struct String> item = Vec::new();
        unsafe {
            if (m.empty) { item.push(String::new()); }
            else { item.push(String::from(l + m.width)); }
        }
        unsigned long j = i + 1UL;
        int in_fence = 0;
        char fch = '`'; unsigned long flen = 0UL; unsigned long find = 0UL;
        {
            char c2; unsigned long l2; unsigned long i2;
            if (md_fence_open_(item.at(0UL).as_ptr(), &c2, &l2, &i2)) { in_fence = 1; fch = c2; flen = l2; }
        }
        int last_blank = 0;
        int last_para = !m.empty;
        while (j < hi) {
            const char* cl = md_line_(L, j);
            if (md_blank_(cl)) {
                // Blank lines stay in the item only if indented content follows.
                unsigned long k = j;
                while (k < hi && md_blank_(md_line_(L, k))) { k = k + 1UL; }
                if (k < hi && md_indent_(md_line_(L, k)) >= m.width) {
                    while (j < k) { item.push(String::new()); j = j + 1UL; }
                    last_blank = 1;
                    last_para = 0;
                    continue;
                }
                break;
            }
            if (md_indent_(cl) >= m.width) {
                struct String d = md_dedent_(cl, m.width);
                if (in_fence) { if (md_fence_close_(d.as_ptr(), fch, flen)) { in_fence = 0; } }
                else {
                    char c2; unsigned long l2; unsigned long i2;
                    if (md_fence_open_(d.as_ptr(), &c2, &l2, &i2)) { in_fence = 1; fch = c2; flen = l2; }
                }
                last_para = !in_fence && !md_blank_(d.as_ptr());
                item.push(d);
                j = j + 1UL;
                last_blank = 0;
                continue;
            }
            // A new list marker ends this item (a sibling or another list).
            {
                struct MdMarker_ other;
                if (md_list_marker_(cl, &other) || md_hr_(cl)) { break; }
            }
            // Lazy continuation of a paragraph.
            if (last_para && !last_blank && !in_fence && !md_interrupts_(st, cl) && !md_setext_(cl)) {
                unsafe { item.push(String::from(cl + md_indent_(cl))); }
                j = j + 1UL;
                continue;
            }
            break;
        }
        (void)find;
        // Interior blank lines between blocks make the list loose.
        {
            int fence = 0;
            char fc = '`'; unsigned long fl = 0UL;
            unsigned long n = item.length();
            unsigned long lastNonBlank = 0UL;
            for (unsigned long k = 0UL; k < n; k = k + 1UL) { if (!md_blank_(item.at(k).as_ptr())) { lastNonBlank = k; } }
            for (unsigned long k = 0UL; k < lastNonBlank; k = k + 1UL) {
                const char* il = item.at(k).as_ptr();
                char c2; unsigned long l2; unsigned long i2;
                if (fence) { if (md_fence_close_(il, fc, fl)) { fence = 0; } continue; }
                if (md_fence_open_(il, &c2, &l2, &i2)) { fence = 1; fc = c2; fl = l2; continue; }
                if (md_blank_(il) && k > 0UL) {
                    // A blank between two lines that belong to *this* item's
                    // top level (a nested list's own blank lines count too,
                    // which matches markdown-it closely enough).
                    unsafe { list->tight = 0; }
                }
            }
        }
        struct MdNode* li = md_new_(MD_LIST_ITEM);
        md_parse_blocks_(st, li, &item, 0UL, item.length());
        md_add_(list, li);
        md_free_strings_(&item);
        i = j;
        // Blank lines before the next item of this list: loose.
        unsigned long k = i;
        while (k < hi && md_blank_(md_line_(L, k))) { k = k + 1UL; }
        if (k > i && k < hi) {
            struct MdMarker_ nm;
            const char* nl = md_line_(L, k);
            if (!md_hr_(nl) && md_list_marker_(nl, &nm) && nm.ordered == first.ordered && nm.ch == first.ch) {
                unsafe { list->tight = 0; }
                i = k;
                continue;
            }
        }
    }
    md_add_(parent, list);
    return i;
}

static void md_parse_blocks_(struct MdParser_* st, struct MdNode* parent, struct Vec<struct String>* L,
                             unsigned long lo, unsigned long hi) {
    struct MdOptions opt;
    unsafe { opt = st->opts; }
    unsigned long i = lo;
    while (i < hi) {
        const char* l = md_line_(L, i);
        if (md_blank_(l)) { i = i + 1UL; continue; }
        unsigned long ind = md_indent_(l);

        // Indented code.
        if (ind >= 4UL) {
            struct MdNode* cb = md_new_(MD_CODE_BLOCK);
            unsigned long j = i;
            unsigned long lastCode = i;
            while (j < hi) {
                const char* cl = md_line_(L, j);
                if (!md_blank_(cl) && md_indent_(cl) < 4UL) { break; }
                if (!md_blank_(cl)) { lastCode = j; }
                j = j + 1UL;
            }
            for (unsigned long k = i; k <= lastCode; k = k + 1UL) {
                struct String d = md_dedent_(md_line_(L, k), 4UL);
                unsafe { cb->text.push_str(&d); cb->text.push_char('\n'); }
                d.free();
            }
            md_add_(parent, cb);
            i = lastCode + 1UL;
            continue;
        }

        // Fenced code.
        char fch; unsigned long flen; unsigned long find;
        if (md_fence_open_(l, &fch, &flen, &find)) {
            struct MdNode* cb = md_new_(MD_CODE_BLOCK);
            unsafe {
                struct String info = String::from(l + find + flen);
                struct String it = info.trim();
                struct String un = md_unescape_(it.as_ptr(), it.len);
                cb->info.push_str(&un);
                info.free(); it.free(); un.free();
            }
            unsigned long j = i + 1UL;
            while (j < hi) {
                const char* cl = md_line_(L, j);
                if (md_fence_close_(cl, fch, flen)) { j = j + 1UL; break; }
                struct String d = md_dedent_(cl, find);
                unsafe { cb->text.push_str(&d); cb->text.push_char('\n'); }
                d.free();
                j = j + 1UL;
            }
            md_add_(parent, cb);
            i = j;
            continue;
        }

        int lvl = md_atx_level_(l);
        if (lvl > 0) {
            struct String t = md_atx_text_(l, lvl);
            md_heading_(st, parent, lvl, t.as_ptr());
            t.free();
            i = i + 1UL;
            continue;
        }

        if (md_hr_(l)) { md_add_(parent, md_new_(MD_HR)); i = i + 1UL; continue; }

        // Block quote.
        if (md_quote_(l)) {
            struct Vec<struct String> q = Vec::new();
            unsigned long j = i;
            int last_para = 0;
            while (j < hi) {
                const char* cl = md_line_(L, j);
                if (md_quote_(cl)) {
                    unsigned long k = md_indent_(cl) + 1UL;
                    unsafe { if (cl[k] == ' ') { k = k + 1UL; } }
                    struct String d;
                    unsafe { d = String::from(cl + k); }
                    last_para = !md_blank_(d.as_ptr());
                    q.push(d);
                    j = j + 1UL;
                    continue;
                }
                if (last_para && !md_blank_(cl) && !md_interrupts_(st, cl)) {   // lazy continuation
                    unsafe { q.push(String::from(cl + md_indent_(cl))); }
                    j = j + 1UL;
                    continue;
                }
                break;
            }
            struct MdNode* bq = md_new_(MD_BLOCKQUOTE);
            md_parse_blocks_(st, bq, &q, 0UL, q.length());
            md_add_(parent, bq);
            md_free_strings_(&q);
            i = j;
            continue;
        }

        // Custom container.
        if (opt.containers) {
            struct String kind = String::new();
            struct String title = String::new();
            if (md_container_open_(l, &kind, &title)) {
                struct Vec<struct String> inner = Vec::new();
                unsigned long j = i + 1UL;
                int depth = 1;
                int fence = 0;
                char fc = '`'; unsigned long fl = 0UL;
                while (j < hi) {
                    const char* cl = md_line_(L, j);
                    char c2; unsigned long l2; unsigned long i2;
                    if (fence) { if (md_fence_close_(cl, fc, fl)) { fence = 0; } }
                    else if (md_fence_open_(cl, &c2, &l2, &i2)) { fence = 1; fc = c2; fl = l2; }
                    else if (md_container_close_(cl)) {
                        depth = depth - 1;
                        if (depth == 0) { j = j + 1UL; break; }
                    } else {
                        struct String k2 = String::new();
                        struct String t2 = String::new();
                        if (md_container_open_(cl, &k2, &t2)) { depth = depth + 1; }
                        k2.free(); t2.free();
                    }
                    unsafe { inner.push(String::from(cl)); }
                    j = j + 1UL;
                }
                struct MdNode* box = md_new_(MD_CONTAINER);
                unsafe { box->info.push_str(&kind); }
                if (title.len > 0UL) {
                    struct MdNode* tn = md_new_(MD_CONTAINER_TITLE);
                    md_parse_inlines_(st, tn, title.as_ptr(), title.len);
                    md_add_(box, tn);
                }
                md_parse_blocks_(st, box, &inner, 0UL, inner.length());
                md_add_(parent, box);
                md_free_strings_(&inner);
                kind.free(); title.free();
                i = j;
                continue;
            }
            kind.free(); title.free();
        }

        // Raw HTML block.
        if (opt.html) {
            int ht = md_html_start_(l);
            if (ht > 0) {
                struct MdNode* hb = md_new_(MD_HTML_BLOCK);
                unsigned long j = i;
                while (j < hi) {
                    const char* cl = md_line_(L, j);
                    if ((ht == 6 || ht == 7) && md_blank_(cl)) { break; }
                    unsafe { hb->text.push(cl); hb->text.push_char('\n'); }
                    j = j + 1UL;
                    if (ht == 2) { unsafe { if (!str_find(cl, "-->").is_null()) { break; } } }
                    if (ht == 1) {
                        struct String lc;
                        unsafe { lc = String::from(cl); }
                        struct String lo2 = lc.to_lower();
                        int done = lo2.contains("</script>") || lo2.contains("</pre>") || lo2.contains("</style>") || lo2.contains("</textarea>");
                        lc.free(); lo2.free();
                        if (done) { break; }
                    }
                }
                md_add_(parent, hb);
                i = j;
                continue;
            }
        }

        // List.
        struct MdMarker_ mk;
        if (md_list_marker_(l, &mk)) {
            i = md_parse_list_(st, parent, L, i, hi);
            continue;
        }

        // Table: a header row with a '|' followed by a matching delimiter row.
        if (opt.tables && md_has_pipe_(l) && i + 1UL < hi) {
            struct Vec<int> aligns = Vec::new();
            int cols = md_table_delim_(md_line_(L, i + 1UL), &aligns);
            struct Vec<struct String> head = Vec::new();
            md_table_cells_(l, &head);
            if (cols > 0 && (unsigned long)cols == head.length()) {
                struct MdNode* table = md_new_(MD_TABLE);
                unsigned long j = i;
                int first = 1;
                while (j < hi) {
                    const char* cl = md_line_(L, j);
                    if (!first && (md_blank_(cl) || md_interrupts_(st, cl))) { break; }
                    struct Vec<struct String> cells = Vec::new();
                    md_table_cells_(cl, &cells);
                    struct MdNode* row = md_new_(MD_TABLE_ROW);
                    unsafe { row->header = first; }
                    for (unsigned long c = 0UL; c < (unsigned long)cols; c = c + 1UL) {
                        struct MdNode* cell = md_new_(MD_TABLE_CELL);
                        unsafe {
                            cell->header = first;
                            cell->align = *aligns.at(c);
                            if (c < cells.length()) {
                                struct String* tx = cells.at(c);
                                md_parse_inlines_(st, cell, tx->as_ptr(), tx->len);
                            }
                        }
                        md_add_(row, cell);
                    }
                    md_free_strings_(&cells);
                    md_add_(table, row);
                    j = j + (first ? 2UL : 1UL);
                    first = 0;
                }
                md_add_(parent, table);
                md_free_strings_(&head);
                aligns.free();
                i = j;
                continue;
            }
            md_free_strings_(&head);
            aligns.free();
        }

        // Paragraph (or setext heading).
        struct String text = String::new();
        unsigned long j = i;
        int setext = 0;
        while (j < hi) {
            const char* cl = md_line_(L, j);
            if (j > i) {
                int sx = md_setext_(cl);
                if (sx > 0) { setext = sx; j = j + 1UL; break; }
                if (md_interrupts_(st, cl)) { break; }
                if (opt.tables && md_has_pipe_(cl) && j + 1UL < hi) {
                    struct Vec<int> al = Vec::new();
                    int c2 = md_table_delim_(md_line_(L, j + 1UL), &al);
                    al.free();
                    if (c2 > 0) { break; }
                }
                text.push_char('\n');
            }
            unsafe { text.push(cl + md_indent_(cl)); }
            j = j + 1UL;
        }
        if (setext > 0) {
            md_heading_(st, parent, setext, text.as_ptr());
        } else {
            struct MdNode* p = md_new_(MD_PARAGRAPH);
            md_parse_inlines_(st, p, text.as_ptr(), text.len);
            md_add_(parent, p);
        }
        text.free();
        i = j;
    }
}

&heap MdNode MdNode::parse_opts(const &char text, const &MdOptions opts) {
    struct MdParser_ st;
    st.opts = *opts;
    st.ids = Vec::new();
    // Lines, with tabs in leading whitespace expanded to 4-column stops
    // and '\r' dropped.
    struct Vec<struct String> lines = Vec::new();
    unsafe {
        const char* s = text;
        unsigned long i = 0UL;
        for (;;) {
            struct String line = String::new();
            int lead = 1;
            while (s[i] != '\0' && s[i] != '\n') {
                char c = s[i];
                if (c == '\r') { i = i + 1UL; continue; }
                if (c == '\t' && lead) {
                    unsigned long w = 4UL - (line.len % 4UL);
                    for (unsigned long k = 0UL; k < w; k = k + 1UL) { line.push_char(' '); }
                } else {
                    if (c != ' ') { lead = 0; }
                    line.push_char(c);
                }
                i = i + 1UL;
            }
            lines.push(line);
            if (s[i] == '\0') { break; }
            i = i + 1UL;
        }
    }
    struct MdNode* doc = md_new_(MD_DOC);
    md_parse_blocks_(&st, doc, &lines, 0UL, lines.length());
    md_free_strings_(&lines);
    md_free_strings_(&st.ids);
    unsafe { return doc; }
}

&heap MdNode MdNode::parse(const &char text) {
    struct MdOptions o = MdOptions::commonmark();
    return MdNode::parse_opts(text, &o);
}

// ── slugs, plain text ────────────────────────────────────────────────────────

void md_slugify(const &char text, &String out) {
    unsigned long start = out.len;
    int dash = 0;
    unsafe {
        const unsigned char* s = (const unsigned char*)text;
        unsigned long i = 0UL;
        while (s[i] != (unsigned char)0) {
            unsigned char c = s[i];
            int special = c <= (unsigned char)0x20 || c == (unsigned char)0x7F ||
                          (c < (unsigned char)0x80 && md_is_punct_((char)c));
            unsigned long width = 1UL;
            // Curly quotes U+2018/2019/201C/201D (E2 80 98/99/9C/9D) count as punctuation too.
            if (c == (unsigned char)0xE2 && s[i + 1UL] == (unsigned char)0x80 &&
                (s[i + 2UL] == (unsigned char)0x98 || s[i + 2UL] == (unsigned char)0x99 ||
                 s[i + 2UL] == (unsigned char)0x9C || s[i + 2UL] == (unsigned char)0x9D)) {
                special = 1;
                width = 3UL;
            }
            if (special) { dash = 1; i = i + width; continue; }
            if (dash && out.len > start) { out.push_char('-'); }
            dash = 0;
            out.push_char((c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c);
            i = i + 1UL;
        }
    }
    if (out.len > start) {
        int c = out.char_at(start);
        if (c >= (int)'0' && c <= (int)'9') { out.insert(start, "_"); }
    }
}

void MdNode::plain_text(&String out) const {
    int k = self.kind;
    if (k == MD_TEXT) {
        for (unsigned long i = 0UL; i < self.text.len; i = i + 1UL) {
            int c = self.text.char_at(i);
            out.push_char(c == (int)'\n' ? ' ' : (char)c);
        }
        return;
    }
    if (k == MD_CODE_SPAN || k == MD_CODE_BLOCK) { out.push_str(&self.text); return; }
    if (k == MD_BREAK) { out.push_char(' '); return; }
    if (k == MD_HTML_INLINE || k == MD_HTML_BLOCK) {
        html_strip_tags(out, self.text.as_ptr());
        return;
    }
    // Separate the children of block containers (their children are blocks).
    int block = k == MD_DOC || k == MD_LIST || k == MD_LIST_ITEM || k == MD_BLOCKQUOTE ||
                k == MD_TABLE || k == MD_TABLE_ROW || k == MD_CONTAINER;
    unsigned long n = self.children.length();
    for (unsigned long i = 0UL; i < n; i = i + 1UL) {
        &heap MdNode c;
        unsafe { c = *self.children.at(i); }
        if (block && i > 0UL && out.len > 0UL && out.char_at(out.len - 1UL) != (int)' ') { out.push_char(' '); }
        c.plain_text(out);
    }
}

// ── HTML rendering ───────────────────────────────────────────────────────────

typedef fn int(void*, const &MdNode, int, &String) MdHookFn_;

static void md_render_node_(const &MdNode n, struct String* out, const struct MdHookRaw_* hook, int tight);

static void md_render_children_(const &MdNode n, struct String* out, const struct MdHookRaw_* hook, int tight) {
    unsigned long cn = n.children.length();
    for (unsigned long i = 0UL; i < cn; i = i + 1UL) {
        &heap MdNode c;
        unsafe { c = *n.children.at(i); }
        md_render_node_(c, out, hook, tight);
    }
}

static void md_esc_(struct String* out, const &String s) {
    unsafe { html_escape(out, s.as_ptr()); }
}

static void md_open_(const &MdNode n, struct String* out) {
    unsafe {
        int k = n.kind;
        if (k == MD_HEADING) {
            out->push("<h"); out->push_int((long long)n.level);
            if (n.id.len > 0UL) { out->push(" id=\""); md_esc_(out, &n.id); out->push("\""); }
            out->push(">");
        } else if (k == MD_PARAGRAPH) { out->push("<p>"); }
        else if (k == MD_LIST) {
            if (n.ordered) {
                out->push("<ol");
                if (n.start != 1) { out->push(" start=\""); out->push_int((long long)n.start); out->push("\""); }
                out->push(">\n");
            } else { out->push("<ul>\n"); }
        }
        else if (k == MD_LIST_ITEM) { out->push("<li>"); }
        else if (k == MD_BLOCKQUOTE) { out->push("<blockquote>\n"); }
        else if (k == MD_BOLD) { out->push("<strong>"); }
        else if (k == MD_ITALIC) { out->push("<em>"); }
        else if (k == MD_STRIKE) { out->push("<del>"); }
        else if (k == MD_LINK) {
            out->push("<a href=\""); md_esc_(out, &n.text); out->push("\"");
            if (n.title.len > 0UL) { out->push(" title=\""); md_esc_(out, &n.title); out->push("\""); }
            out->push(">");
        }
        else if (k == MD_TABLE) { out->push("<table>\n<thead>\n"); }
        else if (k == MD_TABLE_ROW) { out->push("<tr>\n"); }
        else if (k == MD_TABLE_CELL) {
            out->push(n.header ? "<th" : "<td");
            if (n.align == MD_ALIGN_LEFT) { out->push(" style=\"text-align:left\""); }
            else if (n.align == MD_ALIGN_CENTER) { out->push(" style=\"text-align:center\""); }
            else if (n.align == MD_ALIGN_RIGHT) { out->push(" style=\"text-align:right\""); }
            out->push(">");
        }
        else if (k == MD_CONTAINER) {
            int details = n.info.eq_cstr("details");
            out->push(details ? "<details class=\"custom-block details\">" : "<div class=\"custom-block ");
            if (!details) { md_esc_(out, &n.info); out->push("\">"); }
            int titled = 0;
            if (n.children.length() > 0UL) { titled = (*n.children.at(0UL))->kind == MD_CONTAINER_TITLE; }
            if (!titled) {
                if (details) { out->push("<summary>Details</summary>\n"); }
                else {
                    out->push("<p class=\"custom-block-title\">");
                    struct String up = n.info.to_upper();
                    md_esc_(out, &up);
                    up.free();
                    out->push("</p>\n");
                }
            } else { out->push("\n"); }
        }
        else if (k == MD_CONTAINER_TITLE) { out->push("<p class=\"custom-block-title\">"); }
    }
}

static void md_close_(const &MdNode n, struct String* out, const &MdNode parent_hint) {
    unsafe {
        int k = n.kind;
        if (k == MD_HEADING) { out->push("</h"); out->push_int((long long)n.level); out->push(">\n"); }
        else if (k == MD_PARAGRAPH) { out->push("</p>\n"); }
        else if (k == MD_LIST) { out->push(n.ordered ? "</ol>\n" : "</ul>\n"); }
        else if (k == MD_LIST_ITEM) { out->push("</li>\n"); }
        else if (k == MD_BLOCKQUOTE) { out->push("</blockquote>\n"); }
        else if (k == MD_BOLD) { out->push("</strong>"); }
        else if (k == MD_ITALIC) { out->push("</em>"); }
        else if (k == MD_STRIKE) { out->push("</del>"); }
        else if (k == MD_LINK) { out->push("</a>"); }
        else if (k == MD_TABLE) { out->push("</tbody>\n</table>\n"); }
        else if (k == MD_TABLE_ROW) { out->push(n.header ? "</tr>\n</thead>\n<tbody>\n" : "</tr>\n"); }
        else if (k == MD_TABLE_CELL) { out->push(n.header ? "</th>\n" : "</td>\n"); }
        else if (k == MD_CONTAINER) { out->push(n.info.eq_cstr("details") ? "</details>\n" : "</div>\n"); }
        else if (k == MD_CONTAINER_TITLE) { out->push("</p>\n"); }
    }
    (void)parent_hint;
}

static void md_render_node_(const &MdNode n, struct String* out, const struct MdHookRaw_* hook, int tight) {
    int mode = 0;
    unsafe {
        if (hook != (const struct MdHookRaw_*)0 && hook->fn_ != (void*)0) {
            MdHookFn_ f = (MdHookFn_)hook->fn_;
            mode = f(hook->ctx, n, 1, out);
        }
    }
    if (mode == 1) { return; }
    int k = n.kind;
    // Leaves.
    if (mode == 0) {
        if (k == MD_TEXT) { md_esc_(out, &n.text); return; }
        if (k == MD_CODE_SPAN) { unsafe { out->push("<code>"); } md_esc_(out, &n.text); unsafe { out->push("</code>"); } return; }
        if (k == MD_BREAK) { unsafe { out->push("<br>\n"); } return; }
        if (k == MD_HR) { unsafe { out->push("<hr>\n"); } return; }
        if (k == MD_HTML_INLINE || k == MD_HTML_BLOCK) { unsafe { out->push_str(&n.text); } return; }
        if (k == MD_CODE_BLOCK) {
            unsafe {
                out->push("<pre><code");
                if (n.info.len > 0UL) {
                    long long sp = n.info.find_char(' ');
                    struct String lang = sp >= 0LL ? n.info.substr(0UL, (unsigned long)sp) : n.info.clone();
                    out->push(" class=\"language-"); md_esc_(out, &lang); out->push("\"");
                    lang.free();
                }
                out->push(">");
                md_esc_(out, &n.text);
                out->push("</code></pre>\n");
            }
            return;
        }
        if (k == MD_IMAGE) {
            unsafe {
                out->push("<img src=\""); md_esc_(out, &n.text); out->push("\" alt=\"");
                struct String alt = String::new();
                n.plain_text(&alt);
                md_esc_(out, &alt);
                alt.free();
                out->push("\"");
                if (n.title.len > 0UL) { out->push(" title=\""); md_esc_(out, &n.title); out->push("\""); }
                out->push(">");
            }
            return;
        }
        if (k == MD_PARAGRAPH && tight) {
            md_render_children_(n, out, hook, 0);
            unsafe { out->push("\n"); }
            return;
        }
        if (k != MD_DOC) { md_open_(n, out); }
        // markdown-it's item shape: '<li>text' in a tight list, else the
        // first block starts on its own line.
        if (k == MD_LIST_ITEM && n.children.length() > 0UL) {
            int firstPara;
            unsafe { firstPara = (*n.children.at(0UL))->kind == MD_PARAGRAPH; }
            if (!tight || !firstPara) { unsafe { out->push_char('\n'); } }
        }
    }
    // Children.
    int childTight = 0;
    if (k == MD_LIST_ITEM) { childTight = tight; }
    if (k == MD_LIST) {
        unsigned long cn = n.children.length();
        for (unsigned long i = 0UL; i < cn; i = i + 1UL) {
            &heap MdNode c;
            unsafe { c = *n.children.at(i); }
            md_render_node_(c, out, hook, n.tight);
        }
    } else {
        // A tight item starts with its text right after <li>; a block
        // child that follows inline text goes on a new line.
        md_render_children_(n, out, hook, childTight);
    }
    if (mode == 2) {
        int closed = 0;
        unsafe {
            MdHookFn_ f = (MdHookFn_)hook->fn_;
            closed = f(hook->ctx, n, 0, out);
        }
        if (closed == 1) { return; }
    }
    if (k == MD_LIST_ITEM && tight && out->ends_with("\n")) {
        // Text right before </li> (markdown-it's tight-item shape); a
        // trailing block keeps its newline.
        unsigned long cn = n.children.length();
        int lastPara = 0;
        if (cn > 0UL) { unsafe { lastPara = (*n.children.at(cn - 1UL))->kind == MD_PARAGRAPH; } }
        if (lastPara) { unsafe { out->truncate(out->len - 1UL); } }
    }
    if (k != MD_DOC) { md_close_(n, out, n); }
}

struct String MdNode::render_html() const {
    struct String out = String::new();
    md_render_node_(self, &out, (const struct MdHookRaw_*)0, 0);
    return out;
}

struct String md_render_html_raw_(const &MdNode node, const &MdHookRaw_ hook) {
    struct String out = String::new();
    unsafe { md_render_node_(node, &out, (const struct MdHookRaw_*)hook, 0); }
    return out;
}

// ── Markdown rendering ───────────────────────────────────────────────────────

static void md_md_inline_(const &MdNode n, struct String* out);

static void md_md_children_inline_(const &MdNode n, struct String* out) {
    unsigned long cn = n.children.length();
    for (unsigned long i = 0UL; i < cn; i = i + 1UL) {
        &heap MdNode c;
        unsafe { c = *n.children.at(i); }
        md_md_inline_(c, out);
    }
}

static void md_md_inline_(const &MdNode n, struct String* out) {
    unsafe {
        int k = n.kind;
        if (k == MD_TEXT) {
            for (unsigned long i = 0UL; i < n.text.len; i = i + 1UL) {
                char c = (char)n.text.char_at(i);
                if (c == '\\' || c == '*' || c == '_' || c == '`' || c == '[' || c == ']' || c == '<' ||
                    c == '&' || (c == '~')) { out->push_char('\\'); }
                out->push_char(c);
            }
        } else if (k == MD_BOLD) { out->push("**"); md_md_children_inline_(n, out); out->push("**"); }
        else if (k == MD_ITALIC) { out->push("*"); md_md_children_inline_(n, out); out->push("*"); }
        else if (k == MD_STRIKE) { out->push("~~"); md_md_children_inline_(n, out); out->push("~~"); }
        else if (k == MD_CODE_SPAN) {
            // A fence longer than any backtick run inside.
            unsigned long longest = 0UL;
            unsigned long run = 0UL;
            for (unsigned long i = 0UL; i < n.text.len; i = i + 1UL) {
                if (n.text.char_at(i) == (int)'`') { run = run + 1UL; if (run > longest) { longest = run; } } else { run = 0UL; }
            }
            struct String fence = String::repeat_str("`", longest + 1UL);
            out->push_str(&fence);
            int pad = n.text.starts_with("`") || n.text.ends_with("`");
            if (pad) { out->push_char(' '); }
            out->push_str(&n.text);
            if (pad) { out->push_char(' '); }
            out->push_str(&fence);
            fence.free();
        }
        else if (k == MD_LINK || k == MD_IMAGE) {
            if (k == MD_IMAGE) { out->push_char('!'); }
            out->push_char('[');
            md_md_children_inline_(n, out);
            out->push("](<");
            out->push_str(&n.text);
            out->push(">");
            if (n.title.len > 0UL) {
                out->push(" \"");
                for (unsigned long i = 0UL; i < n.title.len; i = i + 1UL) {
                    char c = (char)n.title.char_at(i);
                    if (c == '"' || c == '\\') { out->push_char('\\'); }
                    out->push_char(c);
                }
                out->push("\"");
            }
            out->push(")");
        }
        else if (k == MD_BREAK) { out->push("\\\n"); }
        else if (k == MD_HTML_INLINE) { out->push_str(&n.text); }
        else { md_md_children_inline_(n, out); }
    }
}

// Appends 'text' with every line after the first prefixed by 'prefix'
// (and the first by 'first').
static void md_md_prefixed_(struct String* out, const &String text, const &char first, const &char prefix) {
    unsafe {
        out->push(first);
        for (unsigned long i = 0UL; i < text.len; i = i + 1UL) {
            int c = text.char_at(i);
            out->push_char((char)c);
            if (c == (int)'\n' && i + 1UL < text.len) {
                if (text.char_at(i + 1UL) == (int)'\n') { struct String p = String::from(prefix); struct String t = p.trim_right(); out->push_str(&t); p.free(); t.free(); }
                else { out->push(prefix); }
            }
        }
    }
}

static void md_md_block_(const &MdNode n, struct String* out, int tight);

static void md_md_blocks_(const &MdNode n, struct String* out, int tight) {
    unsigned long cn = n.children.length();
    for (unsigned long i = 0UL; i < cn; i = i + 1UL) {
        &heap MdNode c;
        unsafe { c = *n.children.at(i); }
        if (c.kind == MD_CONTAINER_TITLE) { continue; }
        if (i > 0UL && !tight) { unsafe { out->push_char('\n'); } }
        md_md_block_(c, out, tight);
    }
}

static void md_md_block_(const &MdNode n, struct String* out, int tight) {
    unsafe {
        int k = n.kind;
        if (k == MD_HEADING) {
            for (int i = 0; i < n.level; i = i + 1) { out->push_char('#'); }
            out->push_char(' ');
            md_md_children_inline_(n, out);
            if (n.id.len > 0UL) { out->push(" {#"); out->push_str(&n.id); out->push("}"); }
            out->push_char('\n');
        } else if (k == MD_PARAGRAPH) {
            md_md_children_inline_(n, out);
            out->push_char('\n');
        } else if (k == MD_HR) { out->push("---\n"); }
        else if (k == MD_CODE_BLOCK) {
            out->push("```"); out->push_str(&n.info); out->push_char('\n');
            out->push_str(&n.text);
            out->push("```\n");
        } else if (k == MD_HTML_BLOCK) { out->push_str(&n.text); }
        else if (k == MD_BLOCKQUOTE) {
            struct String inner = String::new();
            md_md_blocks_(n, &inner, 0);
            md_md_prefixed_(out, &inner, "> ", "> ");
        } else if (k == MD_LIST) {
            unsigned long cn = n.children.length();
            for (unsigned long i = 0UL; i < cn; i = i + 1UL) {
                &heap MdNode item;
                item = *n.children.at(i);
                struct String marker = String::new();
                if (n.ordered) { marker.push_int((long long)(n.start + (int)i)); marker.push(". "); }
                else { marker.push("- "); }
                struct String pad = String::repeat_str(" ", marker.len);
                struct String inner = String::new();
                md_md_blocks_(item, &inner, n.tight);
                if (inner.len == 0UL) { inner.push_char('\n'); }
                if (i > 0UL && !n.tight) { out->push_char('\n'); }
                md_md_prefixed_(out, &inner, marker.as_ptr(), pad.as_ptr());
                marker.free(); pad.free(); inner.free();
            }
        } else if (k == MD_TABLE) {
            unsigned long rows = n.children.length();
            for (unsigned long r = 0UL; r < rows; r = r + 1UL) {
                &heap MdNode row;
                row = *n.children.at(r);
                out->push("|");
                for (unsigned long c = 0UL; c < row.children.length(); c = c + 1UL) {
                    &heap MdNode cell;
                    cell = *row.children.at(c);
                    struct String t = String::new();
                    md_md_children_inline_(cell, &t);
                    t.replace_all("|", "\\|");
                    out->push(" "); out->push_str(&t); out->push(" |");
                    t.free();
                }
                out->push_char('\n');
                if (r == 0UL) {
                    out->push("|");
                    for (unsigned long c = 0UL; c < row.children.length(); c = c + 1UL) {
                        int al = (*row.children.at(c))->align;
                        out->push(al == MD_ALIGN_CENTER ? ":---:|" : (al == MD_ALIGN_RIGHT ? "---:|" : (al == MD_ALIGN_LEFT ? ":---|" : "---|")));
                    }
                    out->push_char('\n');
                }
            }
        } else if (k == MD_CONTAINER) {
            out->push("::: "); out->push_str(&n.info);
            if (n.children.length() > 0UL && (*n.children.at(0UL))->kind == MD_CONTAINER_TITLE) {
                out->push_char(' ');
                md_md_children_inline_(*n.children.at(0UL), out);
            }
            out->push_char('\n');
            md_md_blocks_(n, out, 0);
            out->push(":::\n");
        } else {
            md_md_blocks_(n, out, tight);
        }
    }
}

struct String MdNode::render_markdown() const {
    struct String out = String::new();
    md_md_block_(self, &out, 0);
    return out;
}

} // namespace std
