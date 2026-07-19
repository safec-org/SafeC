// SafeC Standard Library — Markdown serialization implementation (see
// markdown.h for exact scope).
#pragma once
#include <std/serial/markdown.h>
#include <std/mem.sc>
#include <std/collections/vec.sc>
#include <std/collections/string.sc>

namespace std {

static struct MdNode* __md_new(int kind) {
    struct MdNode* n;
    unsafe { n = (struct MdNode*)alloc((unsigned long)sizeof(struct MdNode)); }
    unsafe {
        n->kind = kind;
        n->level = 0;
        n->ordered = 0;
        n->text = string_new();
        n->children = vec_new(8UL); // sizeof(struct MdNode*)
    }
    return n;
}

static void __md_add_child(struct MdNode* parent, struct MdNode* child) {
    unsafe { parent->children.push((const void*)&child); }
}

static int __md_is_blank(const char* s) {
    unsafe {
        unsigned long i = 0UL;
        while (s[i] != '\0') {
            if (s[i] != ' ' && s[i] != '\t' && s[i] != '\r') { return 0; }
            i = i + 1UL;
        }
    }
    return 1;
}

// ── inline parsing ───────────────────────────────────────────────────────────

static struct MdNode* __md_parse_inline(const char* text, unsigned long len) {
    struct MdNode* root = __md_new(MD_TEXT); // reused as a throwaway container; real root below
    unsafe { root->text.free(); }
    unsafe { root->kind = -1; } // sentinel: this is just a children-holder, not rendered itself
    unsafe { root->children = vec_new(8UL); }

    struct String pending = string_new();
    unsigned long i = 0UL;

    while (i < len) {
        char c; unsafe { c = text[i]; }
        int isBold = 0; int isItalic = 0;
        char marker = c;
        if ((c == '*' || c == '_')) {
            char c2 = '\0';
            unsafe { if (i + 1UL < len) { c2 = text[i + 1UL]; } }
            if (c2 == marker) { isBold = 1; } else { isItalic = 1; }
        }

        if (isBold) {
            unsigned long closeAt = len;
            unsigned long j = i + 2UL;
            while (j + 1UL < len) {
                char a; char b;
                unsafe { a = text[j]; b = text[j + 1UL]; }
                if (a == marker && b == marker) { closeAt = j; break; }
                j = j + 1UL;
            }
            if (closeAt < len) {
                if (pending.len > 0UL) {
                    struct MdNode* t = __md_new(MD_TEXT);
                    unsafe { t->text = pending; }
                    __md_add_child(root, t);
                    pending = string_new();
                }
                struct MdNode* b = __md_new(MD_BOLD);
                struct MdNode* inner = __md_new(MD_TEXT);
                unsafe { inner->text = string_from_n(text + i + 2UL, closeAt - (i + 2UL)); }
                __md_add_child(b, inner);
                __md_add_child(root, b);
                i = closeAt + 2UL;
                continue;
            }
        } else if (isItalic) {
            unsigned long closeAt = len;
            unsigned long j = i + 1UL;
            while (j < len) {
                char a; unsafe { a = text[j]; }
                if (a == marker) { closeAt = j; break; }
                j = j + 1UL;
            }
            if (closeAt < len && closeAt > i + 1UL) {
                if (pending.len > 0UL) {
                    struct MdNode* t = __md_new(MD_TEXT);
                    unsafe { t->text = pending; }
                    __md_add_child(root, t);
                    pending = string_new();
                }
                struct MdNode* it = __md_new(MD_ITALIC);
                struct MdNode* inner = __md_new(MD_TEXT);
                unsafe { inner->text = string_from_n(text + i + 1UL, closeAt - (i + 1UL)); }
                __md_add_child(it, inner);
                __md_add_child(root, it);
                i = closeAt + 1UL;
                continue;
            }
        } else if (c == '`') {
            unsigned long closeAt = len;
            unsigned long j = i + 1UL;
            while (j < len) {
                char a; unsafe { a = text[j]; }
                if (a == '`') { closeAt = j; break; }
                j = j + 1UL;
            }
            if (closeAt < len) {
                if (pending.len > 0UL) {
                    struct MdNode* t = __md_new(MD_TEXT);
                    unsafe { t->text = pending; }
                    __md_add_child(root, t);
                    pending = string_new();
                }
                struct MdNode* code = __md_new(MD_CODE_SPAN);
                unsafe { code->text = string_from_n(text + i + 1UL, closeAt - (i + 1UL)); }
                __md_add_child(root, code);
                i = closeAt + 1UL;
                continue;
            }
        } else if (c == '[') {
            unsigned long closeBracket = len;
            unsigned long j = i + 1UL;
            while (j < len) {
                char a; unsafe { a = text[j]; }
                if (a == ']') { closeBracket = j; break; }
                j = j + 1UL;
            }
            int hasParen = 0; unsigned long closeParen = len;
            unsafe {
                if (closeBracket < len && closeBracket + 1UL < len && text[closeBracket + 1UL] == '(') {
                    unsigned long k = closeBracket + 2UL;
                    while (k < len) {
                        if (text[k] == ')') { closeParen = k; hasParen = 1; break; }
                        k = k + 1UL;
                    }
                }
            }
            if (hasParen) {
                if (pending.len > 0UL) {
                    struct MdNode* t = __md_new(MD_TEXT);
                    unsafe { t->text = pending; }
                    __md_add_child(root, t);
                    pending = string_new();
                }
                struct MdNode* link = __md_new(MD_LINK);
                unsafe { link->text = string_from_n(text + closeBracket + 2UL, closeParen - (closeBracket + 2UL)); }
                struct MdNode* labelNode = __md_new(MD_TEXT);
                unsafe { labelNode->text = string_from_n(text + i + 1UL, closeBracket - (i + 1UL)); }
                __md_add_child(link, labelNode);
                __md_add_child(root, link);
                i = closeParen + 1UL;
                continue;
            }
        }

        unsafe { pending.push_char(c); }
        i = i + 1UL;
    }
    if (pending.len > 0UL) {
        struct MdNode* t = __md_new(MD_TEXT);
        unsafe { t->text = pending; }
        __md_add_child(root, t);
    } else {
        unsafe { pending.free(); }
    }
    return root;
}

// ── block parsing ────────────────────────────────────────────────────────────

static int __md_starts_with(const char* s, const char* prefix) {
    unsafe {
        unsigned long i = 0UL;
        while (prefix[i] != '\0') {
            if (s[i] != prefix[i]) { return 0; }
            i = i + 1UL;
        }
        return 1;
    }
}

static int __md_heading_level(const char* line) {
    int level = 0;
    unsafe {
        while (line[level] == '#' && level < 6) { level = level + 1; }
        if (level > 0 && line[level] == ' ') { return level; }
    }
    return 0;
}

static int __md_is_hr(const char* line) {
    char first = '\0';
    int count = 0;
    unsafe {
        unsigned long i = 0UL;
        while (line[i] != '\0') {
            char c = line[i];
            if (c == ' ' || c == '\t' || c == '\r') { i = i + 1UL; continue; }
            if (first == '\0') { first = c; }
            if (c != first) { return 0; }
            if (c != '-' && c != '*' && c != '_') { return 0; }
            count = count + 1;
            i = i + 1UL;
        }
    }
    return count >= 3 ? 1 : 0;
}

static int __md_is_unordered_item(const char* line) {
    unsafe {
        if ((line[0] == '-' || line[0] == '*' || line[0] == '+') && line[1] == ' ') { return 1; }
    }
    return 0;
}

static int __md_is_ordered_item(const char* line, unsigned long* markerLen) {
    unsafe {
        unsigned long i = 0UL;
        while (line[i] >= '0' && line[i] <= '9') { i = i + 1UL; }
        if (i > 0UL && line[i] == '.' && line[i + 1UL] == ' ') {
            *markerLen = i + 2UL;
            return 1;
        }
    }
    return 0;
}

struct MdNode* md_parse(const char* text) {
    struct MdNode* doc = __md_new(MD_DOC);

    // Split into lines.
    struct Vec lines = vec_new((unsigned long)sizeof(struct String));
    unsigned long len = 0UL;
    unsafe { while (text[len] != '\0') { len = len + 1UL; } }
    unsigned long lineStart = 0UL;
    unsigned long i = 0UL;
    while (i <= len) {
        int atEnd; unsafe { atEnd = (i == len) ? 1 : 0; }
        char c; unsafe { c = atEnd ? '\n' : text[i]; }
        if (c == '\n') {
            struct String line;
            unsafe { line = string_from_n(text + lineStart, i - lineStart); }
            unsafe { lines.push((const void*)&line); }
            lineStart = i + 1UL;
        }
        if (atEnd) { break; }
        i = i + 1UL;
    }

    unsigned long n; unsafe { n = lines.length(); }
    unsigned long li = 0UL;
    while (li < n) {
        struct String* linePtr;
        unsafe { linePtr = (struct String*)lines.get_raw(li); }
        const char* line; unsafe { line = (const char*)linePtr->data; }

        if (__md_is_blank(line)) { li = li + 1UL; continue; }

        int hlevel = __md_heading_level(line);
        if (hlevel > 0) {
            struct MdNode* h = __md_new(MD_HEADING);
            unsafe { h->level = hlevel; }
            const char* rest; unsafe { rest = line + hlevel + 1; }
            unsigned long restLen; unsafe { restLen = 0UL; while (rest[restLen] != '\0') { restLen = restLen + 1UL; } }
            struct MdNode* inline_ = __md_parse_inline(rest, restLen);
            unsafe {
                unsigned long cn = inline_->children.length();
                unsigned long ci = 0UL;
                while (ci < cn) {
                    struct MdNode** slot = (struct MdNode**)inline_->children.get_raw(ci);
                    __md_add_child(h, *slot);
                    ci = ci + 1UL;
                }
                inline_->children.free();
                dealloc((void*)inline_);
            }
            __md_add_child(doc, h);
            li = li + 1UL;
            continue;
        }

        if (__md_is_hr(line)) {
            __md_add_child(doc, __md_new(MD_HR));
            li = li + 1UL;
            continue;
        }

        int isFence; unsafe { isFence = __md_starts_with(line, "```"); }
        if (isFence) {
            struct String code = string_new();
            li = li + 1UL;
            while (li < n) {
                struct String* lp2; unsafe { lp2 = (struct String*)lines.get_raw(li); }
                const char* l2; unsafe { l2 = (const char*)lp2->data; }
                if (__md_starts_with(l2, "```")) { li = li + 1UL; break; }
                unsafe { code.push((const char*)lp2->data); code.push_char('\n'); }
                li = li + 1UL;
            }
            struct MdNode* cb = __md_new(MD_CODE_BLOCK);
            unsafe { cb->text = code; }
            __md_add_child(doc, cb);
            continue;
        }

        int lineIsQuote; unsafe { lineIsQuote = (line[0] == '>') ? 1 : 0; }
        if (lineIsQuote) {
            struct String quote = string_new();
            while (li < n) {
                struct String* lp3; unsafe { lp3 = (struct String*)lines.get_raw(li); }
                const char* l3; unsafe { l3 = (const char*)lp3->data; }
                int l3IsQuote; unsafe { l3IsQuote = (l3[0] == '>') ? 1 : 0; }
                if (!l3IsQuote) { break; }
                const char* body; unsafe { body = (l3[1] == ' ') ? l3 + 2 : l3 + 1; }
                if (quote.len > 0UL) { unsafe { quote.push_char(' '); } }
                unsafe { quote.push(body); }
                li = li + 1UL;
            }
            struct MdNode* bq = __md_new(MD_BLOCKQUOTE);
            struct MdNode* para = __md_new(MD_PARAGRAPH);
            unsigned long qlen; unsafe { qlen = quote.len; }
            struct MdNode* inl; unsafe { inl = __md_parse_inline((const char*)quote.data, qlen); }
            unsafe {
                unsigned long cn = inl->children.length(); unsigned long ci = 0UL;
                while (ci < cn) {
                    struct MdNode** slot = (struct MdNode**)inl->children.get_raw(ci);
                    __md_add_child(para, *slot);
                    ci = ci + 1UL;
                }
                inl->children.free(); dealloc((void*)inl);
                quote.free();
            }
            __md_add_child(bq, para);
            __md_add_child(doc, bq);
            continue;
        }

        unsigned long orderedMarkerLenProbe = 0UL;
        if (__md_is_unordered_item(line) || __md_is_ordered_item(line, &orderedMarkerLenProbe)) {
            int ordered = __md_is_unordered_item(line) ? 0 : 1;
            struct MdNode* list = __md_new(MD_LIST);
            unsafe { list->ordered = ordered; }
            while (li < n) {
                struct String* lp4; unsafe { lp4 = (struct String*)lines.get_raw(li); }
                const char* l4; unsafe { l4 = (const char*)lp4->data; }
                if (__md_is_blank(l4)) { break; }
                unsigned long markerLen = 0UL;
                int itemOk = 0;
                const char* body = l4;
                if (!ordered && __md_is_unordered_item(l4)) { itemOk = 1; unsafe { body = l4 + 2; } }
                else if (ordered && __md_is_ordered_item(l4, &markerLen)) { itemOk = 1; unsafe { body = l4 + markerLen; } }
                if (!itemOk) { break; }
                unsigned long blen = 0UL; unsafe { while (body[blen] != '\0') { blen = blen + 1UL; } }
                struct MdNode* item = __md_new(MD_LIST_ITEM);
                struct MdNode* inl2 = __md_parse_inline(body, blen);
                unsafe {
                    unsigned long cn = inl2->children.length(); unsigned long ci = 0UL;
                    while (ci < cn) {
                        struct MdNode** slot = (struct MdNode**)inl2->children.get_raw(ci);
                        __md_add_child(item, *slot);
                        ci = ci + 1UL;
                    }
                    inl2->children.free(); dealloc((void*)inl2);
                }
                __md_add_child(list, item);
                li = li + 1UL;
            }
            __md_add_child(doc, list);
            continue;
        }

        // Paragraph: join consecutive non-blank, non-special lines.
        struct String para = string_new();
        while (li < n) {
            struct String* lp5; unsafe { lp5 = (struct String*)lines.get_raw(li); }
            const char* l5; unsafe { l5 = (const char*)lp5->data; }
            if (__md_is_blank(l5) || __md_heading_level(l5) > 0 || __md_is_hr(l5) ||
                __md_is_unordered_item(l5)) { break; }
            unsigned long dummyLen;
            if (__md_is_ordered_item(l5, &dummyLen)) { break; }
            int fenceHere; unsafe { fenceHere = __md_starts_with(l5, "```"); }
            int l5IsQuote; unsafe { l5IsQuote = (l5[0] == '>') ? 1 : 0; }
            if (fenceHere || l5IsQuote) { break; }
            if (para.len > 0UL) { unsafe { para.push_char(' '); } }
            unsafe { para.push(l5); }
            li = li + 1UL;
        }
        struct MdNode* p = __md_new(MD_PARAGRAPH);
        unsigned long plen; unsafe { plen = para.len; }
        struct MdNode* inl3; unsafe { inl3 = __md_parse_inline((const char*)para.data, plen); }
        unsafe {
            unsigned long cn = inl3->children.length(); unsigned long ci = 0UL;
            while (ci < cn) {
                struct MdNode** slot = (struct MdNode**)inl3->children.get_raw(ci);
                __md_add_child(p, *slot);
                ci = ci + 1UL;
            }
            inl3->children.free(); dealloc((void*)inl3);
            para.free();
        }
        __md_add_child(doc, p);
    }

    unsigned long ln; unsafe { ln = lines.length(); }
    unsigned long fi = 0UL;
    while (fi < ln) {
        struct String* lp; unsafe { lp = (struct String*)lines.get_raw(fi); }
        unsafe { lp->free(); }
        fi = fi + 1UL;
    }
    unsafe { lines.free(); }

    return doc;
}

// ── HTML rendering ───────────────────────────────────────────────────────────

static void __md_escape_html(struct String* out, const char* s) {
    unsafe {
        unsigned long i = 0UL;
        while (s[i] != '\0') {
            char c = s[i];
            if (c == '&') { out->push("&amp;"); }
            else if (c == '<') { out->push("&lt;"); }
            else if (c == '>') { out->push("&gt;"); }
            else if (c == '"') { out->push("&quot;"); }
            else { out->push_char(c); }
            i = i + 1UL;
        }
    }
}

static void __md_render_inline_html(const struct MdNode* n, struct String* out) {
    int kind; unsafe { kind = n->kind; }
    if (kind == MD_TEXT) {
        unsafe { __md_escape_html(out, (const char*)n->text.data); }
        return;
    }
    if (kind == MD_CODE_SPAN) {
        unsafe { out->push("<code>"); }
        unsafe { __md_escape_html(out, (const char*)n->text.data); }
        unsafe { out->push("</code>"); }
        return;
    }
    if (kind == MD_BOLD || kind == MD_ITALIC || kind == MD_LINK) {
        const char* openTag; const char* closeTag;
        if (kind == MD_BOLD) { openTag = "<strong>"; closeTag = "</strong>"; }
        else if (kind == MD_ITALIC) { openTag = "<em>"; closeTag = "</em>"; }
        else { openTag = (const char*)0; closeTag = "</a>"; }
        if (kind == MD_LINK) {
            unsafe { out->push("<a href=\""); __md_escape_html(out, (const char*)n->text.data); out->push("\">"); }
        } else {
            unsafe { out->push(openTag); }
        }
        unsigned long cn; unsafe { cn = n->children.length(); }
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { __md_render_inline_html(*slot, out); }
            ci = ci + 1UL;
        }
        unsafe { out->push(closeTag); }
    }
}

static void __md_render_block_html(const struct MdNode* n, struct String* out) {
    int kind; unsafe { kind = n->kind; }
    unsigned long cn; unsafe { cn = n->children.length(); }

    if (kind == MD_DOC) {
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { __md_render_block_html(*slot, out); }
            ci = ci + 1UL;
        }
        return;
    }
    if (kind == MD_HEADING) {
        int level; unsafe { level = n->level; }
        char tag[5]; unsafe { tag[0]='h'; tag[1]=(char)('0'+level); tag[2]='\0'; }
        unsafe { out->push("<"); out->push(tag); out->push(">"); }
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { __md_render_inline_html(*slot, out); }
            ci = ci + 1UL;
        }
        unsafe { out->push("</"); out->push(tag); out->push(">\n"); }
        return;
    }
    if (kind == MD_PARAGRAPH) {
        unsafe { out->push("<p>"); }
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { __md_render_inline_html(*slot, out); }
            ci = ci + 1UL;
        }
        unsafe { out->push("</p>\n"); }
        return;
    }
    if (kind == MD_LIST) {
        int ordered; unsafe { ordered = n->ordered; }
        unsafe { out->push(ordered ? "<ol>\n" : "<ul>\n"); }
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { out->push("<li>"); }
            unsigned long icn; unsafe { icn = (*slot)->children.length(); }
            unsigned long ii = 0UL;
            while (ii < icn) {
                struct MdNode** islot; unsafe { islot = (struct MdNode**)(*slot)->children.get_raw(ii); }
                unsafe { __md_render_inline_html(*islot, out); }
                ii = ii + 1UL;
            }
            unsafe { out->push("</li>\n"); }
            ci = ci + 1UL;
        }
        unsafe { out->push(ordered ? "</ol>\n" : "</ul>\n"); }
        return;
    }
    if (kind == MD_CODE_BLOCK) {
        unsafe { out->push("<pre><code>"); }
        unsafe { __md_escape_html(out, (const char*)n->text.data); }
        unsafe { out->push("</code></pre>\n"); }
        return;
    }
    if (kind == MD_BLOCKQUOTE) {
        unsafe { out->push("<blockquote>\n"); }
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { __md_render_block_html(*slot, out); }
            ci = ci + 1UL;
        }
        unsafe { out->push("</blockquote>\n"); }
        return;
    }
    if (kind == MD_HR) {
        unsafe { out->push("<hr/>\n"); }
        return;
    }
}

struct String md_render_html(const struct MdNode* doc) {
    struct String out = string_new();
    __md_render_block_html(doc, &out);
    return out;
}

// ── Markdown round-trip rendering ───────────────────────────────────────────

static void __md_render_inline_md(const struct MdNode* n, struct String* out) {
    int kind; unsafe { kind = n->kind; }
    if (kind == MD_TEXT) { unsafe { out->push((const char*)n->text.data); } return; }
    if (kind == MD_CODE_SPAN) {
        unsafe { out->push_char('`'); out->push((const char*)n->text.data); out->push_char('`'); }
        return;
    }
    unsigned long cn; unsafe { cn = n->children.length(); }
    if (kind == MD_BOLD) { unsafe { out->push("**"); } }
    else if (kind == MD_ITALIC) { unsafe { out->push_char('*'); } }
    else if (kind == MD_LINK) { unsafe { out->push_char('['); } }
    unsigned long ci = 0UL;
    while (ci < cn) {
        struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
        unsafe { __md_render_inline_md(*slot, out); }
        ci = ci + 1UL;
    }
    if (kind == MD_BOLD) { unsafe { out->push("**"); } }
    else if (kind == MD_ITALIC) { unsafe { out->push_char('*'); } }
    else if (kind == MD_LINK) {
        unsafe { out->push("]("); out->push((const char*)n->text.data); out->push_char(')'); }
    }
}

static void __md_render_block_md(const struct MdNode* n, struct String* out) {
    int kind; unsafe { kind = n->kind; }
    unsigned long cn; unsafe { cn = n->children.length(); }

    if (kind == MD_DOC) {
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { __md_render_block_md(*slot, out); }
            ci = ci + 1UL;
        }
        return;
    }
    if (kind == MD_HEADING) {
        int level; unsafe { level = n->level; }
        int k = 0; while (k < level) { unsafe { out->push_char('#'); } k = k + 1; }
        unsafe { out->push_char(' '); }
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { __md_render_inline_md(*slot, out); }
            ci = ci + 1UL;
        }
        unsafe { out->push("\n\n"); }
        return;
    }
    if (kind == MD_PARAGRAPH) {
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            unsafe { __md_render_inline_md(*slot, out); }
            ci = ci + 1UL;
        }
        unsafe { out->push("\n\n"); }
        return;
    }
    if (kind == MD_LIST) {
        int ordered; unsafe { ordered = n->ordered; }
        unsigned long ci = 0UL;
        int idx = 1;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            if (ordered) {
                unsafe { out->push_int((long long)idx); out->push(". "); }
            } else {
                unsafe { out->push("- "); }
            }
            unsigned long icn; unsafe { icn = (*slot)->children.length(); }
            unsigned long ii = 0UL;
            while (ii < icn) {
                struct MdNode** islot; unsafe { islot = (struct MdNode**)(*slot)->children.get_raw(ii); }
                unsafe { __md_render_inline_md(*islot, out); }
                ii = ii + 1UL;
            }
            unsafe { out->push_char('\n'); }
            idx = idx + 1;
            ci = ci + 1UL;
        }
        unsafe { out->push_char('\n'); }
        return;
    }
    if (kind == MD_CODE_BLOCK) {
        unsafe { out->push("```\n"); out->push((const char*)n->text.data); out->push("```\n\n"); }
        return;
    }
    if (kind == MD_BLOCKQUOTE) {
        unsafe { out->push("> "); }
        unsigned long ci = 0UL;
        while (ci < cn) {
            struct MdNode** slot; unsafe { slot = (struct MdNode**)n->children.get_raw(ci); }
            struct MdNode* para; unsafe { para = *slot; }
            unsigned long pcn; unsafe { pcn = para->children.length(); }
            unsigned long pi = 0UL;
            while (pi < pcn) {
                struct MdNode** pslot; unsafe { pslot = (struct MdNode**)para->children.get_raw(pi); }
                unsafe { __md_render_inline_md(*pslot, out); }
                pi = pi + 1UL;
            }
            ci = ci + 1UL;
        }
        unsafe { out->push("\n\n"); }
        return;
    }
    if (kind == MD_HR) {
        unsafe { out->push("---\n\n"); }
        return;
    }
}

struct String md_render_markdown(const struct MdNode* doc) {
    struct String out = string_new();
    __md_render_block_md(doc, &out);
    return out;
}

// ── lifecycle ─────────────────────────────────────────────────────────────────

void md_free(struct MdNode* node) {
    unsigned long cn; unsafe { cn = node->children.length(); }
    unsigned long ci = 0UL;
    while (ci < cn) {
        struct MdNode** slot; unsafe { slot = (struct MdNode**)node->children.get_raw(ci); }
        struct MdNode* childNode; unsafe { childNode = *slot; }
        md_free(childNode);
        ci = ci + 1UL;
    }
    unsafe {
        node->children.free();
        node->text.free();
        dealloc((void*)node);
    }
}

} // namespace std
