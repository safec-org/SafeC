#pragma once
// SafeC Standard Library — Markdown serialization.
//
// Unlike json.h/xml.h/html.h, Markdown doesn't map onto the generic
// Value tree (std/serial/value.h) — it's document markup (headings,
// paragraphs, lists, emphasis), not key/value data — so this is its own
// small AST, parsed from a CommonMark-ish subset and rendered to either
// HTML (the practically useful direction — pairs naturally with
// std/http's response bodies or std/gui/scx-adjacent server rendering)
// or back to canonical Markdown text.
//
// Supported block syntax: # through ###### headings, paragraphs,
// unordered lists (-, *, or + markers) and ordered lists (1. 2. ...),
// fenced code blocks (```), blockquotes (>), and horizontal rules
// (---/***/___ on their own line). Lists are flat (no nested sub-lists).
// Supported inline syntax: **bold**/__bold__, *italic*/_italic_,
// `inline code`, and [link text](url). No tables, no images, no raw HTML
// passthrough, no reference-style links.

#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

#define MD_DOC        0
#define MD_HEADING    1
#define MD_PARAGRAPH  2
#define MD_LIST       3  // container; 'ordered' distinguishes -/* vs 1.
#define MD_LIST_ITEM  4
#define MD_CODE_BLOCK 5
#define MD_BLOCKQUOTE 6
#define MD_HR         7
#define MD_TEXT       8  // inline: plain text run (in 'text')
#define MD_BOLD       9  // inline: has children
#define MD_ITALIC     10 // inline: has children
#define MD_CODE_SPAN  11 // inline: plain text (in 'text', not re-parsed)
#define MD_LINK       12 // inline: has children (link text); 'text' holds the URL

struct MdNode {
    int kind;
    int level;          // MD_HEADING: 1-6
    int ordered;         // MD_LIST: 1 = "1. 2. ...", 0 = "- " / "* " / "+ "
    struct String text;  // MD_TEXT/MD_CODE_SPAN/MD_CODE_BLOCK content; MD_LINK's URL
    struct Vec children;  // Vec<struct MdNode*> — block children, or inline children
};

// Parses 'text' into a heap-allocated MD_DOC root node (see md_free()).
&heap MdNode md_parse(const char* text);

// Renders 'doc' (or any node) to an HTML fragment (headings become
// <h1>-<h6>, paragraphs <p>, lists <ul>/<ol><li>, code blocks <pre><code>,
// blockquotes <blockquote>, inline runs their usual <strong>/<em>/<code>/
// <a href>). Text content is HTML-escaped; only md_render_html output
// itself is trusted markup.
struct String md_render_html(const &heap MdNode doc);

// Renders 'doc' back to canonical Markdown text (a round-trip — not
// guaranteed byte-identical to arbitrary input, since e.g. '*bold*' and
// '_bold_' both parse to the same MD_BOLD node and always render as
// '**bold**').
struct String md_render_markdown(const &heap MdNode doc);

// Recursively frees 'node' and its whole subtree.
void md_free(&heap MdNode node);

} // namespace std
