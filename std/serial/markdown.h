#pragma once
#include <std/erased.h>
// SafeC Standard Library — Markdown.
//
// Unlike json.h/xml.h/html.h, Markdown doesn't map onto the generic
// Value tree (std/serial/value.h) — it's document markup (headings,
// paragraphs, lists, emphasis), not key/value data — so this is its own
// AST, parsed from CommonMark (plus opt-in GitHub-flavored and
// documentation-site extensions) and rendered to HTML or back to Markdown.
//
// Blocks: ATX and setext headings, paragraphs, thematic breaks, block
// quotes, bullet and ordered lists (nested, tight or loose, items holding
// any blocks), fenced code (``` / ~~~, with an info string) and indented
// code. Inlines: emphasis and strong emphasis with CommonMark's
// delimiter rules (no intraword '_'), code spans, backslash escapes,
// entity and numeric character references, links and images (inline
// form, with titles; link text holds any inlines), <scheme:...> and
// <email> autolinks, hard line breaks. Not supported: reference-style
// links ('[x][ref]' / '[ref]: url') and footnotes.
//
// Opt-in through MdOptions (MdNode::parse_opts):
//   tables         GFM tables, with column alignment ('| a | b |' + '|:--|--:|')
//   html           raw HTML blocks and inline tags pass through unescaped
//                  (off by default: untrusted input stays inert)
//   containers     '::: kind [title]' ... ':::' blocks (tip, warning,
//                  details, ... — any word), nestable
//   heading_ids    '## Title {#custom-id}', and an id for every other
//                  heading from md_slugify (deduplicated with -1, -2, ...)
//   strikethrough  '~~text~~'
//   autolinks      bare 'https://...', 'http://...' and 'www....' URLs
// MdOptions::commonmark() turns all of them off (what MdNode::parse uses);
// MdOptions::gfm() turns on tables, strikethrough and autolinks.
//
//   struct MdOptions o = MdOptions::gfm();
//   o.html = 1; o.containers = 1; o.heading_ids = 1;
//   &heap MdNode doc = MdNode::parse_opts(text, &o);
//   struct String html = doc.render_html();
//   doc.free();
//
// md_render_html_with renders with a hook that can replace or wrap any
// node's HTML (rewrite link targets, add heading anchors, highlight code).

#include <std/collections/vec.h>
#include <std/collections/string.h>

namespace std {

#define MD_DOC             0
#define MD_HEADING         1  // level 1-6; id (with heading_ids)
#define MD_PARAGRAPH       2
#define MD_LIST            3  // 'ordered' 1 = "1." / "1)", 0 = "-" "*" "+"; 'start'; 'tight'
#define MD_LIST_ITEM       4
#define MD_CODE_BLOCK      5  // 'text' = the code (ends with '\n'); 'info' = the fence's info string
#define MD_BLOCKQUOTE      6
#define MD_HR              7
#define MD_TEXT            8  // inline: plain text (in 'text', references decoded)
#define MD_BOLD            9  // inline: has children
#define MD_ITALIC          10 // inline: has children
#define MD_CODE_SPAN       11 // inline: 'text'
#define MD_LINK            12 // inline: children = link text; 'text' = URL; 'title'
#define MD_TABLE           13 // children: rows (the first 'header' row is the thead)
#define MD_TABLE_ROW       14 // children: cells; 'header' 1 for the header row
#define MD_TABLE_CELL      15 // inline children; 'align' (MD_ALIGN_*); 'header'
#define MD_CONTAINER       16 // 'info' = kind ("tip"); first child MD_CONTAINER_TITLE when a title was given
#define MD_HTML_BLOCK      17 // 'text' = raw HTML (html option)
#define MD_HTML_INLINE     18 // 'text' = raw HTML tag/comment (html option)
#define MD_IMAGE           19 // inline: children = alt text; 'text' = src; 'title'
#define MD_STRIKE          20 // inline: has children (strikethrough option)
#define MD_BREAK           21 // inline: hard line break
#define MD_CONTAINER_TITLE 22 // inline children: a container's title

#define MD_ALIGN_NONE   0
#define MD_ALIGN_LEFT   1
#define MD_ALIGN_CENTER 2
#define MD_ALIGN_RIGHT  3

struct MdOptions {
    int tables;
    int html;
    int containers;
    int heading_ids;
    int strikethrough;
    int autolinks;

    static struct MdOptions commonmark();   // everything off
    static struct MdOptions gfm();          // tables, strikethrough, autolinks
};

struct MdNode {
    int kind;
    int level;           // MD_HEADING: 1-6
    int ordered;         // MD_LIST: 1 = ordered, 0 = bullet
    int start;           // MD_LIST (ordered): the first item's number
    int tight;           // MD_LIST: 1 if no blank lines separate items/blocks (no <p> in items)
    int align;           // MD_TABLE_CELL: MD_ALIGN_*
    int header;          // MD_TABLE_ROW / MD_TABLE_CELL: part of the header row
    struct String text;  // MD_TEXT/MD_CODE_SPAN/MD_CODE_BLOCK/MD_HTML_*: content; MD_LINK/MD_IMAGE: URL
    struct String info;  // MD_CODE_BLOCK: info string ("c", "bash"); MD_CONTAINER: kind
    struct String id;    // MD_HEADING: id (heading_ids option), else empty
    struct String title; // MD_LINK/MD_IMAGE: title attribute
    struct Vec<&heap MdNode> children;  // block children, or inline children

    // Parses 'text' with every extension off (MdOptions::commonmark())
    // into a heap-allocated MD_DOC root node (release it with doc.free()).
    static &heap MdNode parse(const &char text);

    // Parses 'text' with the given extensions.
    static &heap MdNode parse_opts(const &char text, const &MdOptions opts);

    // Renders this node (usually the MD_DOC root) to an HTML fragment.
    // Text is HTML-escaped; only raw HTML nodes (html option) pass through.
    // Headings carry their id, code blocks class="language-<info word>",
    // containers render as <div class="custom-block KIND"> with a
    // <p class="custom-block-title"> (the title, or KIND upper-cased), and
    // 'details' containers as <details class="custom-block details"> with
    // a <summary>.
    struct String render_html() const;

    // Renders this node back to Markdown text (a round-trip — not
    // byte-identical to arbitrary input: '*x*' and '_x_' both come back as
    // '*x*', every list marker as '-' or 'N.').
    struct String render_markdown() const;

    // Appends the node's text content — every MD_TEXT/MD_CODE_SPAN/code
    // block below it, without markup — to 'out' (for outlines, search
    // indexes, slugs). Hard and soft breaks become spaces.
    void plain_text(&String out) const;

    // Recursively frees this node's whole subtree and the node itself
    // (every MdNode is heap-allocated by MdNode::parse) — don't use it after.
    void free();
};

// Appends the GitHub/VitePress-style slug of 'text' to 'out': lower-cased;
// every run of whitespace, ASCII punctuation and control characters
// becomes one '-'; no leading/trailing '-'; a leading digit gets a '_'
// prefix. Non-ASCII characters are kept.
void md_slugify(const &char text, &String out);

// ── Rendering with a hook ────────────────────────────────────────────────────
// 'hook' is called for every node before its default HTML is written
// ('entering' = 1). It returns:
//   0  — render the node normally;
//   1  — the hook wrote the node's whole HTML itself (children included);
//   2  — the hook wrote the node's opening HTML: its children are rendered
//        normally, then the hook is called again with 'entering' = 0 and
//        returns 1 if it wrote the closing HTML, 0 for the default one.
// E.g. a link hook rewrites the href on entry and returns 2; a code-block
// hook highlights node.text and returns 1.

// The hook and its context, erased (see std/erased.h).
struct MdHookRaw_ {
    ?fn int(?&Erased, const &MdNode, int, &String) fn_;
    ?&Erased ctx;
};
struct String md_render_html_raw_(const &MdNode node, const &MdHookRaw_ hook);

generic<C>
struct String md_render_html_with(const &MdNode node, fn int(?&C, const &MdNode, int, &String) hook, ?&C ctx) {
    struct MdHookRaw_ h;
    unsafe {
        h.fn_ = (?fn int(?&Erased, const &MdNode, int, &String))hook;
        h.ctx = (?&Erased)ctx;
    }
    return md_render_html_raw_(node, &h);
}

} // namespace std
