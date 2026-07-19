#include "ScxTranspiler.h"
#include <cctype>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace safeguard {
namespace {

int lineOf(const std::string& s, size_t pos) {
    int line = 1;
    for (size_t i = 0; i < pos && i < s.size(); ++i)
        if (s[i] == '\n') ++line;
    return line;
}

[[noreturn]] void fail(const std::string& filename, const std::string& src,
                        size_t pos, const std::string& msg) {
    throw std::runtime_error(filename + ":" + std::to_string(lineOf(src, pos)) +
                              ": scx: " + msg);
}

bool isIdentStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool isIdentChar(char c) {
    return std::isalnum((unsigned char)c) || c == '_' || c == '-';
}
bool isWs(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

// ── markup AST ───────────────────────────────────────────────────────────────

enum class NodeKind { Text, Interp, RawInterp, Element };

struct Attr {
    std::string name;
    bool        isExpr = false; // {expr} vs "literal"
    std::string value;          // literal text, or expr source (no braces)
};

struct Node {
    NodeKind                     kind;
    std::string                  text;    // Text
    std::string                  exprSrc; // Interp / RawInterp
    std::string                  tag;     // Element
    std::vector<Attr>            attrs;   // Element
    std::vector<std::unique_ptr<Node>> children; // Element
};
using NodePtr = std::unique_ptr<Node>;

// ── markup parser ────────────────────────────────────────────────────────────
// Operates directly on the full file source starting at an already-located
// '<'; advances 'p' past the matched element's closing '>' on return.

class MarkupParser {
public:
    MarkupParser(const std::string& src, const std::string& filename)
        : s_(src), file_(filename) {}

    NodePtr parseElement(size_t& p) {
        expect(p, '<');
        std::string tag = readIdent(p);
        auto node   = std::make_unique<Node>();
        node->kind  = NodeKind::Element;
        node->tag   = tag;
        skipWs(p);

        while (p < s_.size() && s_[p] != '>' && !(s_[p] == '/' && peek(p, 1) == '>')) {
            Attr a;
            a.name = readIdent(p);
            skipWs(p);
            if (p < s_.size() && s_[p] == '=') {
                ++p;
                skipWs(p);
                if (p < s_.size() && s_[p] == '"') {
                    a.isExpr = false;
                    a.value  = readQuoted(p);
                } else if (p < s_.size() && s_[p] == '{') {
                    a.isExpr = true;
                    a.value  = readBraced(p);
                } else {
                    fail(file_, s_, p,
                         "expected '\"' or '{' after '=' in attribute '" + a.name + "'");
                }
            } else {
                a.isExpr = false;
                a.value  = "";
            }
            node->attrs.push_back(std::move(a));
            skipWs(p);
        }

        if (p < s_.size() && s_[p] == '/' && peek(p, 1) == '>') {
            p += 2;
            return node; // self-closing, no children
        }
        expect(p, '>');

        for (;;) {
            if (p >= s_.size()) fail(file_, s_, p, "unterminated element <" + tag + ">");
            if (s_[p] == '<' && peek(p, 1) == '/') {
                size_t q = p + 2;
                std::string closeTag = readIdent(q);
                skipWs(q);
                if (q >= s_.size() || s_[q] != '>')
                    fail(file_, s_, q, "malformed closing tag for <" + tag + ">");
                ++q;
                if (closeTag != tag)
                    fail(file_, s_, p, "mismatched closing tag: expected </" + tag +
                                            ">, got </" + closeTag + ">");
                p = q;
                break;
            }
            if (s_[p] == '<') {
                node->children.push_back(parseElement(p));
            } else if (s_[p] == '{') {
                node->children.push_back(parseInterp(p));
            } else {
                node->children.push_back(parseText(p));
            }
        }
        return node;
    }

private:
    const std::string& s_;
    const std::string& file_;

    char peek(size_t p, int off) const {
        return p + off < s_.size() ? s_[p + off] : '\0';
    }
    void expect(size_t& p, char c) {
        if (p >= s_.size() || s_[p] != c)
            fail(file_, s_, p, std::string("expected '") + c + "'");
        ++p;
    }
    void skipWs(size_t& p) { while (p < s_.size() && isWs(s_[p])) ++p; }

    std::string readIdent(size_t& p) {
        size_t start = p;
        if (p >= s_.size() || !isIdentStart(s_[p])) fail(file_, s_, p, "expected identifier");
        while (p < s_.size() && isIdentChar(s_[p])) ++p;
        return s_.substr(start, p - start);
    }

    std::string readQuoted(size_t& p) {
        expect(p, '"');
        size_t start = p;
        while (p < s_.size() && s_[p] != '"') {
            if (s_[p] == '\\') ++p;
            ++p;
        }
        if (p >= s_.size()) fail(file_, s_, start, "unterminated attribute string");
        std::string v = s_.substr(start, p - start);
        ++p; // closing quote
        return v;
    }

    // Reads a balanced {...} (respecting nested braces and string/char
    // literals) and returns the inner text, with 'p' left just past the
    // closing '}'.
    std::string readBraced(size_t& p) {
        size_t open = p;
        expect(p, '{');
        size_t start = p;
        int depth = 1;
        while (p < s_.size() && depth > 0) {
            char c = s_[p];
            if (c == '"' || c == '\'') {
                char q = c;
                ++p;
                while (p < s_.size() && s_[p] != q) {
                    if (s_[p] == '\\') ++p;
                    ++p;
                }
                if (p < s_.size()) ++p;
                continue;
            }
            if (c == '{') ++depth;
            else if (c == '}') { --depth; if (depth == 0) break; }
            ++p;
        }
        if (depth != 0) fail(file_, s_, open, "unterminated '{' expression");
        std::string v = s_.substr(start, p - start);
        ++p; // closing brace
        return v;
    }

    NodePtr parseInterp(size_t& p) {
        std::string inner = readBraced(p);
        bool raw = !inner.empty() && inner[0] == '!';
        auto node   = std::make_unique<Node>();
        node->kind  = raw ? NodeKind::RawInterp : NodeKind::Interp;
        node->exprSrc = raw ? inner.substr(1) : inner;
        return node;
    }

    NodePtr parseText(size_t& p) {
        size_t start = p;
        while (p < s_.size() && s_[p] != '<' && s_[p] != '{') ++p;
        auto node  = std::make_unique<Node>();
        node->kind = NodeKind::Text;
        node->text = s_.substr(start, p - start);
        return node;
    }
};

// ── escaping helpers ─────────────────────────────────────────────────────────

// HTML-escapes text known statically (source literal text/attrs) so it can
// be embedded directly into the generated .push("...") string literal.
std::string htmlEscapeStatic(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        switch (c) {
            case '&':  out += "&amp;";  break;
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&#39;";  break;
            default:   out += c;
        }
    }
    return out;
}

// Escapes text so it can be embedded inside a SafeC/C "..." string literal.
std::string cEscape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += (char)c;
        }
    }
    return out;
}

// ── code generator ───────────────────────────────────────────────────────────
// Walks an Element tree, emitting SafeC statements that build 'bufVar' (an
// already-declared 'struct String'). Coalesces consecutive static text
// (including static attribute syntax) into single .push("...") calls,
// breaking only at dynamic ({expr}/{!expr}) points.

class CodeGen {
public:
    explicit CodeGen(std::string bufVar) : buf_(std::move(bufVar)) {}

    void emit(const Node& root, std::ostringstream& out) {
        emitElement(root, out);
        flush(out);
    }

private:
    std::string buf_;
    std::string pending_;

    void flush(std::ostringstream& out) {
        if (!pending_.empty()) {
            out << buf_ << ".push(\"" << cEscape(pending_) << "\");\n";
            pending_.clear();
        }
    }

    void emitElement(const Node& el, std::ostringstream& out) {
        pending_ += "<" + el.tag;
        for (auto& a : el.attrs) {
            if (!a.isExpr) {
                pending_ += " " + a.name + "=\"" + htmlEscapeStatic(a.value) + "\"";
            } else {
                pending_ += " " + a.name + "=\"";
                flush(out);
                out << "std::scx_append_esc(&" << buf_ << ", (" << a.value << "));\n";
                pending_ += "\"";
            }
        }
        if (el.children.empty()) {
            pending_ += "/>";
            return;
        }
        pending_ += ">";
        for (auto& c : el.children) emitNode(*c, out);
        pending_ += "</" + el.tag + ">";
    }

    void emitNode(const Node& n, std::ostringstream& out) {
        switch (n.kind) {
            case NodeKind::Text:
                // Literal template text is trusted (author-controlled),
                // same as JSX: only {expr} interpolation gets escaped.
                pending_ += n.text;
                break;
            case NodeKind::Interp:
                flush(out);
                out << "std::scx_append_esc(&" << buf_ << ", (" << n.exprSrc << "));\n";
                break;
            case NodeKind::RawInterp:
                flush(out);
                out << buf_ << ".push((" << n.exprSrc << "));\n";
                break;
            case NodeKind::Element:
                emitElement(n, out);
                break;
        }
    }
};

bool matchesKeywordAt(const std::string& s, size_t i, const char* kw) {
    size_t n = std::strlen(kw);
    if (i + n > s.size()) return false;
    if (s.compare(i, n, kw) != 0) return false;
    // Must be a whole word: not preceded/followed by an identifier char.
    if (i > 0 && isIdentChar(s[i - 1])) return false;
    if (i + n < s.size() && isIdentChar(s[i + n])) return false;
    return true;
}

} // namespace

std::string transpileScx(const std::string& src, const std::string& filename) {
    std::string out;
    out.reserve(src.size() + 256);
    size_t i = 0;
    int tmpCounter = 0;
    bool usedMarkup = false;

    while (i < src.size()) {
        char c = src[i];

        // String / char literals — copy verbatim, don't scan their contents.
        if (c == '"' || c == '\'') {
            char q = c;
            out += c;
            ++i;
            while (i < src.size() && src[i] != q) {
                if (src[i] == '\\' && i + 1 < src.size()) {
                    out += src[i];
                    out += src[i + 1];
                    i += 2;
                    continue;
                }
                out += src[i];
                ++i;
            }
            if (i < src.size()) { out += src[i]; ++i; }
            continue;
        }
        // Line comments.
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '/') {
            while (i < src.size() && src[i] != '\n') { out += src[i]; ++i; }
            continue;
        }
        // Block comments.
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '*') {
            out += src[i]; out += src[i + 1]; i += 2;
            while (i + 1 < src.size() && !(src[i] == '*' && src[i + 1] == '/')) {
                out += src[i]; ++i;
            }
            if (i + 1 < src.size()) { out += src[i]; out += src[i + 1]; i += 2; }
            continue;
        }

        if (matchesKeywordAt(src, i, "return")) {
            size_t afterKw = i + 6;
            size_t j = afterKw;
            while (j < src.size() && isWs(src[j])) ++j;
            if (j < src.size() && src[j] == '<' && j + 1 < src.size() &&
                isIdentStart(src[j + 1])) {
                MarkupParser mp(src, filename);
                size_t p = j;
                NodePtr root = mp.parseElement(p); // throws on malformed markup

                size_t k = p;
                while (k < src.size() && isWs(src[k])) ++k;
                if (k >= src.size() || src[k] != ';')
                    fail(filename, src, k, "expected ';' after scx markup in return statement");
                ++k;

                std::string bufVar = "__scx" + std::to_string(tmpCounter++);
                out += "struct String " + bufVar + " = std::string_new();\n";
                out += "unsafe {\n";
                std::ostringstream body;
                CodeGen gen(bufVar);
                gen.emit(*root, body);
                out += body.str();
                out += "}\n";
                out += "return " + bufVar + ";\n";
                usedMarkup = true;
                i = k;
                continue;
            }
            out += src.substr(i, 6);
            i = afterKw;
            continue;
        }

        out += c;
        ++i;
    }

    if (usedMarkup) {
        // Declarations only (.h) — NOT the .sc implementations. A .scx
        // file is compiled to its own standalone object file, same as any
        // other project source; if it (and some sibling project file)
        // both pulled in std/collections/string.sc's implementation
        // textually, both object files would strongly define the same
        // String methods and the final link would fail with duplicate
        // symbols (only safe for files that are themselves archived
        // std/ members, where the linker's lazy per-symbol archive pull
        // means at most one member ever actually provides a given
        // symbol — not true for a project's own directly-linked .o
        // files). Leaving these as undefined references lets them
        // resolve against whichever single place in the link — another
        // project file, or libsafec_std.a — actually defines them.
        out = "#include <std/scx/scx.h>\n"
              "#include <std/collections/string.h>\n" + out;
    }
    return out;
}

} // namespace safeguard
