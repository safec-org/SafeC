// --emit-doc-json (see DocJson.h).
//
// Doc comment conventions (documented for users in the SafeC docs,
// advanced/doc-comments):
//
//   /// text        an item's doc comment, Markdown, on the lines right
//   /** text */     above the declaration (no blank line in between)
//   // text         the std convention: plain comment lines right above a
//                   declaration document it too (used only when the block
//                   has no '///' or '/**' line; read as Markdown with its
//                   indented runs taken as code)
//   int x;  // text a trailing comment on the declaration's own line (and
//   ///< text       comment-only lines below it, aligned with it) — what
//                   std uses for fields, enumerators and short methods
//   //! text        the file's (module's) own doc; without one, the file's
//                   first comment block, when a blank line or a preprocessor
//                   line separates it from the first declaration
//
// A rule line ('// ── Section ──', '// -----', '// =====') ends a leading
// block: it and everything above it are not part of the doc. A line that
// is just '@hidden' or '#[doc(hidden)]' hides the item. Names starting or
// ending with '_' (std's type-erased '_raw_' functions) are marked hidden
// too; 'static' functions and globals are left out unless asked for.
#include "safec/DocJson.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace safec {

namespace {

// ── JSON output ──────────────────────────────────────────────────────────────

void jsonEscape(std::string &out, const std::string &s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof buf, "\\u%04x", c);
                out += buf;
            } else {
                out += (char)c;
            }
        }
    }
    out += '"';
}

// A tiny JSON object builder: fields in insertion order.
struct JObj {
    std::string s = "{";
    bool first = true;
    void key(const char *k) {
        if (!first) s += ',';
        first = false;
        jsonEscape(s, k);
        s += ':';
    }
    JObj &str(const char *k, const std::string &v) { key(k); jsonEscape(s, v); return *this; }
    JObj &strIf(const char *k, const std::string &v) { if (!v.empty()) str(k, v); return *this; }
    JObj &num(const char *k, long long v) { key(k); s += std::to_string(v); return *this; }
    JObj &flag(const char *k, bool v) { if (v) { key(k); s += "true"; } return *this; }
    JObj &raw(const char *k, const std::string &v) { key(k); s += v; return *this; }
    std::string done() { return s + "}"; }
};

std::string jArray(const std::vector<std::string> &elems, const char *sep = ",") {
    std::string s = "[";
    for (size_t i = 0; i < elems.size(); ++i) {
        if (i) s += sep;
        s += elems[i];
    }
    return s + "]";
}

std::string jStrArray(const std::vector<std::string> &v) {
    std::vector<std::string> e;
    for (auto &x : v) { std::string t; jsonEscape(t, x); e.push_back(t); }
    return jArray(e);
}

// ── strings ──────────────────────────────────────────────────────────────────

std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

bool startsWith(const std::string &s, const char *p) { return s.rfind(p, 0) == 0; }

std::string collapseWs(const std::string &s) {
    std::string out;
    bool ws = false;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ws = true; continue; }
        if (ws && !out.empty()) {
            char p = out.back();
            // No space just inside brackets or before ',' / ';' / ')'.
            if (p != '(' && p != '[' && c != ')' && c != ']' && c != ',' && c != ';') out += ' ';
        }
        ws = false;
        out += c;
    }
    return out;
}

// ── source files: per-line code / comment split ─────────────────────────────

enum CommentStyle { C_NONE = 0, C_LINE, C_DOC_LINE, C_MODULE_LINE, C_BLOCK, C_DOC_BLOCK, C_MODULE_BLOCK };

struct LineInfo {
    std::string code;        // the line with comments blanked out (columns kept)
    std::string text;        // the comment's text, marker stripped
    int  style = C_NONE;
    int  commentCol = -1;    // byte column (0-based) the comment starts at
    bool commentOnly = false;
    bool blank = false;
    bool trailing = false;   // a comment after code on the same line
};

struct SourceFile {
    std::vector<LineInfo> lines;
    bool ok = false;
};

std::string stripOneSpace(const std::string &s) {
    return !s.empty() && s[0] == ' ' ? s.substr(1) : s;
}

SourceFile analyze(const std::string &path) {
    SourceFile f;
    std::ifstream in(path, std::ios::binary);
    if (!in) return f;
    f.ok = true;
    std::string raw;
    int blockStyle = C_NONE;   // inside a /* */ comment
    while (std::getline(in, raw)) {
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        LineInfo li;
        li.code = raw;
        bool hasCode = false;
        bool hasComment = false;
        std::string text;
        size_t i = 0;
        auto noteComment = [&](int style, size_t col) {
            if (!hasComment) { li.style = style; li.commentCol = (int)col; }
            hasComment = true;
        };
        if (blockStyle != C_NONE) noteComment(blockStyle, 0);
        while (i < raw.size()) {
            if (blockStyle != C_NONE) {
                size_t end = raw.find("*/", i);
                std::string part = raw.substr(i, end == std::string::npos ? std::string::npos : end - i);
                // ' * text' continuation lines: drop the leading '*'.
                std::string t = part;
                size_t a = t.find_first_not_of(" \t");
                if (a != std::string::npos && t[a] == '*') {
                    if (a + 1 >= t.size() || t[a + 1] != '/') t = t.substr(a + 1);
                }
                if (!text.empty() && !t.empty()) text += ' ';
                text += stripOneSpace(t);
                size_t stop = end == std::string::npos ? raw.size() : end + 2;
                for (size_t k = i; k < stop; ++k) li.code[k] = ' ';
                i = stop;
                if (end != std::string::npos) blockStyle = C_NONE;
                continue;
            }
            char c = raw[i];
            if (c == '"' || c == '\'') {
                hasCode = true;
                char q = c;
                ++i;
                while (i < raw.size() && raw[i] != q) { if (raw[i] == '\\') ++i; ++i; }
                ++i;
                continue;
            }
            if (c == '/' && i + 1 < raw.size() && raw[i + 1] == '/') {
                int style = C_LINE;
                size_t m = i + 2;
                if (m < raw.size() && raw[m] == '/' && !(m + 1 < raw.size() && raw[m + 1] == '/')) { style = C_DOC_LINE; ++m; }
                else if (m < raw.size() && raw[m] == '!') { style = C_MODULE_LINE; ++m; }
                noteComment(style, i);
                if (style == C_DOC_LINE && m < raw.size() && raw[m] == '<') ++m;   // '///<'
                if (!text.empty()) text += ' ';
                text += stripOneSpace(raw.substr(m));
                for (size_t k = i; k < raw.size(); ++k) li.code[k] = ' ';
                i = raw.size();
                break;
            }
            if (c == '/' && i + 1 < raw.size() && raw[i + 1] == '*') {
                int style = C_BLOCK;
                size_t m = i + 2;
                if (m < raw.size() && raw[m] == '*' && !(m + 1 < raw.size() && raw[m + 1] == '/')) { style = C_DOC_BLOCK; ++m; }
                else if (m < raw.size() && raw[m] == '!') { style = C_MODULE_BLOCK; ++m; }
                noteComment(style, i);
                if (style == C_DOC_BLOCK && m < raw.size() && raw[m] == '<') ++m;
                for (size_t k = i; k < m && k < raw.size(); ++k) li.code[k] = ' ';
                blockStyle = style;
                i = m;
                continue;
            }
            if (c != ' ' && c != '\t') hasCode = true;
            ++i;
        }
        li.text = text;
        // Trailing whitespace of a block comment's last line.
        while (!li.text.empty() && (li.text.back() == ' ' || li.text.back() == '\t')) li.text.pop_back();
        li.blank = !hasCode && !hasComment;
        li.commentOnly = hasComment && !hasCode;
        li.trailing = hasComment && hasCode;
        f.lines.push_back(std::move(li));
    }
    return f;
}

// '── Section ──', '-----', '=====', '*****': a separator, not documentation.
bool isRule(const std::string &text) {
    std::string t = trim(text);
    if (startsWith(t, "\xE2\x94\x80") || startsWith(t, "\xE2\x95\x90") || startsWith(t, "\xE2\x94\x81")) return true;
    if (t.size() >= 3 && (startsWith(t, "---") || startsWith(t, "===") || startsWith(t, "***") || startsWith(t, "###")))
        return true;
    return false;
}

struct DocText {
    std::string text;
    bool markdown = false;   // from '///' / '/**' (else plain '//' comments)
    bool hidden = false;
};

// Removes the '@hidden' / '#[doc(hidden)]' lines.
void applyHidden(DocText &d) {
    std::istringstream in(d.text);
    std::string line, out;
    bool firstOut = true;
    while (std::getline(in, line)) {
        std::string t = trim(line);
        if (t == "@hidden" || t == "#[doc(hidden)]") { d.hidden = true; continue; }
        if (!firstOut) out += '\n';
        out += line;
        firstOut = false;
    }
    d.text = trim(out).empty() ? "" : out;
}

// ── declaration source text ──────────────────────────────────────────────────

struct Extracted {
    std::string text;
    unsigned endLine = 0;    // 1-based line holding the terminator
};

// The declaration's source text from (line, col) up to the first ';' or '{'
// (or ',' / ')' with stopAtComma, for one parameter) outside brackets.
// A '{' after a top-level '=' (an initializer list) is kept.
Extracted extractDecl(const SourceFile &f, unsigned line, unsigned col, bool stopAtComma = false) {
    Extracted ex;
    if (line == 0 || line > f.lines.size()) return ex;
    int depth = 0;
    bool afterEq = false;
    std::string out;
    for (unsigned l = line; l <= f.lines.size() && l < line + 80; ++l) {
        const std::string &code = f.lines[l - 1].code;
        size_t start = l == line ? (col > 0 ? col - 1 : 0) : 0;
        for (size_t i = start; i < code.size(); ++i) {
            char c = code[i];
            if (c == '"' || c == '\'') {
                char q = c;
                out += c;
                ++i;
                while (i < code.size() && code[i] != q) {
                    if (code[i] == '\\' && i + 1 < code.size()) out += code[i++];
                    out += code[i++];
                }
                if (i < code.size()) out += code[i];
                continue;
            }
            if (c == '(' || c == '[') ++depth;
            if (c == ')' || c == ']') {
                if (depth == 0 && stopAtComma) { ex.text = collapseWs(out); ex.endLine = l; return ex; }
                --depth;
            }
            if (depth == 0) {
                if (c == ';' || (c == '{' && !afterEq)) { ex.text = collapseWs(out); ex.endLine = l; return ex; }
                if (c == ',' && stopAtComma) { ex.text = collapseWs(out); ex.endLine = l; return ex; }
                if (c == '=') afterEq = true;
            }
            if (c == '{') ++depth;
            if (c == '}') --depth;
            out += c;
        }
        out += '\n';
        if (out.size() > 4000) break;
    }
    ex.text = collapseWs(out);
    ex.endLine = line;
    return ex;
}

// ── types ────────────────────────────────────────────────────────────────────

// The generic parameter names in scope (the parser reads an unknown 'T' as
// a struct name): printed bare.
std::set<std::string> gGenericNames;

struct GenericScope {
    std::set<std::string> saved;
    GenericScope(const std::vector<GenericParam> &a, const std::vector<GenericParam> &b = {}) : saved(gGenericNames) {
        for (auto &g : a) gGenericNames.insert(g.name);
        for (auto &g : b) gGenericNames.insert(g.name);
    }
    ~GenericScope() { gGenericNames = saved; }
};

std::string typeStr(const TypePtr &t) {
    if (!t) return "";
    switch (t->kind) {
    case TypeKind::Struct: {
        auto &st = static_cast<const StructType &>(*t);
        if (st.typeArgs.empty() && gGenericNames.count(st.name)) return st.name;
        std::string s = (st.isUnion ? "union " : "struct ") + st.name;
        if (!st.typeArgs.empty()) {
            s += '<';
            for (size_t i = 0; i < st.typeArgs.size(); ++i) s += (i ? ", " : "") + typeStr(st.typeArgs[i]);
            s += '>';
        }
        return s;
    }
    case TypeKind::Enum: return "enum " + static_cast<const EnumType &>(*t).name;
    case TypeKind::Reference: {
        auto &r = static_cast<const ReferenceType &>(*t);
        // 'fn R(Args)' / '?fn R(Args)': a &static reference to a function.
        if (r.base && r.base->kind == TypeKind::Function && r.region == Region::Static)
            return (r.nullable ? "?" : "") + typeStr(r.base);
        std::string s = r.nullable ? "?&" : "&";
        if (r.region == Region::Arena) s += "arena<" + r.arenaName + "> ";
        else if (r.region == Region::Stack || r.region == Region::Heap || r.region == Region::Static)
            s += regionName(r.region) + " ";
        if (!r.mut) s = "const " + s;
        if (r.pointeeVolatile) s += "volatile ";
        return s + typeStr(r.base);
    }
    case TypeKind::Pointer: {
        auto &p = static_cast<const PointerType &>(*t);
        return std::string(p.isConst ? "const " : "") + (p.pointeeVolatile ? "volatile " : "") + typeStr(p.base) + "*";
    }
    case TypeKind::Optional: return "?" + typeStr(static_cast<const OptionalType &>(*t).inner);
    case TypeKind::Slice: return "[]" + typeStr(static_cast<const SliceType &>(*t).element);
    case TypeKind::Array: {
        auto &a = static_cast<const ArrayType &>(*t);
        return typeStr(a.element) + (a.size < 0 ? "[]" : "[" + std::to_string(a.size) + "]");
    }
    case TypeKind::Function: {
        auto &fn = static_cast<const FunctionType &>(*t);
        std::string s = "fn " + typeStr(fn.returnType) + "(";
        for (size_t i = 0; i < fn.paramTypes.size(); ++i) s += (i ? ", " : "") + typeStr(fn.paramTypes[i]);
        if (fn.variadic) s += fn.paramTypes.empty() ? "..." : ", ...";
        return s + ")";
    }
    default: return t->str();
    }
}

std::string genericsJson(const std::vector<GenericParam> &gps) {
    std::vector<std::string> e;
    for (auto &g : gps) {
        JObj o;
        o.str("name", g.name);
        if (!g.constraints.empty()) o.raw("bounds", jStrArray(g.constraints));
        o.flag("pack", g.isPack);
        e.push_back(o.done());
    }
    return jArray(e);
}

std::string genericsPrefix(const std::vector<GenericParam> &gps) {
    if (gps.empty()) return "";
    std::string s = "generic<";
    for (size_t i = 0; i < gps.size(); ++i) {
        if (i) s += ", ";
        s += gps[i].name;
        if (gps[i].isPack) s += "...";
        for (size_t k = 0; k < gps[i].constraints.size(); ++k) s += (k ? " + " : ": ") + gps[i].constraints[k];
    }
    return s + "> ";
}

std::string paramsJson(const std::vector<ParamDecl> &ps) {
    std::vector<std::string> e;
    for (auto &p : ps) {
        JObj o;
        o.str("name", p.name).str("type", typeStr(p.type));
        o.flag("pack", p.isPack);
        e.push_back(o.done());
    }
    return jArray(e);
}

bool hiddenName(const std::string &n) {
    return !n.empty() && (n.front() == '_' || n.back() == '_');
}

} // namespace

// ── the collector ────────────────────────────────────────────────────────────

struct DocCollector::Impl {
    DocOptions opts;
    std::vector<std::string> roots;                               // real paths
    std::vector<std::pair<std::string, std::string>> externs;     // name, real dir
    std::map<std::string, SourceFile> files;                      // real path -> analysis

    // One collected item; its JSON is built at the end, after merging.
    struct Item {
        std::string kind, name, ns, crate;  // crate: "" (local) or an extern's name
        std::string file;                   // relative to its root ('/'-separated)
        unsigned line = 0, col = 0;
        DocText doc;
        std::string sig;
        std::string extra;                  // more JSON fields (",\"k\":v...")
        std::string owner;                  // out-of-line method definition: its struct
        bool isDef = false;                 // has a body (function)
        bool header = false;                // declared in a .h file
        std::string defFile; unsigned defLine = 0;
        bool hidden = false;
        // struct / trait members, merged with out-of-line method definitions
        struct Member {
            std::string kind, name;         // "field" | "method" | "variant"
            unsigned line = 0;
            DocText doc;
            std::string json;               // without the doc
            std::string defFile; unsigned defLine = 0;
        };
        std::vector<Member> members;
    };
    std::vector<Item> items;
    std::set<std::string> seen;             // kind|crate|file|line|name
    std::map<std::string, DocText> moduleDocs;   // rel path -> doc (local files)
    std::set<std::string> localFiles;

    static std::string real(const std::string &p) {
        llvm::SmallString<256> out;
        if (llvm::sys::fs::real_path(p, out)) {
            llvm::SmallString<256> abs(p);
            llvm::sys::fs::make_absolute(abs);
            llvm::sys::path::remove_dots(abs, true);
            return std::string(abs.str());
        }
        return std::string(out.str());
    }

    static std::string slashes(std::string s) {
        std::replace(s.begin(), s.end(), '\\', '/');
        return s;
    }

    // Where 'file' belongs: 0 local (rel set), 1 extern (crate, rel set), -1 skip.
    int classify(const char *file, std::string &rel, std::string &crate, std::string &realPath) {
        if (!file || !*file || file[0] == '<') return -1;
        realPath = real(file);
        std::string rp = slashes(realPath);
        auto under = [&](const std::string &dir, std::string &out) {
            std::string d = slashes(dir);
            if (!d.empty() && d.back() != '/') d += '/';
            if (rp.rfind(d, 0) == 0) { out = rp.substr(d.size()); return true; }
            return false;
        };
        if (roots.empty()) { rel = rp; crate.clear(); return 0; }
        for (auto &r : roots) if (under(r, rel)) { crate.clear(); return 0; }
        for (auto &[name, dir] : externs) if (under(dir, rel)) { crate = name; return 1; }
        return -1;
    }

    const SourceFile &source(const std::string &realPath) {
        auto it = files.find(realPath);
        if (it != files.end()) return it->second;
        return files.emplace(realPath, analyze(realPath)).first->second;
    }

    static bool isHeader(const std::string &rel) {
        auto dot = rel.rfind('.');
        return dot != std::string::npos && rel.substr(dot) == ".h";
    }

    // ── doc comments ─────────────────────────────────────────────────────────

    DocText leadingDoc(const SourceFile &f, unsigned line) {
        DocText d;
        if (line < 2 || line > f.lines.size()) return d;
        long i = (long)line - 2;
        // Lines of the declaration's own prefix: 'generic<T>' above a
        // struct, attributes on their own line.
        while (i >= 0 && !f.lines[i].commentOnly && !f.lines[i].blank) {
            std::string t = trim(f.lines[i].code);
            bool prefix = (startsWith(t, "generic") || startsWith(t, "__attribute__") || startsWith(t, "[["))
                          && t.find(';') == std::string::npos && t.find('}') == std::string::npos;
            if (!prefix) return d;
            --i;
        }
        long end = i;
        while (i >= 0 && f.lines[i].commentOnly) --i;
        long begin = i + 1;
        if (begin > end) return d;
        // Continuation lines of a trailing comment on the line above.
        if (begin > 0 && f.lines[begin - 1].trailing) {
            int col = f.lines[begin - 1].commentCol;
            while (begin <= end && f.lines[begin].commentCol == col &&
                   f.lines[begin].style != C_DOC_LINE && f.lines[begin].style != C_DOC_BLOCK) ++begin;
        }
        // A rule line ends the block.
        for (long k = end; k >= begin; --k) {
            if (isRule(f.lines[k].text)) { begin = k + 1; break; }
        }
        bool anyDoc = false;
        for (long k = begin; k <= end; ++k) {
            int s = f.lines[k].style;
            if (s == C_DOC_LINE || s == C_DOC_BLOCK) anyDoc = true;
        }
        std::string text;
        bool firstLine = true;
        for (long k = begin; k <= end; ++k) {
            int s = f.lines[k].style;
            if (s == C_MODULE_LINE || s == C_MODULE_BLOCK) continue;
            if (anyDoc && s != C_DOC_LINE && s != C_DOC_BLOCK) continue;
            if (!firstLine) text += '\n';
            text += f.lines[k].text;
            firstLine = false;
        }
        d.text = trim(text).empty() ? "" : text;
        d.markdown = anyDoc;
        applyHidden(d);
        return d;
    }

    DocText trailingDoc(const SourceFile &f, unsigned line, unsigned endLine) {
        DocText d;
        if (line == 0 || line > f.lines.size()) return d;
        if (endLine < line) endLine = line;
        for (unsigned l = line; l <= endLine && l <= f.lines.size(); ++l) {
            const LineInfo &li = f.lines[l - 1];
            if (!li.trailing) continue;
            if (isRule(li.text)) return d;
            d.text = li.text;
            d.markdown = li.style == C_DOC_LINE || li.style == C_DOC_BLOCK;
            int col = li.commentCol;
            for (unsigned k = l + 1; k <= f.lines.size(); ++k) {
                const LineInfo &n = f.lines[k - 1];
                if (!n.commentOnly || n.commentCol != col || n.style == C_DOC_LINE || n.style == C_DOC_BLOCK) break;
                d.text += '\n' + n.text;
            }
            applyHidden(d);
            return d;
        }
        return d;
    }

    // 'trailing' false: another declaration shares the line (enumerators on
    // one line, a struct's first field beside its '{'), so a trailing
    // comment there is not this one's.
    DocText docFor(const SourceFile &f, unsigned line, unsigned endLine, bool trailing = true) {
        DocText lead = leadingDoc(f, line);
        DocText trail = trailing ? trailingDoc(f, line, endLine) : DocText{};
        if (lead.text.empty()) { trail.hidden = trail.hidden || lead.hidden; return trail; }
        if (!trail.text.empty()) lead.text += "\n\n" + trail.text;
        lead.hidden = lead.hidden || trail.hidden;
        return lead;
    }

    // The file's own doc: its '//!' lines, else its first comment block when
    // something other than a declaration follows it.
    DocText moduleDoc(const SourceFile &f) {
        DocText d;
        std::string bang;
        for (auto &li : f.lines) {
            if (li.style == C_MODULE_LINE || li.style == C_MODULE_BLOCK) {
                if (!bang.empty()) bang += '\n';
                bang += li.text;
            }
        }
        if (!bang.empty()) { d.text = bang; d.markdown = true; return d; }
        size_t i = 0;
        while (i < f.lines.size()) {
            // Skip blank and preprocessor lines before the first comment block.
            while (i < f.lines.size() && (f.lines[i].blank || (!f.lines[i].commentOnly && startsWith(trim(f.lines[i].code), "#"))))
                ++i;
            if (i >= f.lines.size() || !f.lines[i].commentOnly) return d;
            size_t b = i;
            while (i < f.lines.size() && f.lines[i].commentOnly) ++i;
            // Followed by a declaration: that declaration's doc, not the file's.
            bool followedByDecl = i < f.lines.size() && !f.lines[i].blank &&
                                  !startsWith(trim(f.lines[i].code), "#");
            std::string text;
            bool anyDoc = false;
            for (size_t k = b; k < i; ++k) {
                if (f.lines[k].style == C_DOC_LINE || f.lines[k].style == C_DOC_BLOCK) anyDoc = true;
                if (k > b) text += '\n';
                text += f.lines[k].text;
            }
            std::string t = trim(text);
            if (followedByDecl) return d;
            if (t.find("SPDX-License-Identifier") != std::string::npos || startsWith(t, "Copyright"))
                continue;   // a license header: the module doc may come next
            d.text = text;
            d.markdown = anyDoc;
            return d;
        }
        return d;
    }

    // ── items ────────────────────────────────────────────────────────────────

    bool begin(Item &it, const char *kind, const Decl &d, int &where, std::string &realPath) {
        it.kind = kind;
        it.name = d.name;
        it.ns = d.docNamespace;
        where = classify(d.loc.file, it.file, it.crate, realPath);
        if (where < 0) return false;
        it.line = d.loc.line;
        it.col = d.loc.col;
        it.header = isHeader(it.file);
        if (where == 0) localFiles.insert(it.file);
        return true;
    }

    bool remember(const Item &it) {
        std::string key = it.kind + "|" + it.crate + "|" + it.file + "|" + std::to_string(it.line) + "|" + it.name;
        return seen.insert(key).second;
    }

    Item::Member methodMember(const SourceFile &f, const MethodDecl &md, bool local) {
        Item::Member m;
        m.kind = "method";
        m.name = md.name;
        SourceLocation s = md.startLoc.line ? md.startLoc : md.loc;
        m.line = s.line;
        Extracted ex = extractDecl(f, s.line, s.col);
        if (local) m.doc = docFor(f, s.line, ex.endLine);
        GenericScope gs(md.genericParams);
        JObj o;
        o.str("kind", "method").str("name", md.name).num("line", s.line).str("sig", ex.text);
        if (!md.genericParams.empty()) o.raw("generics", genericsJson(md.genericParams));
        o.raw("params", paramsJson(md.params)).str("ret", typeStr(md.returnType));
        o.flag("static", md.isStatic).flag("const", md.isConst);
        m.json = o.done();
        return m;
    }

    void addStruct(const StructDecl &sd) {
        if (sd.name.empty()) return;
        Item it;
        int where; std::string rp;
        if (!begin(it, sd.isUnion ? "union" : "struct", sd, where, rp)) return;
        if (!remember(it)) return;
        const SourceFile &f = source(rp);
        bool local = where == 0;
        GenericScope gs(sd.genericParams);
        // The declaration's head line ('struct Vec {').
        it.sig = genericsPrefix(sd.genericParams) + (sd.isPacked ? "packed " : "") +
                 (sd.isUnion ? "union " : "struct ") + sd.name;
        bool shared = false;
        for (auto &fd : sd.fields) if (fd.loc.line == it.line) shared = true;
        for (auto &md : sd.methodDecls) if ((md.startLoc.line ? md.startLoc.line : md.loc.line) == it.line) shared = true;
        if (local) it.doc = docFor(f, it.line, it.line, !shared);
        JObj extra;
        if (!sd.genericParams.empty()) extra.raw("generics", genericsJson(sd.genericParams));
        extra.flag("packed", sd.isPacked);
        it.extra = extra.done();
        for (size_t fi = 0; fi < sd.fields.size(); ++fi) {
            auto &fd = sd.fields[fi];
            if (fd.isAnonymous) continue;
            Item::Member m;
            m.kind = "field";
            m.name = fd.name;
            m.line = fd.loc.line;
            Extracted ex = extractDecl(f, fd.loc.line, fd.loc.col);
            // 'int a, b;' style neighbours on the line: only the last gets it.
            bool last = fi + 1 >= sd.fields.size() || sd.fields[fi + 1].loc.line != fd.loc.line;
            if (local) m.doc = docFor(f, fd.loc.line, ex.endLine, last);
            JObj o;
            o.str("kind", "field").str("name", fd.name).num("line", fd.loc.line)
             .str("sig", ex.text).str("type", typeStr(fd.type));
            if (fd.bitWidth >= 0) o.num("bits", fd.bitWidth);
            m.json = o.done();
            it.members.push_back(std::move(m));
        }
        for (auto &md : sd.methodDecls) it.members.push_back(methodMember(f, md, local));
        it.hidden = it.doc.hidden || hiddenName(it.name);
        items.push_back(std::move(it));
    }

    void addTrait(const TraitDecl &td) {
        Item it;
        int where; std::string rp;
        if (!begin(it, "trait", td, where, rp)) return;
        if (!remember(it)) return;
        const SourceFile &f = source(rp);
        bool local = where == 0;
        it.sig = "trait " + td.name;
        for (size_t i = 0; i < td.supertraits.size(); ++i) it.sig += (i ? " + " : " : ") + td.supertraits[i];
        if (local) it.doc = docFor(f, it.line, it.line);
        JObj extra;
        if (!td.supertraits.empty()) extra.raw("supertraits", jStrArray(td.supertraits));
        it.extra = extra.done();
        for (auto &md : td.methods) it.members.push_back(methodMember(f, md, local));
        it.hidden = it.doc.hidden || hiddenName(it.name);
        items.push_back(std::move(it));
    }

    void addEnum(const EnumDecl &ed) {
        if (ed.name.empty()) return;
        Item it;
        int where; std::string rp;
        if (!begin(it, "enum", ed, where, rp)) return;
        if (!remember(it)) return;
        const SourceFile &f = source(rp);
        bool local = where == 0;
        it.sig = "enum " + ed.name;
        JObj extra;
        if (ed.underlyingType) {
            it.sig += " : " + typeStr(ed.underlyingType);
            extra.str("underlying", typeStr(ed.underlyingType));
        }
        it.extra = extra.done();
        bool shared = false;
        for (auto &l : ed.enumeratorLocs) if (l.line == it.line) shared = true;
        if (local) it.doc = docFor(f, it.line, it.line, !shared);
        for (size_t i = 0; i < ed.enumerators.size(); ++i) {
            Item::Member m;
            m.kind = "variant";
            m.name = ed.enumerators[i].first;
            SourceLocation l = i < ed.enumeratorLocs.size() ? ed.enumeratorLocs[i] : SourceLocation{};
            m.line = l.line;
            bool last = i + 1 >= ed.enumeratorLocs.size() || ed.enumeratorLocs[i + 1].line != l.line;
            // The leading block above the enum's own line is the enum's.
            bool ownLine = l.line != it.line;
            if (local && l.line) m.doc = ownLine ? docFor(f, l.line, l.line, last)
                                                 : (last ? trailingDoc(f, l.line, l.line) : DocText{});
            JObj o;
            o.str("kind", "variant").str("name", m.name).num("line", l.line);
            if (ed.enumerators[i].second) o.num("value", (long long)*ed.enumerators[i].second);
            m.json = o.done();
            it.members.push_back(std::move(m));
        }
        it.hidden = it.doc.hidden || hiddenName(it.name);
        items.push_back(std::move(it));
    }

    void addFunction(const FunctionDecl &fd) {
        if (fd.isStatic && !opts.includePrivate) return;
        Item it;
        int where; std::string rp;
        if (!begin(it, "function", fd, where, rp)) return;
        if (!fd.namespaceName.empty()) it.ns = fd.namespaceName;
        if (fd.isMethod) it.owner = fd.methodOwner;
        if (!remember(it)) return;
        const SourceFile &f = source(rp);
        GenericScope gs(fd.genericParams);
        Extracted ex = extractDecl(f, it.line, it.col);
        it.sig = ex.text;
        if (!fd.genericParams.empty() && !startsWith(it.sig, "generic")) it.sig = genericsPrefix(fd.genericParams) + it.sig;
        if (where == 0) it.doc = docFor(f, it.line, ex.endLine);
        it.isDef = fd.body != nullptr;
        JObj extra;
        if (!fd.genericParams.empty()) extra.raw("generics", genericsJson(fd.genericParams));
        extra.raw("params", paramsJson(fd.params)).str("ret", typeStr(fd.returnType));
        extra.flag("variadic", fd.isVariadic).flag("static", fd.isStatic).flag("inline", fd.isInline)
             .flag("extern", fd.isExtern).flag("const", fd.isConst).flag("consteval", fd.isConsteval)
             .flag("must_use", fd.isMustUse).flag("noreturn", fd.isNoReturn).flag("pure", fd.isPure);
        if (!fd.wasmExportName.empty()) extra.str("export_name", fd.wasmExportName);
        it.extra = extra.done();
        it.hidden = it.doc.hidden || hiddenName(it.name);
        items.push_back(std::move(it));
    }

    void addGlobal(const GlobalVarDecl &gv) {
        if (gv.isStatic && !opts.includePrivate) return;
        Item it;
        int where; std::string rp;
        if (!begin(it, gv.isConst ? "const" : "global", gv, where, rp)) return;
        if (!gv.namespaceName.empty()) it.ns = gv.namespaceName;
        if (!remember(it)) return;
        const SourceFile &f = source(rp);
        Extracted ex = extractDecl(f, it.line, it.col);
        it.sig = ex.text;
        if (where == 0) it.doc = docFor(f, it.line, ex.endLine);
        JObj extra;
        extra.str("type", typeStr(gv.type)).flag("extern", gv.isExtern).flag("static", gv.isStatic);
        it.extra = extra.done();
        it.hidden = it.doc.hidden || hiddenName(it.name);
        items.push_back(std::move(it));
    }

    void addSimple(const Decl &d, const char *kind, const std::string &sigOverride, const std::string &extraJson) {
        Item it;
        int where; std::string rp;
        if (!begin(it, kind, d, where, rp)) return;
        if (!remember(it)) return;
        const SourceFile &f = source(rp);
        unsigned endLine = it.line;
        if (sigOverride.empty()) {
            Extracted ex = extractDecl(f, it.line, it.col);
            it.sig = ex.text;
            endLine = ex.endLine;
        } else {
            it.sig = sigOverride;
        }
        if (where == 0) it.doc = docFor(f, it.line, endLine);
        it.extra = extraJson;
        it.hidden = it.doc.hidden || hiddenName(it.name);
        items.push_back(std::move(it));
    }

    void addMacro(const MacroDef &m) {
        if (m.file.empty() || m.name.empty()) return;
        // Include guards and the compiler's own predefined names.
        if (startsWith(m.name, "__")) return;
        bool guard = trim(m.body).empty() && m.params.empty() &&
                     (m.name.size() > 2 && (m.name.substr(m.name.size() - 2) == "_H" ||
                                            m.name.substr(m.name.size() - 3) == "_H_"));
        if (guard) return;
        Item it;
        std::string rp;
        int where = classify(m.file.c_str(), it.file, it.crate, rp);
        if (where < 0) return;
        it.kind = "macro";
        it.name = m.name;
        it.line = m.loc.line;
        it.col = 1;
        it.header = isHeader(it.file);
        if (!remember(it)) return;
        if (where == 0) localFiles.insert(it.file);
        std::string sig = "#define " + m.name;
        if (m.isFunctionLike) {
            sig += "(";
            for (size_t i = 0; i < m.params.size(); ++i) sig += (i ? ", " : "") + m.params[i];
            sig += ")";
        }
        std::string body = collapseWs(m.body);
        if (body.size() > 300) body = body.substr(0, 300) + " ...";
        if (!body.empty()) sig += " " + body;
        it.sig = sig;
        const SourceFile &f = source(rp);
        if (where == 0) it.doc = docFor(f, it.line, it.line);
        JObj extra;
        extra.flag("function_like", m.isFunctionLike);
        if (m.isFunctionLike) extra.raw("params", jStrArray(m.params));
        extra.str("value", body);
        it.extra = extra.done();
        it.hidden = it.doc.hidden || hiddenName(it.name);
        items.push_back(std::move(it));
    }

    // ── merging ──────────────────────────────────────────────────────────────

    // Out-of-line method definitions lend their doc (when the in-struct
    // declaration has none) and their location to the struct's member;
    // a free function declared in a header and defined elsewhere becomes
    // one item (the header's), with the definition's location and, when
    // the header has none, its doc.
    void merge() {
        std::map<std::string, size_t> structIdx;   // crate|ns-less name -> index
        for (size_t i = 0; i < items.size(); ++i) {
            auto &it = items[i];
            if (it.kind == "struct" || it.kind == "union" || it.kind == "trait") {
                std::string key = it.crate + "|" + it.name;
                if (!structIdx.count(key)) structIdx[key] = i;
            }
        }
        std::vector<bool> drop(items.size(), false);
        for (size_t i = 0; i < items.size(); ++i) {
            auto &it = items[i];
            if (it.kind != "function" || it.owner.empty()) continue;
            drop[i] = true;
            auto s = structIdx.find(it.crate + "|" + it.owner);
            if (s == structIdx.end()) continue;
            for (auto &m : items[s->second].members) {
                if (m.kind != "method" || m.name != it.name) continue;
                if (m.doc.text.empty() && !it.doc.text.empty()) m.doc = it.doc;
                if (it.isDef && m.defLine == 0) { m.defFile = it.file; m.defLine = it.line; }
                break;
            }
        }
        std::map<std::string, size_t> fnIdx;       // crate|ns|name -> primary index
        for (size_t i = 0; i < items.size(); ++i) {
            auto &it = items[i];
            if (drop[i] || it.kind != "function") continue;
            std::string key = it.crate + "|" + it.ns + "|" + it.name;
            auto p = fnIdx.find(key);
            if (p == fnIdx.end()) { fnIdx[key] = i; continue; }
            Item &a = items[p->second];
            Item &b = it;
            // The primary is the header declaration (else the first seen).
            bool bPrimary = b.header && !a.header;
            Item &prim = bPrimary ? b : a;
            Item &other = bPrimary ? a : b;
            if (prim.doc.text.empty() && !other.doc.text.empty()) prim.doc = other.doc;
            if (other.isDef && prim.defLine == 0) { prim.defFile = other.file; prim.defLine = other.line; }
            prim.hidden = prim.hidden || other.doc.hidden;
            if (bPrimary) { drop[p->second] = true; fnIdx[key] = i; }
            else drop[i] = true;
        }
        std::vector<Item> kept;
        for (size_t i = 0; i < items.size(); ++i)
            if (!drop[i]) kept.push_back(std::move(items[i]));
        items = std::move(kept);
    }

    std::string itemJson(const Item &it) {
        JObj o;
        o.str("kind", it.kind).str("name", it.name).strIf("ns", it.ns).strIf("crate", it.crate)
         .str("file", it.file).num("line", it.line).str("sig", it.sig);
        if (!it.doc.text.empty()) o.str("doc", it.doc.text).flag("md", it.doc.markdown);
        o.flag("hidden", it.hidden);
        if (it.defLine) o.str("def_file", it.defFile).num("def_line", it.defLine);
        // 'extra' is a complete object: splice its fields in.
        if (it.extra.size() > 2) o.s += "," + it.extra.substr(1, it.extra.size() - 2);
        if (!it.members.empty()) {
            std::vector<std::string> ms;
            for (auto &m : it.members) {
                std::string j = m.json.substr(0, m.json.size() - 1);
                if (!m.doc.text.empty()) {
                    j += ",\"doc\":";
                    jsonEscape(j, m.doc.text);
                    if (m.doc.markdown) j += ",\"md\":true";
                }
                if (m.doc.hidden || hiddenName(m.name)) j += ",\"hidden\":true";
                if (m.defLine) {
                    j += ",\"def_file\":";
                    jsonEscape(j, m.defFile);
                    j += ",\"def_line\":" + std::to_string(m.defLine);
                }
                ms.push_back(j + "}");
            }
            o.raw("members", "[\n" + jArray(ms, ",\n").substr(1));
        }
        return o.done();
    }
};

DocCollector::DocCollector(DocOptions opts) : impl_(new Impl) {
    impl_->opts = std::move(opts);
    for (auto &r : impl_->opts.roots) impl_->roots.push_back(Impl::real(r));
    for (auto &[n, d] : impl_->opts.externs) impl_->externs.push_back({n, Impl::real(d)});
}

DocCollector::~DocCollector() = default;

void DocCollector::add(const TranslationUnit &tu,
                       const std::unordered_map<std::string, MacroDef> &macros,
                       const std::string &inputPath) {
    Impl &I = *impl_;
    // The input itself is a documented file even when it declares nothing.
    {
        std::string rel, crate, rp;
        if (I.classify(inputPath.c_str(), rel, crate, rp) == 0) I.localFiles.insert(rel);
    }
    for (auto &dp : tu.decls) {
        const Decl &d = *dp;
        switch (d.kind) {
        case DeclKind::Struct: I.addStruct(static_cast<const StructDecl &>(d)); break;
        case DeclKind::Trait: I.addTrait(static_cast<const TraitDecl &>(d)); break;
        case DeclKind::Enum: I.addEnum(static_cast<const EnumDecl &>(d)); break;
        case DeclKind::Function: I.addFunction(static_cast<const FunctionDecl &>(d)); break;
        case DeclKind::GlobalVar: I.addGlobal(static_cast<const GlobalVarDecl &>(d)); break;
        case DeclKind::TypeAlias: {
            auto &ta = static_cast<const TypeAliasDecl &>(d);
            JObj extra;
            extra.str("type", typeStr(ta.aliasedType));
            I.addSimple(d, "typedef", "", extra.done());
            break;
        }
        case DeclKind::Newtype: {
            auto &nt = static_cast<const NewtypeDecl &>(d);
            JObj extra;
            extra.str("type", typeStr(nt.baseType));
            I.addSimple(d, "newtype", "newtype " + nt.name + " = " + typeStr(nt.baseType), extra.done());
            break;
        }
        case DeclKind::Region: {
            auto &rd = static_cast<const RegionDecl &>(d);
            JObj extra;
            extra.num("capacity", rd.capacity);
            I.addSimple(d, "region", "region " + rd.name + " { capacity: " + std::to_string(rd.capacity) + " }",
                        extra.done());
            break;
        }
        default: break;
        }
    }
    for (auto &[name, m] : macros) I.addMacro(m);
}

std::string DocCollector::json() {
    Impl &I = *impl_;
    I.merge();
    // Items in a stable order: crate, file, line, name.
    std::stable_sort(I.items.begin(), I.items.end(), [](const Impl::Item &a, const Impl::Item &b) {
        if (a.crate != b.crate) return a.crate < b.crate;
        if (a.file != b.file) return a.file < b.file;
        if (a.line != b.line) return a.line < b.line;
        return a.name < b.name;
    });
    std::string out = "{\"format\":\"safec-doc\",\"version\":1,\n\"files\":[\n";
    bool first = true;
    for (auto &rel : I.localFiles) {
        // The file's real path: the first root that holds it.
        std::string full;
        for (auto &r : I.roots) {
            std::string cand = r + "/" + rel;
            if (llvm::sys::fs::exists(cand)) { full = cand; break; }
        }
        if (I.roots.empty()) full = rel;
        JObj o;
        o.str("path", rel).flag("header", Impl::isHeader(rel));
        if (!full.empty()) {
            DocText d = I.moduleDoc(I.source(Impl::real(full)));
            applyHidden(d);
            if (!d.text.empty()) o.str("doc", d.text).flag("md", d.markdown);
            o.flag("hidden", d.hidden);
        }
        out += (first ? "" : ",\n") + o.done();
        first = false;
    }
    out += "\n],\n\"externs\":[";
    for (size_t i = 0; i < I.externs.size(); ++i) {
        out += i ? "," : "";
        jsonEscape(out, I.externs[i].first);
    }
    out += "],\n\"items\":[\n";
    for (size_t i = 0; i < I.items.size(); ++i) {
        out += (i ? ",\n" : "") + I.itemJson(I.items[i]);
    }
    out += "\n]}\n";
    return out;
}

} // namespace safec
