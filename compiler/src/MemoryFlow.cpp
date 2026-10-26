// ── Heap/arena lifetime analysis ──────────────────────────────────────────────
// Use-after-free, double-free, use-after-arena_reset/destroy and
// use-after-arena_free_to checking, run by Sema::checkMemoryFlow once every
// function body has been type-checked (it needs resolved identifiers, call
// targets and expression types, and must see monomorphized instances).
//
// It is an abstract interpretation over each function body:
//
//  * Objects. Every storage location the checker can name is an abstract
//    object: a local or global variable, a heap/arena allocation site, an
//    "entry" object (whatever a parameter or global pointed to when the
//    function was entered, named by the access path that reached it, e.g.
//    'p0*.data' = the object the 'data' field of *param0 pointed to), an
//    arena mark, a temporary (call arguments/results, compound literals) and
//    a view (a pointer into the middle of another object, '&x.f'), and one
//    "trusted" object standing for memory reached through a raw pointer or
//    unsafe pointer indexing (a container's buffer): stores there are not
//    modelled and every load from it is a fresh, untracked pointer. An
//    allocation site has three objects: the most recent allocation, which
//    can be updated strongly, the earlier ones of the current iteration of
//    the site's outermost loop, and those of earlier iterations (allocating
//    again folds the first into the second; starting the loop's next
//    iteration folds both into the third).
//
//  * State. For each object, the values stored in its cells, keyed by an
//    access path ('.field', '[3]', '[*]' for an unknown index — a store
//    through an unknown index is a weak update of every element, a load
//    reads every element). A value is the set of objects a pointer may
//    point to, and each pointer carries its own "may be dangling" bits
//    (freed by std::dealloc, or invalidated by an arena reset/free_to), so
//    joining two paths keeps "q points to the live node on this path and to
//    the freed one only on that path" apart. Aggregates are copied cell by
//    cell; a copy of entry memory that was never written is recorded as a
//    lazy reference to the entry path rather than enumerated. Also tracked:
//    marks taken per region, which marks each arena allocation follows (an
//    arena_free_to(m) invalidates exactly the allocations made after m —
//    per mark, so nested marks are precise), and which regions were reset
//    since entry (invalidating entry objects of that region too).
//
//  * Flow. Statements are interpreted in order; if/else, switch, match,
//    '?:', '&&'/'||' are joined at their merge points (a condition is split
//    into the states where it is true and false, through '&&', '||', '!',
//    ',' and constants), break/continue/goto/
//    return/'try' carry their state to their targets, defers run where
//    CodeGen runs them (at each return, in reverse registration order), and
//    each loop (and a function with labels) is iterated until its head state
//    stops growing. Diagnostics are only collected on the final pass over
//    a converged loop, so every report is against the fixpoint.
//
//  * Calls. Each function with a body gets a summary, computed bottom-up
//    over the call graph's strongly connected components (iterated to a
//    fixpoint within a recursive component): which entry objects it may
//    free, which regions it may reset (or free_to through a mark it was
//    passed), which entry pointers it reads, and what it leaves in its
//    parameters' and globals' memory and in its return value — each in terms
//    of entry paths and fresh allocations. A call site maps those paths onto
//    its own arguments. Calls through function pointers and to functions
//    with no body are trusted: they free nothing and return fresh,
//    unaliased memory.
//
// 'unsafe { }' suppresses the reports inside it (its frees/resets still
// count). Inside one function, distinct parameters and globals are assumed
// not to share an allocation unless the function itself makes them.

#include "safec/Sema.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <functional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace safec {
namespace {

constexpr uint8_t kFreed = 1;   // std::dealloc() may have freed it
constexpr uint8_t kReset = 2;   // an arena reset/destroy/free_to may have invalidated it
constexpr int kMaxDerefs = 3;   // deeper entry paths share one collapsed object
constexpr int kMaxIters  = 200;

// Ownership of must-drop values (see the ownership section below): the
// state of each owned slot (a local's leaf value, keyed by object + cell
// path).
constexpr uint8_t kOLive   = 1;   // holds a value that must still be dropped
constexpr uint8_t kODrop   = 2;   // dropped (drop() or a 'consume' method)
constexpr uint8_t kOMove   = 4;   // moved out (returned, passed by value, stored elsewhere, sent)
constexpr uint8_t kOUninit = 8;   // declared without an initializer
constexpr uint8_t kOBuild  = 16;  // with kOLive: being built field by field (its fields may be assigned)
constexpr uint8_t kOSub    = 32;  // a field of a Drop value, dropped/moved out until it is assigned again
constexpr uint8_t kOGone   = kODrop | kOMove;
using OwnKey = std::pair<int, std::string>;

// ── Values ─────────────────────────────────────────────────────────────────────
struct VEnt { int obj; uint8_t bits; };
inline bool operator==(const VEnt &a, const VEnt &b) { return a.obj == b.obj && a.bits == b.bits; }
using VSet = std::vector<VEnt>;   // sorted by obj

void vsAdd(VSet &s, int obj, uint8_t bits) {
    auto it = std::lower_bound(s.begin(), s.end(), obj,
                               [](const VEnt &e, int o) { return e.obj < o; });
    if (it != s.end() && it->obj == obj) it->bits |= bits;
    else s.insert(it, VEnt{obj, bits});
}
void vsUnion(VSet &d, const VSet &s) { for (auto &e : s) vsAdd(d, e.obj, e.bits); }
uint8_t vsBits(const VSet &s) { uint8_t b = 0; for (auto &e : s) b |= e.bits; return b; }

template <class T> void sortedUnion(std::vector<T> &d, const std::vector<T> &s) {
    for (auto &x : s) {
        auto it = std::lower_bound(d.begin(), d.end(), x);
        if (it == d.end() || *it != x) d.insert(it, x);
    }
}

// ── Access paths ───────────────────────────────────────────────────────────────
// Cell paths: '.name', '.|' (any member of a union), '[k]', '[*]'. Entry
// paths (EPs) add a root ('p<i>' parameter i, 'g:<name>' global) and '*'
// (follow the pointer stored there); '~' ends a collapsed path ("anything
// reachable from here").
bool underPrefix(const std::string &p, const std::string &pre) {
    if (pre.empty()) return true;
    if (p.size() < pre.size() || p.compare(0, pre.size(), pre) != 0) return false;
    return p.size() == pre.size() || p[pre.size()] == '.' || p[pre.size()] == '[';
}

std::string starAll(const std::string &p) {
    if (p.find('[') == std::string::npos) return p;
    std::string out;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] == '[') {
            size_t j = p.find(']', i);
            if (j == std::string::npos) { out += p.substr(i); break; }
            out += "[*]";
            i = j;
        } else out += p[i];
    }
    return out;
}

// At most one element of a multi-dimensional index is kept apart.
std::string normPath(const std::string &p) {
    int n = 0;
    for (char c : p) if (c == '[') ++n;
    return n > 1 ? starAll(p) : p;
}

bool hasStar(const std::string &p) { return p.find("[*]") != std::string::npos; }

std::string canonEP(const std::string &e0) {
    std::string e = starAll(e0);
    size_t t = e.find('~');
    if (t != std::string::npos) return e.substr(0, t + 1);
    int n = 0;
    for (size_t i = 0; i < e.size(); ++i)
        if (e[i] == '*' && ++n > kMaxDerefs) return e.substr(0, i) + "~";
    return e;
}

// Is entry path 'e' at or below the prefix 'd'?
bool epUnder(const std::string &e0, const std::string &d) {
    std::string e = (!e0.empty() && e0.back() == '~') ? e0.substr(0, e0.size() - 1) + "*" : e0;
    if (e.size() < d.size() || e.compare(0, d.size(), d) != 0) return false;
    return e.size() == d.size() || d.back() == '*' || std::strchr(".[*~", e[d.size()]);
}

bool epRooted(const std::string &e) {
    return (e.size() > 1 && e[0] == 'p' && isdigit((unsigned char)e[1])) || e.rfind("g:", 0) == 0;
}

std::string ptrKey(const void *p) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%p", p);
    return buf;
}

// ── Types ──────────────────────────────────────────────────────────────────────
TypePtr stripNewtype(TypePtr t) {
    while (t && t->kind == TypeKind::Newtype) t = static_cast<NewtypeType &>(*t).base;
    return t;
}
bool isAgg(const TypePtr &t0) {
    TypePtr t = stripNewtype(t0);
    if (!t) return false;
    if (t->kind == TypeKind::Struct || t->kind == TypeKind::Array || t->kind == TypeKind::Tuple) return true;
    if (t->kind == TypeKind::Optional) return isAgg(static_cast<OptionalType &>(*t).inner);
    return false;
}
// Values of these types can carry an object identity: references, pointers,
// slices, and 'unsigned long' (what arena_mark<R>() returns).
bool isTracked(const TypePtr &t0) {
    TypePtr t = stripNewtype(t0);
    if (!t) return false;
    switch (t->kind) {
    case TypeKind::Reference: case TypeKind::Pointer: case TypeKind::Slice: case TypeKind::UInt64:
        return true;
    case TypeKind::Optional: return isTracked(static_cast<OptionalType &>(*t).inner);
    default: return false;
    }
}
const ReferenceType *refOf(const TypePtr &t0) {
    TypePtr t = stripNewtype(t0);
    if (!t) return nullptr;
    if (t->kind == TypeKind::Reference) return static_cast<const ReferenceType *>(t.get());
    if (t->kind == TypeKind::Optional) return refOf(static_cast<OptionalType &>(*t).inner);
    return nullptr;
}
bool isRawPtr(const TypePtr &t0) {
    TypePtr t = stripNewtype(t0);
    if (t && t->kind == TypeKind::Optional) t = stripNewtype(static_cast<OptionalType &>(*t).inner);
    return t && t->kind == TypeKind::Pointer;
}
const StructType *structOf(TypePtr t) {
    t = stripNewtype(t);
    while (t && (t->kind == TypeKind::Reference || t->kind == TypeKind::Pointer || t->kind == TypeKind::Optional)) {
        if (t->kind == TypeKind::Reference) t = static_cast<ReferenceType &>(*t).base;
        else if (t->kind == TypeKind::Pointer) t = static_cast<PointerType &>(*t).base;
        else t = static_cast<OptionalType &>(*t).inner;
        t = stripNewtype(t);
    }
    return t && t->kind == TypeKind::Struct ? static_cast<const StructType *>(t.get()) : nullptr;
}
bool isUnionType(const StructType *st) { return st && (st->isUnion || st->isTaggedUnion); }

// The cell-path selector for 'base.field' ('.|' inside a union, through
// anonymous members too).
std::string fieldSel(const TypePtr &baseTy, const std::string &field) {
    const StructType *st = structOf(baseTy);
    if (!st) return "." + field;
    std::vector<int> path;
    if (!st->findFieldPath(field, path)) return isUnionType(st) ? ".|" : "." + field;
    std::string sel;
    const StructType *cur = st;
    for (int idx : path) {
        if (!cur) break;
        const FieldDecl *fd = nullptr;
        for (auto &f : cur->fields) if (f.index == idx) { fd = &f; break; }
        if (!fd) break;
        if (isUnionType(cur)) sel += ".|";
        else sel += "." + (fd->name.empty() ? "#" + std::to_string(idx) : fd->name);
        cur = structOf(fd->type);
    }
    return sel;
}

// ── Objects ────────────────────────────────────────────────────────────────────
enum class OK : uint8_t { Local, Global, Ext, Alloc, Mark, Temp, Ret, View, Trusted, Token };

struct ObjInfo {
    OK          kind = OK::Local;
    std::string ep;          // Ext: canonical entry path; Global: its root ('g:name')
    int         param = -1;  // Local: parameter index
    bool        collapsed = false;
    int         tier = 0;    // Alloc: 0 the latest, 1 earlier ones, 2 ones from earlier loop iterations
    const Stmt *loop = nullptr; // Alloc: outermost loop around the site
    bool        heap = false;
    std::string region;      // arena region, when known
    std::string site;        // Alloc/Mark: stable key used in summaries
    int         base = -1;   // View
    std::string prefix;      // View
    bool        exprTemp = false; // Temp: a full-expression temporary (not a match binding)
};
ObjInfo objInfo(OK k) { ObjInfo oi; oi.kind = k; return oi; }

// ── Summaries ──────────────────────────────────────────────────────────────────
// A value as a callee leaves it, in terms its caller can resolve: 'E' the
// object an entry path pointed to, 'G' a global variable, 'F' a fresh
// allocation (or untracked call result) made during the call, 'M' a mark
// taken during it. 'prefix' makes it a view into that object.
struct Term {
    char        kind = 'E';
    std::string key;
    std::string prefix;
    uint8_t     bits = 0;
    bool        heap = false;
    std::string region;
    bool operator<(const Term &o) const {
        return std::tie(kind, key, prefix, bits, heap, region) <
               std::tie(o.kind, o.key, o.prefix, o.bits, o.heap, o.region);
    }
    bool operator==(const Term &o) const {
        return kind == o.kind && key == o.key && prefix == o.prefix && bits == o.bits &&
               heap == o.heap && region == o.region;
    }
    bool operator!=(const Term &o) const { return !(*this == o); }
};

struct FactCells {
    std::map<std::string, std::vector<Term>> vals;          // cell path -> values
    std::map<std::string, std::vector<std::string>> lazy;   // prefix -> entry paths copied there
    std::vector<Term> any;                                   // possibly stored anywhere in it
    bool operator==(const FactCells &o) const { return vals == o.vals && lazy == o.lazy && any == o.any; }
};

struct Summary {
    std::set<std::string> frees;                              // entry objects it may free
    std::set<std::string> resets;                             // regions it may reset/destroy
    std::set<std::pair<std::string, std::string>> freeTos;    // (region, mark's entry path)
    std::set<std::string> uses;                               // entry cells it reads as references
    std::map<std::string, FactCells> writes;                  // designator -> its cells at exit
    FactCells ret;
    bool operator==(const Summary &o) const {
        return frees == o.frees && resets == o.resets && freeTos == o.freeTos && uses == o.uses &&
               writes == o.writes && ret == o.ret;
    }
};

// ── State ──────────────────────────────────────────────────────────────────────
struct ObjCells {
    std::map<std::string, VSet> vals;
    std::map<std::string, std::vector<std::string>> lazy;
    VSet any;
    bool operator==(const ObjCells &o) const { return vals == o.vals && lazy == o.lazy && any == o.any; }
};

struct Deep { uint8_t bits = 0; VSet vals; bool operator==(const Deep &o) const { return bits == o.bits && vals == o.vals; } };

struct State {
    bool live = false;
    std::map<int, ObjCells> mem;
    std::map<int, uint8_t> invalid;                     // objects that may be dangling
    std::set<std::string> extReset;                      // regions reset since entry
    std::map<std::string, Deep> deep;                    // entry-path prefixes freed/written as a whole
    std::map<int, std::vector<int>> after;               // arena allocation -> marks taken before it
    std::map<std::string, std::vector<int>> marks;       // region -> marks taken so far
    std::map<OwnKey, uint8_t> own;                       // owned slot -> kO* bits
    std::map<int, std::vector<OwnKey>> tok;              // borrow token -> the owned slots it borrows
    std::map<int, std::vector<int>> lent;                // object lent to a scoped thread -> handle tokens
    bool operator==(const State &o) const {
        if (live != o.live) return false;
        if (!live) return true;
        return mem == o.mem && invalid == o.invalid && extReset == o.extReset && deep == o.deep &&
               after == o.after && marks == o.marks && own == o.own && tok == o.tok && lent == o.lent;
    }
    bool operator!=(const State &o) const { return !(*this == o); }
};

struct CellRef { int obj; std::string path; uint8_t bits = 0; bool weak = false; };

struct EV {
    VSet v;
    std::vector<CellRef> cells;   // aggregates: where the value lives
    bool agg = false;
};

struct Diag {
    SourceLocation loc;
    std::string msg;
    std::vector<std::pair<SourceLocation, std::string>> notes;
};

class FlowDriver;

// ── Per-function analysis ──────────────────────────────────────────────────────
class Walker {
public:
    Walker(FlowDriver &d, FunctionDecl &fn) : drv_(d), fn_(fn) {}
    void run();
    Summary              summary;
    std::vector<Diag>    diags;

private:
    FlowDriver   &drv_;
    FunctionDecl &fn_;
    std::vector<ObjInfo> objs_;
    std::unordered_map<std::string, int> byKey_;
    std::unordered_map<const VarDecl *, int> paramIdx_;
    State cur_, exit_;
    int quiet_ = 0, unsafe_ = 0;
    struct Target { std::string label; bool isSwitch; State brk, cont; size_t depth = 0; };
    std::vector<Target> targets_;
    std::map<std::string, State> labels_;
    std::vector<std::pair<int, const DeferStmt *>> defers_;   // (preorder index, stmt)
    std::unordered_map<const Stmt *, int> order_;
    std::unordered_map<const Stmt *, int> blockEnd_;   // a block's last preorder index
    int curPos_ = 0;
    int inDefer_ = 0;
    std::vector<std::pair<std::string, int>> binds_;
    int retObj_ = -1;
    bool recordUses_ = false;

    // objects
    int intern(const std::string &key, const ObjInfo &info);
    int varObj(const VarDecl *vd);
    int extObj(const std::string &cellEP);
    int viewObj(int base, std::string prefix);
    int allocObj(const std::string &site, int tier, bool heap, const std::string &region, const Stmt *loop);
    int siteObj(const Expr &where, const std::string &site, int tier, bool heap, const std::string &region) {
        auto it = outerLoop_.find(&where);
        return allocObj(site, tier, heap, region, it == outerLoop_.end() ? nullptr : it->second);
    }
    int markObj(const std::string &site, const std::string &region);
    int tempObj(const void *p, int tag);
    void dropTemps(State &st);
    int trustedObj() { return intern("U", objInfo(OK::Trusted)); }
    bool isTrusted(int o) const { return objs_[o].kind == OK::Trusted; }
    int canonBase(int o) const { return objs_[o].kind == OK::View ? objs_[o].base : o; }
    bool strongObj(int o) const;
    CellRef cellOf(const VEnt &e, const std::string &sel) const;

    // state primitives
    std::vector<std::string> defaultEPs(const State &st, int obj, const std::string &path) const;
    VSet lookup(const State &st, int obj, const std::string &path, const ReferenceType *rt = nullptr);
    void addExt(const State &st, VSet &out, const std::string &cellEP, const ReferenceType *rt);
    void writeCell(int obj, std::string path, const VSet &v, bool strong);
    void write(const std::vector<CellRef> &cells, const VSet &v);
    void clearSubtree(int obj, const std::string &prefix);
    void copyAgg(const std::vector<CellRef> &dst, const std::vector<CellRef> &src);
    void taint(const std::set<int> &bases, uint8_t bits);
    void addDeep(const std::string &prefix, uint8_t bits, const VSet &vals);
    void rename(State &st, const std::map<int, int> &m);
    void foldIteration(State &st, const Stmt *loop);
    std::unordered_map<const Expr *, const Stmt *> outerLoop_;
    void freshAlloc(int mr, int old, const std::string &region);
    void freeObjs(const VSet &v);
    void resetRegion(const std::string &region, bool fromUnknownMark);
    void freeTo(const std::string &region, const VSet &marks);
    State join(const State &a, const State &b);

    // statements
    void walkBody();
    void stmt(Stmt &s);
    void loop(const Stmt &s, Expr *cond, Stmt *body, Expr *incr, bool doWhile, const std::string &label);
    void loopOnce(Expr *cond, Stmt *body, Expr *incr, bool doWhile, const std::string &label,
                  bool alwaysTrue, State &back, State &exitS);
    void doReturn(bool isError);
    void jumpTo(bool isBreak, const std::string &label, SourceLocation loc);
    void bindPattern(const MatchPattern &pat, const EV &subject, const TypePtr &subjTy);

    // expressions
    EV rval(Expr &e);
    void cond(Expr *c, State &t, State &f);
    std::vector<CellRef> lval(Expr &e);
    VSet load(const std::vector<CellRef> &cells, const TypePtr &ty, const Expr *site = nullptr);
    void checkLoad(const Expr &e, const VSet &v, int form, const std::string &name);
    EV evalCall(CallExpr &e);
    EV evalCallInner(CallExpr &e);
    SourceLocation freeLoc_;
    std::vector<OwnKey> callOwners_;
    EV opaqueResult(const Expr &site, const TypePtr &ty);
    void writeInit(const std::vector<CellRef> &dst, Expr &init, const TypePtr &ty);
    EV applySummary(const Expr &e, FunctionDecl *callee, const Summary &s, std::vector<EV> &args);
    struct Res { std::vector<CellRef> cells; std::vector<std::string> deep; };
    Res resolveCells(const std::string &ep, const std::vector<int> &argObjs);
    VSet resolveObjs(const std::string &ep, const std::vector<int> &argObjs, std::vector<std::string> *deep);
    VSet mapTerms(const std::vector<Term> &terms, const std::vector<int> &argObjs, const Expr &call,
                  FunctionDecl *callee, std::vector<std::string> *deep);
    void closure(const std::vector<CellRef> &cells, std::vector<CellRef> &outCells, std::vector<std::string> &deep);
    std::string calleeSite(const Expr &call, FunctionDecl *callee, const std::string &key);
    void report(SourceLocation loc, std::string msg) {
        if (!quiet_ && !unsafe_) diags.push_back({loc, std::move(msg)});
    }
    void buildSummary();
    Term termOf(const VEnt &e) const;
    void reportN(SourceLocation loc, std::string msg, std::vector<std::pair<SourceLocation, std::string>> notes) {
        if (!quiet_ && !unsafe_) diags.push_back({loc, std::move(msg), std::move(notes)});
    }

    // ── ownership of must-drop values ──────────────────────────────────────
    struct OwnVar {
        TypePtr ty;
        std::string name;
        SourceLocation decl;
        const VarDecl *vd = nullptr;
        bool pseudo = false;   // *self of a destructor ('consume' method)
    };
    std::unordered_map<int, OwnVar> ownVars_;
    struct Frame { const Stmt *s; std::vector<int> objs; std::vector<int> locals; };
    // Scoped threads: the referents lent to each (State::lent) and where.
    struct SpawnInfo { SourceLocation loc; bool mut; bool sync; };   // sync: a synchronized type, still usable here
    std::unordered_map<int, SpawnInfo> spawns_;               // token -> spawn site
    std::unordered_map<const Expr *, int> spawnTok_;          // spawn_scoped call -> token
    std::unordered_map<int, int> handleTok_;                  // handle variable -> token
    struct LocalInfo { std::string name; SourceLocation decl; };
    std::unordered_map<int, LocalInfo> locals_;               // every local declared
    void lendCheck(int obj, SourceLocation loc, bool write);
    void lendExit(int obj, SourceLocation loc);
    std::string objName(int obj) const;
    std::vector<Frame> frames_;
    std::vector<const Expr *> pending_;                 // must-drop temporaries not yet consumed
    std::map<OwnKey, SourceLocation> goneAt_;           // where a slot was last dropped/moved
    std::vector<std::set<const VarDecl *>> deferVars_;  // parallel to defers_: variables each defer names
    std::unordered_map<std::string, std::set<const Stmt *>> labelScopes_; // label -> enclosing blocks
    bool isDtor_ = false;          // fn_ is drop() or a 'consume' method
    bool refDropReported_ = false;
    void refDropCheck(const Expr *place, SourceLocation loc);
    int selfObj_ = -1;             // the object *self is, in a destructor
    std::set<std::string> dtorFields_;   // fields of *self a destructor drops or moves
    SourceLocation endLoc_;
    bool mustDrop(const TypePtr &t) const;
    bool tracked(int obj) const { return ownVars_.count(obj) != 0; }
    void ownLeaves(const TypePtr &t, const std::string &path, std::vector<std::string> &out, int depth = 0);
    void ownDeclare(int obj, const TypePtr &ty, bool init, SourceLocation loc, const std::string &name,
                    const VarDecl *vd);
    std::vector<std::map<OwnKey, uint8_t>::iterator> ownOverlap(int obj, const std::string &path);
    void ownUse(const std::vector<CellRef> &cells, SourceLocation loc, bool write);
    void ownMove(Expr &e, const EV &v, uint8_t kind);
    void ownConsumeCells(const std::vector<CellRef> &cells, uint8_t kind, SourceLocation loc, const Expr *what);
    void ownConsumeKey(int obj, const std::string &path, uint8_t kind, SourceLocation loc);
    void ownAssign(const std::vector<CellRef> &cells, SourceLocation loc);
    void ownTemp(const Expr &e) { if (cur_.live && mustDrop(e.type)) pending_.push_back(&e); }
    void ownTakeTemp(const Expr *e);
    void ownFlush();
    void ownExitFrame(const Frame &f, SourceLocation loc, const char *how);
    void ownExitAbove(size_t depth, SourceLocation loc, const char *how);
    void ownExitForGoto(const std::string &label, SourceLocation loc);
    bool keptByDefer(int obj) const;
    std::string ownDesc(int obj, const std::string &path) const;
    std::string danglingIn(const VSet &v) const;
    void ownCheckVal(const Expr &e, const VSet &v);
    void ownCheckObjTokens(int obj, SourceLocation loc);
    int ownToken(const Expr &site, const std::vector<OwnKey> &owners);
    void ownEndFunction();
    bool ownMatchSubject(Expr *subject, const EV &subj, const std::vector<const std::vector<MatchPattern> *> &arms);
    void ownBind(const MatchPattern &p, SourceLocation loc);
    std::string typeName(const TypePtr &t) const;
};

// ── Driver ─────────────────────────────────────────────────────────────────────
class FlowDriver {
public:
    FlowDriver(DiagEngine &diag, const std::vector<FunctionDecl *> &fns, const Sema &sema)
        : diag_(diag), fns_(fns), sema(sema) {}
    const Sema &sema;
    void run();
    FunctionDecl *bodyOf(FunctionDecl *fd) const {
        if (!fd) return nullptr;
        if (fd->body && checked_.count(fd)) return fd;
        auto it = byName_.find(fd->name);
        return it == byName_.end() ? nullptr : it->second;
    }
    const Summary *summaryOf(FunctionDecl *fd) const {
        auto it = sums_.find(fd);
        return it == sums_.end() ? nullptr : &it->second;
    }
    bool sameScc(FunctionDecl *a, FunctionDecl *b) const {
        auto ia = scc_.find(a), ib = scc_.find(b);
        return ia != scc_.end() && ib != scc_.end() && ia->second == ib->second;
    }

private:
    DiagEngine &diag_;
    const std::vector<FunctionDecl *> &fns_;
    std::unordered_map<std::string, FunctionDecl *> byName_;
    std::set<const FunctionDecl *> checked_;
    std::unordered_map<FunctionDecl *, Summary> sums_;
    std::unordered_map<FunctionDecl *, int> scc_;
};

bool Walker::mustDrop(const TypePtr &t) const { return drv_.sema.isMustDrop(t); }
std::string Walker::typeName(const TypePtr &t) const { return drv_.sema.displayType(t); }

// ── Generic AST traversal (call graph, labels, statement order) ───────────────
template <class FS, class FE>
void visitStmt(Stmt *s, FS &&onStmt, FE &&onExpr);

template <class FS, class FE>
void visitExpr(Expr *e, FS &&onStmt, FE &&onExpr) {
    if (!e) return;
    onExpr(*e);
    auto ve = [&](Expr *x) { visitExpr(x, onStmt, onExpr); };
    switch (e->kind) {
    case ExprKind::Unary: case ExprKind::AddrOf: case ExprKind::Deref:
        ve(static_cast<UnaryExpr *>(e)->operand.get()); break;
    case ExprKind::Binary: {
        auto *b = static_cast<BinaryExpr *>(e); ve(b->left.get()); ve(b->right.get()); break;
    }
    case ExprKind::Ternary: {
        auto *t = static_cast<TernaryExpr *>(e); ve(t->cond.get()); ve(t->then.get()); ve(t->else_.get()); break;
    }
    case ExprKind::Call: {
        auto *c = static_cast<CallExpr *>(e);
        ve(c->callee.get()); ve(c->methodBase.get()); ve(c->nullOpBase.get());
        for (auto &a : c->args) ve(a.get());
        break;
    }
    case ExprKind::Subscript: {
        auto *s = static_cast<SubscriptExpr *>(e); ve(s->base.get()); ve(s->index.get()); break;
    }
    case ExprKind::Member: case ExprKind::Arrow: ve(static_cast<MemberExpr *>(e)->base.get()); break;
    case ExprKind::Cast: ve(static_cast<CastExpr *>(e)->operand.get()); break;
    case ExprKind::Assign: {
        auto *a = static_cast<AssignExpr *>(e); ve(a->lhs.get()); ve(a->rhs.get()); break;
    }
    case ExprKind::Compound: for (auto &i : static_cast<CompoundInitExpr *>(e)->inits) ve(i.get()); break;
    case ExprKind::New: ve(static_cast<NewExpr *>(e)->arraySize.get()); break;
    case ExprKind::TupleLit: for (auto &i : static_cast<TupleLitExpr *>(e)->elements) ve(i.get()); break;
    case ExprKind::Spawn: {
        auto *s = static_cast<SpawnExpr *>(e); ve(s->fnExpr.get()); ve(s->argExpr.get()); break;
    }
    case ExprKind::Try: ve(static_cast<TryExpr *>(e)->inner.get()); break;
    case ExprKind::Slice: {
        auto *s = static_cast<SliceExpr *>(e); ve(s->base.get()); ve(s->start.get()); ve(s->end.get()); break;
    }
    case ExprKind::TaggedUnionInit: ve(static_cast<TaggedUnionInitExpr *>(e)->payload.get()); break;
    case ExprKind::GenericSelection: {
        auto *g = static_cast<GenericSelectionExpr *>(e);
        if (g->selectedIndex >= 0 && g->selectedIndex < (int)g->associations.size())
            ve(g->associations[g->selectedIndex].expr.get());
        break;
    }
    case ExprKind::Match: {
        auto *m = static_cast<MatchExpr *>(e); ve(m->subject.get());
        for (auto &a : m->arms) ve(a.value.get());
        break;
    }
    default: break;
    }
}

template <class FS, class FE>
void visitStmt(Stmt *s, FS &&onStmt, FE &&onExpr) {
    if (!s) return;
    onStmt(*s);
    auto vs = [&](Stmt *x) { visitStmt(x, onStmt, onExpr); };
    auto ve = [&](Expr *x) { visitExpr(x, onStmt, onExpr); };
    switch (s->kind) {
    case StmtKind::Compound: for (auto &c : static_cast<CompoundStmt *>(s)->body) vs(c.get()); break;
    case StmtKind::Expr: ve(static_cast<ExprStmt *>(s)->expr.get()); break;
    case StmtKind::If: {
        auto *i = static_cast<IfStmt *>(s); ve(i->cond.get()); vs(i->then.get()); vs(i->else_.get()); break;
    }
    case StmtKind::While: case StmtKind::DoWhile: {
        auto *w = static_cast<WhileStmt *>(s); ve(w->cond.get()); vs(w->body.get()); break;
    }
    case StmtKind::For: {
        auto *f = static_cast<ForStmt *>(s);
        vs(f->init.get()); ve(f->cond.get()); ve(f->incr.get()); vs(f->body.get()); break;
    }
    case StmtKind::Return: ve(static_cast<ReturnStmt *>(s)->value.get()); break;
    case StmtKind::Label: vs(static_cast<LabelStmt *>(s)->body.get()); break;
    case StmtKind::VarDecl: ve(static_cast<VarDeclStmt *>(s)->init.get()); break;
    case StmtKind::MultiVarDecl: for (auto &d : static_cast<MultiVarDeclStmt *>(s)->decls) vs(d.get()); break;
    case StmtKind::Unsafe: vs(static_cast<UnsafeStmt *>(s)->body.get()); break;
    case StmtKind::IfConst: {
        auto *i = static_cast<IfConstStmt *>(s); ve(i->cond.get()); vs(i->then.get()); vs(i->else_.get()); break;
    }
    case StmtKind::Defer: vs(static_cast<DeferStmt *>(s)->body.get()); break;
    case StmtKind::Match: {
        auto *m = static_cast<MatchStmt *>(s); ve(m->subject.get());
        for (auto &a : m->arms) vs(a.body.get());
        break;
    }
    case StmtKind::Switch: {
        auto *w = static_cast<SwitchStmt *>(s); ve(w->controlling.get());
        for (auto &c : w->cases) for (auto &b : c.body) vs(b.get());
        break;
    }
    case StmtKind::Asm: {
        auto *a = static_cast<AsmStmt *>(s);
        for (auto &x : a->outputExprs) ve(x.get());
        for (auto &x : a->inputExprs) ve(x.get());
        break;
    }
    default: break;
    }
}

// ── Objects ────────────────────────────────────────────────────────────────────
int Walker::intern(const std::string &key, const ObjInfo &info) {
    auto it = byKey_.find(key);
    if (it != byKey_.end()) return it->second;
    objs_.push_back(info);
    int id = (int)objs_.size() - 1;
    byKey_.emplace(key, id);
    return id;
}

int Walker::varObj(const VarDecl *vd) {
    if (vd->isGlobal || vd->isStatic) {
        std::string root = "g:" + vd->name + (vd->isGlobal ? "" : "@" + ptrKey(vd));
        ObjInfo oi = objInfo(OK::Global);
        oi.ep = root;
        return intern("G" + root, oi);
    }
    ObjInfo oi = objInfo(OK::Local);
    auto it = paramIdx_.find(vd);
    if (it != paramIdx_.end()) oi.param = it->second;
    return intern("L" + ptrKey(vd), oi);
}

int Walker::extObj(const std::string &cellEP) {
    std::string k = canonEP(cellEP);
    ObjInfo oi = objInfo(OK::Ext);
    oi.ep = k;
    oi.collapsed = !k.empty() && k.back() == '~';
    return intern("E" + k, oi);
}

int Walker::viewObj(int base, std::string prefix) {
    if (objs_[base].kind == OK::Trusted) return base;
    if (objs_[base].kind == OK::View) {
        prefix = objs_[base].prefix + prefix;
        base = objs_[base].base;
    }
    if (prefix.empty()) return base;
    prefix = normPath(prefix);
    ObjInfo oi = objInfo(OK::View);
    oi.base = base;
    oi.prefix = prefix;
    oi.heap = objs_[base].heap;
    oi.region = objs_[base].region;
    return intern("V" + std::to_string(base) + "|" + prefix, oi);
}

int Walker::allocObj(const std::string &site, int tier, bool heap, const std::string &region, const Stmt *loop) {
    ObjInfo oi = objInfo(OK::Alloc);
    oi.site = site;
    oi.tier = tier;
    oi.heap = heap;
    oi.region = region;
    oi.loop = loop;
    return intern("A" + std::to_string(tier) + site, oi);
}

int Walker::markObj(const std::string &site, const std::string &region) {
    ObjInfo oi = objInfo(OK::Mark);
    oi.site = site;
    oi.region = region;
    return intern("M" + site, oi);
}

int Walker::tempObj(const void *p, int tag) {
    ObjInfo oi = objInfo(OK::Temp);
    oi.exprTemp = tag != 2;
    return intern("T" + ptrKey(p) + ":" + std::to_string(tag), oi);
}

// Expression temporaries (call results, compound literals, materialized
// rvalues) die with the full expression that made them. Dropping them at
// each statement boundary keeps a temporary written on only one side of a
// join ('k ? f() : g()') from carrying its previous evaluation's contents —
// an allocation freed since — into the other side.
void Walker::dropTemps(State &st) {
    for (auto it = st.mem.begin(); it != st.mem.end();)
        it = objs_[it->first].exprTemp ? st.mem.erase(it) : std::next(it);
}

bool Walker::strongObj(int o) const {
    const ObjInfo &oi = objs_[o];
    switch (oi.kind) {
    case OK::Local: case OK::Global: case OK::Temp: case OK::Ret: return true;
    case OK::Ext:   return !oi.collapsed;
    case OK::Alloc: return oi.tier == 0;
    default:        return false;
    }
}

CellRef Walker::cellOf(const VEnt &e, const std::string &sel) const {
    const ObjInfo &oi = objs_[e.obj];
    if (oi.kind == OK::View) return {oi.base, normPath(oi.prefix + sel), e.bits};
    return {e.obj, normPath(sel), e.bits};
}

// ── State primitives ───────────────────────────────────────────────────────────
// The entry paths whose contents the cell (obj, path) still holds when it
// has not been written: a parameter's or global's own memory, an entry
// object's, or an aggregate copied from one of those.
std::vector<std::string> Walker::defaultEPs(const State &st, int obj, const std::string &path) const {
    auto it = st.mem.find(obj);
    if (it != st.mem.end() && !it->second.lazy.empty()) {
        const std::string *best = nullptr;
        const std::vector<std::string> *eps = nullptr;
        for (auto &kv : it->second.lazy)
            if (underPrefix(path, kv.first) && (!best || kv.first.size() > best->size())) {
                best = &kv.first;
                eps = &kv.second;
            }
        if (eps) {
            std::vector<std::string> out;
            for (auto &e : *eps) out.push_back(e + path.substr(best->size()));
            return out;
        }
    }
    const ObjInfo &oi = objs_[obj];
    switch (oi.kind) {
    case OK::Local:
        if (oi.param >= 0) return {"p" + std::to_string(oi.param) + path};
        return {};
    case OK::Global: return {oi.ep + path};
    case OK::Ext:    return {oi.collapsed ? oi.ep : oi.ep + "*" + path};
    default:         return {};
    }
}

void Walker::addExt(const State &st, VSet &out, const std::string &cellEP, const ReferenceType *rt) {
    int o = extObj(cellEP);
    ObjInfo &oi = objs_[o];
    if (rt && oi.region.empty() && !oi.heap) {
        if (rt->region == Region::Arena) oi.region = rt->arenaName;
        else if (rt->region == Region::Heap) oi.heap = true;
    }
    uint8_t bits = 0;
    auto iv = st.invalid.find(o);
    if (iv != st.invalid.end()) bits |= iv->second;
    if (!oi.region.empty() && st.extReset.count(oi.region)) bits |= kReset;
    for (auto &d : st.deep)
        if (epUnder(oi.ep, d.first)) {
            bits |= d.second.bits;
            vsUnion(out, d.second.vals);
        }
    vsAdd(out, o, bits);
    if (recordUses_ && !unsafe_) {
        std::string ce = canonEP(cellEP);
        bool bare = ce.size() > 1 && ce[0] == 'p' && ce.find_first_not_of("0123456789", 1) == std::string::npos;
        if (epRooted(ce) && !bare) summary.uses.insert(ce);
    }
}

VSet Walker::lookup(const State &st, int obj, const std::string &path, const ReferenceType *rt) {
    VSet out;
    if (objs_[obj].kind == OK::Trusted) return out;
    auto it = st.mem.find(obj);
    const ObjCells *oc = it == st.mem.end() ? nullptr : &it->second;
    if (oc) {
        vsUnion(out, oc->any);
        auto v = oc->vals.find(path);
        if (v != oc->vals.end()) {
            vsUnion(out, v->second);
            if (!hasStar(path)) return out;
        } else if (!hasStar(path)) {
            std::string sp = starAll(path);
            if (sp != path) {
                auto v2 = oc->vals.find(sp);
                if (v2 != oc->vals.end()) { vsUnion(out, v2->second); return out; }
            }
        }
        if (hasStar(path)) {
            // An unknown index reads every element written so far.
            for (auto &kv : oc->vals)
                if (kv.first != path && starAll(kv.first) == path) vsUnion(out, kv.second);
            if (v != oc->vals.end()) return out;
        }
    }
    for (auto &e : defaultEPs(st, obj, path)) addExt(st, out, e, rt);
    return out;
}

void Walker::writeCell(int obj, std::string path, const VSet &v, bool strong) {
    if (objs_[obj].kind == OK::Trusted) return;
    path = normPath(path);
    if (hasStar(path)) {
        VSet merged = lookup(cur_, obj, path);
        vsUnion(merged, v);
        auto &oc = cur_.mem[obj];
        for (auto &kv : oc.vals)
            if (kv.first != path && starAll(kv.first) == path) vsUnion(kv.second, v);
        oc.vals[path] = std::move(merged);
        return;
    }
    if (!strong || !strongObj(obj)) {
        VSet merged = lookup(cur_, obj, path);
        vsUnion(merged, v);
        cur_.mem[obj].vals[path] = std::move(merged);
        return;
    }
    if (v.empty()) {
        auto it = cur_.mem.find(obj);
        bool explicitHere = it != cur_.mem.end() && it->second.vals.count(path);
        if (!explicitHere && defaultEPs(cur_, obj, path).empty() &&
            (it == cur_.mem.end() || it->second.any.empty()))
            return;
    }
    cur_.mem[obj].vals[path] = v;
}

void Walker::write(const std::vector<CellRef> &cells, const VSet &v) {
    bool strong = cells.size() == 1 && !cells[0].weak;
    for (auto &c : cells) writeCell(c.obj, c.path, v, strong);
}

void Walker::clearSubtree(int obj, const std::string &prefix) {
    auto it = cur_.mem.find(obj);
    if (it == cur_.mem.end()) return;
    auto &oc = it->second;
    for (auto i = oc.vals.begin(); i != oc.vals.end();)
        i = underPrefix(i->first, prefix) ? oc.vals.erase(i) : std::next(i);
    for (auto i = oc.lazy.begin(); i != oc.lazy.end();)
        i = underPrefix(i->first, prefix) ? oc.lazy.erase(i) : std::next(i);
    if (prefix.empty()) oc.any.clear();
    if (oc.vals.empty() && oc.lazy.empty() && oc.any.empty()) cur_.mem.erase(it);
}

// dst := src for an aggregate: explicit cells are copied, unwritten entry
// memory is copied as a lazy reference to its entry path.
void Walker::copyAgg(const std::vector<CellRef> &dst, const std::vector<CellRef> &src) {
    if (dst.empty()) return;
    struct Snap {
        int obj; std::string path;
        std::vector<std::pair<std::string, VSet>> vals;
        std::vector<std::pair<std::string, std::vector<std::string>>> lazy;
        std::vector<std::string> def;
        VSet any;
    };
    std::vector<Snap> snaps;
    for (auto &s : src) {
        Snap sn{s.obj, s.path, {}, {}, {}, {}};
        auto it = cur_.mem.find(s.obj);
        if (it != cur_.mem.end()) {
            for (auto &kv : it->second.vals)
                if (underPrefix(kv.first, s.path)) sn.vals.push_back({kv.first.substr(s.path.size()), kv.second});
            for (auto &kv : it->second.lazy)
                if (underPrefix(kv.first, s.path) && kv.first.size() > s.path.size())
                    sn.lazy.push_back({kv.first.substr(s.path.size()), kv.second});
            sn.any = it->second.any;
        }
        sn.def = defaultEPs(cur_, s.obj, s.path);
        for (auto &e : sn.def) e = canonEP(e);
        std::sort(sn.def.begin(), sn.def.end());
        sn.def.erase(std::unique(sn.def.begin(), sn.def.end()), sn.def.end());
        snaps.push_back(std::move(sn));
    }
    bool strong = dst.size() == 1 && snaps.size() == 1 && !dst[0].weak && strongObj(dst[0].obj) &&
                  !hasStar(dst[0].path);
    for (auto &d : dst) {
        if (isTrusted(d.obj)) continue;
        if (strong) {
            const Snap &sn = snaps[0];
            clearSubtree(d.obj, d.path);
            auto &oc = cur_.mem[d.obj];
            for (auto &kv : sn.vals) vsUnion(oc.vals[normPath(d.path + kv.first)], kv.second);
            for (auto &kv : sn.lazy) oc.lazy[d.path + kv.first] = kv.second;
            if (!sn.def.empty()) oc.lazy[d.path] = sn.def;
            vsUnion(oc.any, sn.any);
            if (oc.vals.empty() && oc.lazy.empty() && oc.any.empty()) cur_.mem.erase(d.obj);
            continue;
        }
        for (auto &sn : snaps) {
            // Paths the destination has that the source doesn't spell out:
            // they may now hold whatever the source holds there.
            std::vector<std::string> dstPaths;
            auto it = cur_.mem.find(d.obj);
            if (it != cur_.mem.end())
                for (auto &kv : it->second.vals)
                    if (underPrefix(kv.first, d.path)) dstPaths.push_back(kv.first);
            for (auto &p : dstPaths) {
                std::string rel = p.substr(d.path.size());
                VSet sv = lookup(cur_, sn.obj, normPath(sn.path + rel));
                writeCell(d.obj, p, sv, false);
            }
            for (auto &kv : sn.vals) writeCell(d.obj, d.path + kv.first, kv.second, false);
            auto addLazy = [&](const std::string &at, const std::vector<std::string> &eps) {
                std::vector<std::string> mine = defaultEPs(cur_, d.obj, at);
                for (auto &e : mine) e = canonEP(e);
                std::sort(mine.begin(), mine.end());
                mine.erase(std::unique(mine.begin(), mine.end()), mine.end());
                sortedUnion(mine, eps);
                if (!mine.empty()) cur_.mem[d.obj].lazy[at] = mine;
            };
            for (auto &kv : sn.lazy) addLazy(d.path + kv.first, kv.second);
            if (!sn.def.empty()) addLazy(d.path, sn.def);
            if (!sn.any.empty()) vsUnion(cur_.mem[d.obj].any, sn.any);
        }
    }
}

void Walker::taint(const std::set<int> &bases, uint8_t bits) {
    if (bases.empty()) return;
    auto hit = [&](int o) { return bases.count(canonBase(o)) != 0; };
    auto fix = [&](VSet &v) { for (auto &e : v) if (hit(e.obj)) e.bits |= bits; };
    for (auto &om : cur_.mem) {
        for (auto &kv : om.second.vals) fix(kv.second);
        fix(om.second.any);
    }
    for (auto &d : cur_.deep) fix(d.second.vals);
    for (int b : bases) cur_.invalid[b] |= bits;
}

// Everything reachable through entry paths under 'prefix' is freed
// (bits) or may hold 'vals'.
void Walker::addDeep(const std::string &prefix, uint8_t bits, const VSet &vals) {
    Deep &d = cur_.deep[prefix];
    d.bits |= bits;
    vsUnion(d.vals, vals);
    if (bits) {
        std::set<int> hitObjs;
        for (size_t i = 0; i < objs_.size(); ++i)
            if (objs_[i].kind == OK::Ext && epUnder(objs_[i].ep, prefix)) hitObjs.insert((int)i);
        taint(hitObjs, bits);
    }
}

// Merge objects (from -> to, every occurrence, cells joined): the
// recency fold ahead of a new allocation, and the per-loop fold below.
void Walker::rename(State &st, const std::map<int, int> &m) {
    if (m.empty()) return;
    auto remap = [&](int o) -> int {
        auto it = m.find(o);
        if (it != m.end()) return it->second;
        if (objs_[o].kind == OK::View) {
            auto ib = m.find(objs_[o].base);
            if (ib != m.end()) return viewObj(ib->second, objs_[o].prefix);
        }
        return o;
    };
    auto fix = [&](VSet &v) {
        bool touched = false;
        for (auto &e : v) if (remap(e.obj) != e.obj) { touched = true; break; }
        if (!touched) return;
        VSet out;
        for (auto &e : v) vsAdd(out, remap(e.obj), e.bits);
        v = std::move(out);
    };
    for (auto &om : st.mem) {
        for (auto &kv : om.second.vals) fix(kv.second);
        fix(om.second.any);
    }
    for (auto &d : st.deep) fix(d.second.vals);
    for (auto &kv : m) {
        int from = kv.first, to = kv.second;
        auto it = st.mem.find(from);
        if (it != st.mem.end()) {
            ObjCells moved = std::move(it->second);
            st.mem.erase(it);
            auto &oc = st.mem[to];
            for (auto &c : moved.vals) vsUnion(oc.vals[c.first], c.second);
            for (auto &c : moved.lazy) sortedUnion(oc.lazy[c.first], c.second);
            vsUnion(oc.any, moved.any);
        }
        auto iv = st.invalid.find(from);
        if (iv != st.invalid.end()) { st.invalid[to] |= iv->second; st.invalid.erase(iv); }
        auto af = st.after.find(from);
        if (af != st.after.end()) {
            std::vector<int> marks = std::move(af->second);
            st.after.erase(af);
            sortedUnion(st.after[to], marks);
        }
    }
}

// Starting another iteration of 'loop': the allocations made in the
// iteration just finished, at sites whose outermost enclosing loop it is,
// join the "earlier iterations" object of their site. A free inside one
// iteration then can't reach a buffer an earlier iteration handed on.
void Walker::foldIteration(State &st, const Stmt *loop) {
    std::map<int, int> m;
    for (size_t i = 0; i < objs_.size(); ++i) {
        const ObjInfo &oi = objs_[i];
        if (oi.kind == OK::Alloc && oi.loop == loop && oi.tier < 2)
            m[(int)i] = allocObj(oi.site, 2, oi.heap, oi.region, loop);
    }
    rename(st, m);
}

void Walker::freshAlloc(int mr, int old, const std::string &region) {
    rename(cur_, {{mr, old}});
    cur_.invalid.erase(mr);
    if (!region.empty()) {
        auto m = cur_.marks.find(region);
        if (m != cur_.marks.end() && !m->second.empty()) cur_.after[mr] = m->second;
    }
}

void Walker::freeObjs(const VSet &v) {
    std::set<int> bases;
    for (auto &e : v) {
        int b = canonBase(e.obj);
        if (cur_.lent.count(b)) lendExit(b, freeLoc_);
        const ObjInfo &oi = objs_[b];
        if (oi.kind == OK::Local || oi.kind == OK::Global || oi.kind == OK::Mark ||
            oi.kind == OK::Temp || oi.kind == OK::Ret || oi.kind == OK::Trusted || oi.kind == OK::Token)
            continue;
        bases.insert(b);
        if (oi.kind == OK::Ext && epRooted(oi.ep)) summary.frees.insert(oi.ep);
    }
    taint(bases, kFreed);
    for (int b : bases)
        if (objs_[b].kind == OK::Ext && objs_[b].collapsed)
            addDeep(objs_[b].ep.substr(0, objs_[b].ep.size() - 1) + "*", kFreed, {});
}

void Walker::resetRegion(const std::string &region, bool fromUnknownMark) {
    if (!fromUnknownMark) summary.resets.insert(region);
    std::set<int> bases;
    for (size_t i = 0; i < objs_.size(); ++i)
        if (objs_[i].region == region && objs_[i].kind != OK::View && objs_[i].kind != OK::Mark)
            bases.insert((int)i);
    cur_.extReset.insert(region);
    taint(bases, kReset);
}

void Walker::freeTo(const std::string &region, const VSet &v) {
    std::vector<int> ms;
    bool unknown = v.empty(), paramMark = false;
    for (auto &e : v) {
        const ObjInfo &oi = objs_[canonBase(e.obj)];
        if (oi.kind == OK::Mark && oi.region == region) ms.push_back(canonBase(e.obj));
        else if (oi.kind == OK::Ext && epRooted(oi.ep)) {
            summary.freeTos.insert({region, oi.ep});
            paramMark = true;
        } else unknown = true;
    }
    if (unknown) { resetRegion(region, false); return; }
    if (paramMark) resetRegion(region, true);
    if (ms.empty()) return;
    std::sort(ms.begin(), ms.end());
    std::set<int> bases;
    for (auto &kv : cur_.after) {
        if (objs_[kv.first].region != region) continue;
        for (int m : kv.second)
            if (std::binary_search(ms.begin(), ms.end(), m)) { bases.insert(kv.first); break; }
    }
    taint(bases, kReset);
}

State Walker::join(const State &a, const State &b) {
    if (!a.live) return b;
    if (!b.live) return a;
    if (a == b) return a;
    State r;
    r.live = true;
    std::set<int> keys;
    for (auto &kv : a.mem) keys.insert(kv.first);
    for (auto &kv : b.mem) keys.insert(kv.first);
    static const ObjCells empty;
    for (int o : keys) {
        auto ia = a.mem.find(o), ib = b.mem.find(o);
        const ObjCells &ca = ia == a.mem.end() ? empty : ia->second;
        const ObjCells &cb = ib == b.mem.end() ? empty : ib->second;
        if (ia != a.mem.end() && ib != b.mem.end() && ca == cb) { r.mem[o] = ca; continue; }
        ObjCells out;
        std::set<std::string> paths;
        for (auto &kv : ca.vals) paths.insert(kv.first);
        for (auto &kv : cb.vals) paths.insert(kv.first);
        for (auto &p : paths) {
            auto va = ca.vals.find(p), vb = cb.vals.find(p);
            VSet v = va != ca.vals.end() ? va->second : lookup(a, o, p);
            vsUnion(v, vb != cb.vals.end() ? vb->second : lookup(b, o, p));
            out.vals[p] = std::move(v);
        }
        std::set<std::string> prefixes;
        for (auto &kv : ca.lazy) prefixes.insert(kv.first);
        for (auto &kv : cb.lazy) prefixes.insert(kv.first);
        for (auto &p : prefixes) {
            auto la = ca.lazy.find(p), lb = cb.lazy.find(p);
            std::vector<std::string> l = la != ca.lazy.end() ? la->second : defaultEPs(a, o, p);
            std::vector<std::string> l2 = lb != cb.lazy.end() ? lb->second : defaultEPs(b, o, p);
            for (auto *x : {&l, &l2}) {
                for (auto &e : *x) e = canonEP(e);
                std::sort(x->begin(), x->end());
                x->erase(std::unique(x->begin(), x->end()), x->end());
            }
            sortedUnion(l, l2);
            if (!l.empty()) out.lazy[p] = std::move(l);
        }
        out.any = ca.any;
        vsUnion(out.any, cb.any);
        r.mem[o] = std::move(out);
    }
    r.invalid = a.invalid;
    for (auto &kv : b.invalid) r.invalid[kv.first] |= kv.second;
    r.extReset = a.extReset;
    r.extReset.insert(b.extReset.begin(), b.extReset.end());
    r.deep = a.deep;
    for (auto &kv : b.deep) {
        Deep &d = r.deep[kv.first];
        d.bits |= kv.second.bits;
        vsUnion(d.vals, kv.second.vals);
    }
    r.after = a.after;
    for (auto &kv : b.after) sortedUnion(r.after[kv.first], kv.second);
    r.marks = a.marks;
    for (auto &kv : b.marks) sortedUnion(r.marks[kv.first], kv.second);
    r.own = a.own;
    for (auto &kv : b.own) r.own[kv.first] |= kv.second;
    // A field split off on one side only is still in place on the other.
    for (auto &kv : r.own)
        if ((kv.second & kOSub) && (!a.own.count(kv.first) || !b.own.count(kv.first))) kv.second |= kOLive;
    r.tok = a.tok;
    for (auto &kv : b.tok) sortedUnion(r.tok[kv.first], kv.second);
    r.lent = a.lent;
    for (auto &kv : b.lent) sortedUnion(r.lent[kv.first], kv.second);
    return r;
}

// ── Expressions ────────────────────────────────────────────────────────────────
VSet Walker::load(const std::vector<CellRef> &cells, const TypePtr &ty, const Expr *site) {
    if (!isTracked(ty)) return {};
    if (site && !cells.empty() &&
        std::all_of(cells.begin(), cells.end(), [&](const CellRef &c) { return isTrusted(c.obj); }))
        return opaqueResult(*site, ty).v;
    const ReferenceType *rt = refOf(ty);
    bool saved = recordUses_;
    recordUses_ = rt != nullptr;
    VSet out;
    for (auto &c : cells) vsUnion(out, lookup(cur_, c.obj, c.path, rt));
    recordUses_ = saved;
    return out;
}

// form: 0 a named variable, 1 a field/element, 2 through a reference
void Walker::checkLoad(const Expr &e, const VSet &v, int form, const std::string &name) {
    if (quiet_ || unsafe_ || !cur_.live) return;
    const ReferenceType *rt = refOf(e.type);
    if (!rt) return;
    uint8_t b = vsBits(v);
    if (!b) return;
    bool heapCause = (b & kFreed) != 0;
    std::string region;
    if (!heapCause) {
        if (rt->region == Region::Arena) region = rt->arenaName;
        for (auto &x : v)
            if (region.empty() && (x.bits & kReset)) region = objs_[canonBase(x.obj)].region;
    }
    std::string after = heapCause
        ? "std::dealloc() freed it"
        : "arena_reset<" + region + ">(), arena_destroy<" + region + ">(), or arena_free_to<" +
              region + ">() invalidated it";
    std::string what;
    if (form == 0) {
        if (heapCause && rt->region == Region::Heap) what = "'" + name + "' (&heap reference)";
        else if (!heapCause && rt->region == Region::Arena) what = "'" + name + "' (&arena<" + region + "> reference)";
        else what = "'" + name + "' (a reference into " + (heapCause ? std::string("a &heap allocation") : "arena<" + region + ">") + ")";
        report(e.loc, "use of " + what + " after " + after);
        return;
    }
    std::string ty = heapCause ? "&heap T" : "&arena<" + region + "> T";
    if ((heapCause && rt->region != Region::Heap) || (!heapCause && rt->region != Region::Arena))
        ty = "&T";
    report(e.loc, "use of a '" + ty + "' value read through " +
                  (form == 1 ? "a struct field/array element" : "a reference") + " after " + after);
}

std::vector<CellRef> Walker::lval(Expr &e) {
    std::vector<CellRef> out;
    if (!cur_.live) return out;
    switch (e.kind) {
    case ExprKind::Ident: {
        auto &id = static_cast<IdentExpr &>(e);
        if (id.resolved) { out.push_back({varObj(id.resolved), ""}); return out; }
        if (!id.resolvedFn)
            for (auto it = binds_.rbegin(); it != binds_.rend(); ++it)
                if (it->first == id.name) { out.push_back({it->second, ""}); return out; }
        return out;
    }
    case ExprKind::Member: case ExprKind::Arrow: {
        auto &me = static_cast<MemberExpr &>(e);
        TypePtr bt = stripNewtype(me.base->type);
        if (bt && bt->kind == TypeKind::Slice) { rval(*me.base); return out; }
        std::string sel;
        if (bt && bt->kind == TypeKind::Tuple) sel = "." + me.field;
        else sel = fieldSel(bt, me.field);
        bool indirect = me.isArrow || (bt && (bt->kind == TypeKind::Reference || bt->kind == TypeKind::Pointer));
        if (indirect && isRawPtr(bt)) {
            rval(*me.base);
            out.push_back({trustedObj(), ""});
            return out;
        }
        if (indirect) {
            EV b = rval(*me.base);
            for (auto &x : b.v) out.push_back(cellOf(x, sel));
        } else {
            for (auto &c : lval(*me.base)) {
                c.path = normPath(c.path + sel);
                out.push_back(c);
            }
        }
        return out;
    }
    case ExprKind::Subscript: {
        auto &se = static_cast<SubscriptExpr &>(e);
        TypePtr bt = stripNewtype(se.base->type);
        std::string sel = "[*]";
        if (se.index && se.index->kind == ExprKind::IntLit && static_cast<IntLitExpr &>(*se.index).value >= 0)
            sel = "[" + std::to_string(static_cast<IntLitExpr &>(*se.index).value) + "]";
        if (bt && bt->kind == TypeKind::Array) {
            auto cells = lval(*se.base);
            if (se.index) rval(*se.index);
            for (auto &c : cells) { c.path = normPath(c.path + sel); out.push_back(c); }
            return out;
        }
        if (bt && bt->kind == TypeKind::Vector) {
            auto cells = lval(*se.base);
            if (se.index) rval(*se.index);
            return cells;
        }
        EV b = rval(*se.base);
        if (se.index) rval(*se.index);
        bool refToArray = bt && bt->kind == TypeKind::Reference &&
                          stripNewtype(static_cast<ReferenceType &>(*bt).base) &&
                          stripNewtype(static_cast<ReferenceType &>(*bt).base)->kind == TypeKind::Array;
        if (bt && bt->kind == TypeKind::Slice) {
            for (auto &x : b.v) {
                CellRef c = cellOf(x, "");
                c.weak = true;
                out.push_back(c);
            }
            return out;
        }
        if (!refToArray) {
            // Raw pointer or unsafe reference indexing: trusted memory.
            out.push_back({trustedObj(), ""});
            return out;
        }
        for (auto &x : b.v) out.push_back(cellOf(x, sel));
        return out;
    }
    case ExprKind::Deref: {
        auto &op = *static_cast<UnaryExpr &>(e).operand;
        EV b = rval(op);
        if (isRawPtr(op.type)) { out.push_back({trustedObj(), ""}); return out; }
        for (auto &x : b.v) out.push_back(cellOf(x, ""));
        return out;
    }
    case ExprKind::Unary: {
        auto &ue = static_cast<UnaryExpr &>(e);
        if (ue.op == UnaryOp::ForceUnwrap) {
            TypePtr ot = stripNewtype(ue.operand->type);
            if (ot && ot->kind == TypeKind::Optional && !isRawPtr(ot)) return lval(*ue.operand);
            EV b = rval(*ue.operand);
            if (isRawPtr(ot)) { out.push_back({trustedObj(), ""}); return out; }
            for (auto &x : b.v) out.push_back(cellOf(x, ""));
            return out;
        }
        break;
    }
    case ExprKind::Cast:
        return lval(*static_cast<CastExpr &>(e).operand);
    default: break;
    }
    // An rvalue used where a location is needed (a call result's field, a
    // compound literal): materialize it.
    EV ev = rval(e);
    if (ev.agg) return ev.cells;
    int t = tempObj(&e, 7);
    clearSubtree(t, "");
    if (!ev.v.empty()) cur_.mem[t].vals[""] = ev.v;
    out.push_back({t, ""});
    return out;
}

EV Walker::opaqueResult(const Expr &site, const TypePtr &ty) {
    EV r;
    if (isAgg(ty)) {
        int t = tempObj(&site, 3);
        clearSubtree(t, "");
        r.agg = true;
        r.cells.push_back({t, ""});
        return r;
    }
    if (!isTracked(ty)) return r;
    const ReferenceType *rt = refOf(ty);
    TypePtr st = stripNewtype(ty);
    if (st && st->kind == TypeKind::UInt64) return r;
    bool heap = rt && rt->region == Region::Heap;
    std::string region = rt && rt->region == Region::Arena ? rt->arenaName : "";
    std::string site_ = "x" + ptrKey(&site);
    int mr = siteObj(site, site_, 0, heap, region), old = siteObj(site, site_, 1, heap, region);
    freshAlloc(mr, old, region);
    vsAdd(r.v, mr, 0);
    return r;
}

void Walker::writeInit(const std::vector<CellRef> &dst, Expr &init, const TypePtr &ty) {
    if (init.kind == ExprKind::Compound) {
        auto &ci = static_cast<CompoundInitExpr &>(init);
        TypePtr t = stripNewtype(ci.type ? ci.type : ty);
        if (dst.size() == 1 && !dst[0].weak) clearSubtree(dst[0].obj, dst[0].path);
        for (size_t i = 0; i < ci.inits.size(); ++i) {
            if (!ci.inits[i]) continue;
            int64_t slot = i < ci.resolvedSlots.size() ? ci.resolvedSlots[i] : (int64_t)i;
            std::string sel;
            TypePtr et;
            if (t && t->kind == TypeKind::Struct) {
                auto &st = static_cast<StructType &>(*t);
                if (slot < 0 || slot >= (int64_t)st.fields.size()) { rval(*ci.inits[i]); continue; }
                sel = isUnionType(&st) ? ".|" : "." + st.fields[slot].name;
                et = st.fields[slot].type;
            } else if (t && t->kind == TypeKind::Array) {
                sel = "[" + std::to_string(slot) + "]";
                et = static_cast<ArrayType &>(*t).element;
            } else { rval(*ci.inits[i]); continue; }
            std::vector<CellRef> sub = dst;
            for (auto &c : sub) c.path = normPath(c.path + sel);
            writeInit(sub, *ci.inits[i], et);
        }
        return;
    }
    EV v = rval(init);
    if (mustDrop(init.type)) ownMove(init, v, kOMove);
    if (isAgg(ty) || v.agg) { if (v.agg) copyAgg(dst, v.cells); return; }
    if (isTracked(ty) || !v.v.empty()) write(dst, v.v);
}

EV Walker::rval(Expr &e) {
    EV r;
    if (!cur_.live) return r;
    switch (e.kind) {
    case ExprKind::Ident: {
        auto &id = static_cast<IdentExpr &>(e);
        if (!id.resolved && id.resolvedFn) return r;
        auto cells = lval(e);
        if (cells.empty()) return r;
        ownUse(cells, e.loc, false);
        if (isAgg(e.type)) { r.agg = true; r.cells = cells; return r; }
        r.v = load(cells, e.type);
        checkLoad(e, r.v, 0, id.name);
        ownCheckVal(e, r.v);
        return r;
    }
    case ExprKind::Member: case ExprKind::Arrow: case ExprKind::Subscript: case ExprKind::Deref: {
        if (e.kind == ExprKind::Member || e.kind == ExprKind::Arrow) {
            auto &me = static_cast<MemberExpr &>(e);
            TypePtr bt = stripNewtype(me.base->type);
            if (bt && bt->kind == TypeKind::Slice) {
                EV b = rval(*me.base);
                if (me.field == "ptr") r.v = b.v;
                return r;
            }
        }
        auto cells = lval(e);
        ownUse(cells, e.loc, false);
        if (isAgg(e.type)) { r.agg = true; r.cells = cells; return r; }
        r.v = load(cells, e.type, &e);
        checkLoad(e, r.v, e.kind == ExprKind::Deref ? 2 : 1, "");
        ownCheckVal(e, r.v);
        return r;
    }
    case ExprKind::AddrOf: {
        auto cells = lval(*static_cast<UnaryExpr &>(e).operand);
        ownUse(cells, e.loc, false);
        if (!cells.empty() &&
            std::all_of(cells.begin(), cells.end(), [&](const CellRef &c) { return isTrusted(c.obj); }))
            return opaqueResult(e, e.type);
        for (auto &c : cells) vsAdd(r.v, viewObj(c.obj, c.path), c.bits);
        return r;
    }
    case ExprKind::Unary: {
        auto &ue = static_cast<UnaryExpr &>(e);
        switch (ue.op) {
        case UnaryOp::SizeofExpr: return r;
        case UnaryOp::ForceUnwrap: {
            EV b = rval(*ue.operand);
            return b;
        }
        case UnaryOp::PreInc: case UnaryOp::PreDec: case UnaryOp::PostInc: case UnaryOp::PostDec: {
            EV b = rval(*ue.operand);
            r.v = b.v;
            return r;
        }
        default:
            rval(*ue.operand);
            return r;
        }
    }
    case ExprKind::Binary: {
        auto &be = static_cast<BinaryExpr &>(e);
        if (be.resolvedOperator) {
            // 'a OP b' on a struct: operator method with self = &a.
            FunctionDecl *callee = drv_.bodyOf(be.resolvedOperator);
            std::vector<EV> args(2);
            auto lc = lval(*be.left);
            for (auto &c : lc) vsAdd(args[0].v, viewObj(c.obj, c.path), 0);
            args[1] = rval(*be.right);
            const Summary *s = callee ? drv_.summaryOf(callee) : nullptr;
            if (!s) return opaqueResult(e, e.type);
            return applySummary(e, callee, *s, args);
        }
        if (be.op == BinaryOp::LogAnd || be.op == BinaryOp::LogOr) {
            State t, f;
            cond(&e, t, f);
            cur_ = join(t, f);
            return r;
        }
        if (be.op == BinaryOp::Comma) { rval(*be.left); return rval(*be.right); }
        EV a = rval(*be.left);
        EV b = rval(*be.right);
        if (be.op == BinaryOp::Add || be.op == BinaryOp::Sub || be.op == BinaryOp::WrapAdd ||
            be.op == BinaryOp::WrapSub || be.op == BinaryOp::BitAnd || be.op == BinaryOp::BitOr) {
            if (isTracked(e.type)) { r.v = a.v; vsUnion(r.v, b.v); }
        }
        return r;
    }
    case ExprKind::Ternary: {
        auto &te = static_cast<TernaryExpr &>(e);
        State ts, fs;
        cond(te.cond.get(), ts, fs);
        cur_ = std::move(ts);
        EV a = rval(*te.then);
        if (mustDrop(e.type)) ownMove(*te.then, a, kOMove);
        State sa = cur_;
        cur_ = std::move(fs);
        EV b = rval(*te.else_);
        if (mustDrop(e.type)) ownMove(*te.else_, b, kOMove);
        cur_ = join(sa, cur_);
        ownTemp(e);
        r.v = a.v;
        vsUnion(r.v, b.v);
        if (a.agg || b.agg) {
            r.agg = true;
            r.cells = a.cells;
            for (auto &c : b.cells) { CellRef x = c; x.weak = true; r.cells.push_back(x); }
            for (auto &c : r.cells) c.weak = r.cells.size() > 1 || c.weak;
        }
        return r;
    }
    case ExprKind::Call: return evalCall(static_cast<CallExpr &>(e));
    case ExprKind::Cast: return rval(*static_cast<CastExpr &>(e).operand);
    case ExprKind::Assign: {
        auto &ae = static_cast<AssignExpr &>(e);
        TypePtr lt = ae.lhs->type;
        if (ae.op == AssignOp::Assign) {
            bool owned = mustDrop(lt);
            if (ae.rhs->kind == ExprKind::Compound) {
                auto dst = lval(*ae.lhs);
                writeInit(dst, *ae.rhs, lt);
                if (owned) ownAssign(dst, e.loc); else ownUse(dst, e.loc, true);
                return r;
            }
            EV v = rval(*ae.rhs);
            if (owned) ownMove(*ae.rhs, v, kOMove);
            auto dst = lval(*ae.lhs);
            {   // 'h = spawn_scoped(...)': h is that thread's handle
                const Expr *re = ae.rhs.get();
                while (re && re->kind == ExprKind::Cast) re = static_cast<const CastExpr *>(re)->operand.get();
                auto st = spawnTok_.find(re);
                if (st != spawnTok_.end() && dst.size() == 1 && objs_[dst[0].obj].kind == OK::Local)
                    handleTok_[dst[0].obj] = st->second;
            }
            if (owned) ownAssign(dst, e.loc); else ownUse(dst, e.loc, true);
            if (v.agg) { copyAgg(dst, v.cells); return v; }
            if (isTracked(lt) || !v.v.empty()) write(dst, v.v);
            return v;
        }
        EV v = rval(*ae.rhs);
        auto dst = lval(*ae.lhs);
        if (isTracked(lt)) {
            VSet old = load(dst, lt);
            checkLoad(*ae.lhs, old, ae.lhs->kind == ExprKind::Ident ? 0 : 1,
                      ae.lhs->kind == ExprKind::Ident ? static_cast<IdentExpr &>(*ae.lhs).name : "");
            vsUnion(old, v.v);
            write(dst, old);
            r.v = old;
        }
        return r;
    }
    case ExprKind::Compound: {
        int t = tempObj(&e, 1);
        clearSubtree(t, "");
        writeInit({{t, ""}}, e, e.type);
        r.agg = true;
        r.cells.push_back({t, ""});
        ownTemp(e);
        return r;
    }
    case ExprKind::TupleLit: {
        auto &tl = static_cast<TupleLitExpr &>(e);
        int t = tempObj(&e, 1);
        clearSubtree(t, "");
        for (size_t i = 0; i < tl.elements.size(); ++i) {
            if (!tl.elements[i]) continue;
            writeInit({{t, "." + std::to_string(i)}}, *tl.elements[i], tl.elements[i]->type);
        }
        r.agg = true;
        r.cells.push_back({t, ""});
        ownTemp(e);
        return r;
    }
    case ExprKind::TaggedUnionInit: {
        auto &tu = static_cast<TaggedUnionInitExpr &>(e);
        int t = tempObj(&e, 1);
        clearSubtree(t, "");
        if (tu.payload) writeInit({{t, ".|"}}, *tu.payload, tu.payload->type);
        r.agg = true;
        r.cells.push_back({t, ""});
        ownTemp(e);
        return r;
    }
    case ExprKind::New: {
        auto &ne = static_cast<NewExpr &>(e);
        if (ne.arraySize) rval(*ne.arraySize);
        bool heap = ne.regionName.empty();
        std::string site = "n" + ptrKey(&e);
        int mr = siteObj(e, site, 0, heap, ne.regionName), old = siteObj(e, site, 1, heap, ne.regionName);
        freshAlloc(mr, old, ne.regionName);
        clearSubtree(mr, "");
        vsAdd(r.v, mr, 0);
        return r;
    }
    case ExprKind::Spawn: {
        auto &sp = static_cast<SpawnExpr &>(e);
        if (sp.fnExpr) rval(*sp.fnExpr);
        if (sp.argExpr) rval(*sp.argExpr);
        return r;
    }
    case ExprKind::Try: {
        auto &te = static_cast<TryExpr &>(e);
        EV v = rval(*te.inner);
        if (mustDrop(te.inner->type)) ownMove(*te.inner, v, kOMove);
        ownFlush();
        State ok = cur_;
        clearSubtree(retObj_, "");
        SourceLocation savedEnd = endLoc_;
        endLoc_ = e.loc;
        doReturn(true);
        endLoc_ = savedEnd;
        cur_ = std::move(ok);
        ownTemp(e);
        return v;
    }
    case ExprKind::Slice: {
        auto &se = static_cast<SliceExpr &>(e);
        TypePtr bt = stripNewtype(se.base->type);
        if (bt && bt->kind == TypeKind::Array) {
            for (auto &c : lval(*se.base)) vsAdd(r.v, viewObj(c.obj, normPath(c.path + "[*]")), 0);
        } else {
            r.v = rval(*se.base).v;
        }
        if (se.start) rval(*se.start);
        if (se.end) rval(*se.end);
        return r;
    }
    case ExprKind::GenericSelection: {
        auto &gs = static_cast<GenericSelectionExpr &>(e);
        if (gs.selectedIndex >= 0 && gs.selectedIndex < (int)gs.associations.size() &&
            gs.associations[gs.selectedIndex].expr)
            return rval(*gs.associations[gs.selectedIndex].expr);
        return r;
    }
    case ExprKind::Match: {
        auto &me = static_cast<MatchExpr &>(e);
        EV subj = rval(*me.subject);
        std::vector<const std::vector<MatchPattern> *> armPats;
        for (auto &arm : me.arms) armPats.push_back(&arm.patterns);
        bool moving = ownMatchSubject(me.subject.get(), subj, armPats);
        State base = cur_;
        State acc;
        for (auto &arm : me.arms) {
            cur_ = base;
            size_t nb = binds_.size();
            frames_.push_back({nullptr, {}});
            for (auto &p : arm.patterns) {
                bindPattern(p, subj, me.subject->type);
                if (moving) ownBind(p, me.subject->loc);
            }
            if (arm.value) {
                EV v = rval(*arm.value);
                if (mustDrop(e.type)) ownMove(*arm.value, v, kOMove);
                vsUnion(r.v, v.v);
                if (v.agg) {
                    r.agg = true;
                    for (auto &c : v.cells) r.cells.push_back(c);
                }
            }
            ownExitFrame(frames_.back(), arm.value ? arm.value->loc : e.loc, "at the end of its match arm");
            frames_.pop_back();
            binds_.resize(nb);
            acc = join(acc, cur_);
        }
        cur_ = me.arms.empty() ? base : acc;
        if (r.cells.size() > 1) for (auto &c : r.cells) c.weak = true;
        ownTemp(e);
        return r;
    }
    default:
        return r;
    }
}

// The states in which condition 'c' is true and false, split through
// '&&', '||', '!', ',' and constants, so a free in one operand isn't joined
// into the outcome where that operand never ran.
void Walker::cond(Expr *c, State &t, State &f) {
    if (!c || !cur_.live) { t = cur_; f = c ? cur_ : State{}; return; }
    if (c->kind == ExprKind::IntLit || c->kind == ExprKind::BoolLit) {
        bool v = c->kind == ExprKind::IntLit ? static_cast<IntLitExpr *>(c)->value != 0
                                             : static_cast<BoolLitExpr *>(c)->value;
        t = v ? cur_ : State{};
        f = v ? State{} : cur_;
        return;
    }
    if (c->kind == ExprKind::Unary && static_cast<UnaryExpr *>(c)->op == UnaryOp::Not) {
        cond(static_cast<UnaryExpr *>(c)->operand.get(), f, t);
        return;
    }
    if (c->kind == ExprKind::Binary) {
        auto *be = static_cast<BinaryExpr *>(c);
        if (!be->resolvedOperator && (be->op == BinaryOp::LogAnd || be->op == BinaryOp::LogOr)) {
            State lt, lf;
            cond(be->left.get(), lt, lf);
            cur_ = be->op == BinaryOp::LogAnd ? std::move(lt) : std::move(lf);
            State rt, rf;
            cond(be->right.get(), rt, rf);
            if (be->op == BinaryOp::LogAnd) { t = std::move(rt); f = join(lf, rf); }
            else { f = std::move(rf); t = join(lt, rt); }
            return;
        }
        if (!be->resolvedOperator && be->op == BinaryOp::Comma) {
            rval(*be->left);
            cond(be->right.get(), t, f);
            return;
        }
    }
    rval(*c);
    t = cur_;
    f = cur_;
}

void Walker::bindPattern(const MatchPattern &pat, const EV &subject, const TypePtr &subjTy) {
    if (pat.bindName.empty()) return;
    int b = tempObj(&pat, 2);
    clearSubtree(b, "");
    const StructType *st = nullptr;
    TypePtr t = stripNewtype(subjTy);
    if (t && t->kind == TypeKind::Struct) st = static_cast<const StructType *>(t.get());
    if (st && st->isTaggedUnion) {
        std::vector<CellRef> src = subject.cells;
        for (auto &c : src) c.path = normPath(c.path + ".|");
        if (isAgg(pat.payloadType)) copyAgg({{b, ""}}, src);
        else {
            VSet v = load(src, pat.payloadType);
            if (!v.empty()) cur_.mem[b].vals[""] = v;
        }
    } else if (subject.agg) {
        copyAgg({{b, ""}}, subject.cells);
    } else if (!subject.v.empty()) {
        cur_.mem[b].vals[""] = subject.v;
    }
    binds_.push_back({pat.bindName, b});
}

static bool isPlaceExpr(const Expr *e) {
    while (e && e->kind == ExprKind::Cast) e = static_cast<const CastExpr *>(e)->operand.get();
    if (!e) return false;
    switch (e->kind) {
    case ExprKind::Ident: case ExprKind::Member: case ExprKind::Arrow: case ExprKind::Subscript:
    case ExprKind::Deref:
        return true;
    case ExprKind::Unary: return static_cast<const UnaryExpr *>(e)->op == UnaryOp::ForceUnwrap;
    default: return false;
    }
}

// ── Calls ──────────────────────────────────────────────────────────────────────
// Does a value of this type hold a reference (so it can borrow)? A Drop
// type's '&heap' fields are buffers it owns, not borrows.
static bool holdsRefs(const Sema &sema, const TypePtr &t0, int depth = 0) {
    TypePtr t = stripNewtype(t0);
    if (!t || depth > 4) return false;
    switch (t->kind) {
    case TypeKind::Reference: {
        auto &rt = static_cast<ReferenceType &>(*t);
        return !(rt.base && rt.base->kind == TypeKind::Function) && rt.region != Region::Static;
    }
    case TypeKind::Slice: return true;
    case TypeKind::Optional: return holdsRefs(sema, static_cast<OptionalType &>(*t).inner, depth + 1);
    case TypeKind::Struct: {
        auto &st = static_cast<StructType &>(*t);
        bool owner = sema.hasDropMethod(st);
        for (auto &f : st.fields) {
            const ReferenceType *rt = refOf(f.type);
            if (owner && rt && rt->region == Region::Heap) continue;
            if (holdsRefs(sema, f.type, depth + 1)) return true;
        }
        return false;
    }
    default: return false;
    }
}

static bool isNoReturnCallee(const FunctionDecl *fd) {
    if (!fd) return false;
    if (fd->isNoReturn) return true;
    static const char *names[] = {"std_panic_at", "panic_at", "abort", "exit", "_exit", "_Exit",
                                  "__builtin_trap", "__builtin_unreachable", "std_panic", "quick_exit", nullptr};
    for (const char **n = names; *n; ++n) if (fd->name == *n) return true;
    return false;
}

// See evalCall: does a method's reference result default to borrowing
// from its receiver (p0)?
static bool elidesToSelf(const FunctionDecl &fd, const Summary &s) {
    TypePtr rt = fd.returnType;
    while (rt && rt->kind == TypeKind::Optional) rt = static_cast<OptionalType &>(*rt).inner;
    if (!rt || rt->kind != TypeKind::Reference) return false;
    auto &ref = static_cast<ReferenceType &>(*rt);
    if (ref.region != Region::Extern && ref.region != Region::Unknown) return false;
    if (ref.base && ref.base->kind == TypeKind::Function) return false;
    for (size_t i = 1; i < fd.params.size(); ++i) {
        TypePtr pt = fd.params[i].type;
        while (pt && pt->kind == TypeKind::Optional) pt = static_cast<OptionalType &>(*pt).inner;
        if (pt && (pt->kind == TypeKind::Reference || pt->kind == TypeKind::Pointer || pt->kind == TypeKind::Slice))
            return false;
    }
    bool any = false;
    auto fresh = [&](const std::vector<Term> &ts) {
        for (auto &t : ts) { if (t.kind != 'F') return false; any = true; }
        return true;
    };
    for (auto &kv : s.ret.vals) if (!fresh(kv.second)) return false;
    if (!fresh(s.ret.any)) return false;
    if (!s.ret.lazy.empty()) return false;
    return any;
}


// How a drop reaches the place 'e': 0 owned here (a local, a by-value
// parameter, a global, a field/element of one, or what a reference
// variable itself names: a handle like File::open's result, or a
// reference parameter, which the ownership walk checks), 1 through an
// owning '&heap' reference or a raw pointer (the heap analysis tracks
// those), 2 borrowed: a field or element reached through any other
// reference or a slice, or a reference a call returned (a container's
// element), which the owner drops again. selfRoot: the place is under
// 'self'.
static int dropReach(const Expr *e, bool &selfRoot) {
    auto viaType = [](const TypePtr &t0) -> int {
        TypePtr t = stripNewtype(t0);
        while (t && t->kind == TypeKind::Optional) t = stripNewtype(static_cast<OptionalType &>(*t).inner);
        if (!t) return 0;
        if (t->kind == TypeKind::Pointer) return 1;
        if (t->kind == TypeKind::Reference) return static_cast<ReferenceType &>(*t).region == Region::Heap ? 1 : 2;
        if (t->kind == TypeKind::Slice) return 2;
        return -1;   // a value: keep walking
    };
    for (int guard = 0; e && guard < 64; ++guard) {
        switch (e->kind) {
        case ExprKind::Cast: e = static_cast<const CastExpr *>(e)->operand.get(); continue;
        case ExprKind::Ident: {
            auto &id = static_cast<const IdentExpr &>(*e);
            if (id.name == "self") selfRoot = true;
            return 0;
        }
        case ExprKind::Member: case ExprKind::Arrow: {
            const Expr *b = static_cast<const MemberExpr *>(e)->base.get();
            int r = b ? viaType(b->type) : 0;
            if (r >= 0) {
                if (b && b->kind == ExprKind::Ident && static_cast<const IdentExpr *>(b)->name == "self") selfRoot = true;
                return r;
            }
            e = b;
            continue;
        }
        case ExprKind::Subscript: {
            const Expr *b = static_cast<const SubscriptExpr *>(e)->base.get();
            int r = b ? viaType(b->type) : 0;
            if (r >= 0) return r;
            e = b;
            continue;
        }
        case ExprKind::Deref: {
            const Expr *b = static_cast<const UnaryExpr *>(e)->operand.get();
            if (b && b->kind == ExprKind::Call) return viaType(b->type) == 2 ? 2 : 0;
            return 0;
        }
        case ExprKind::Unary: return 0;
        case ExprKind::Call: return viaType(e->type) == 2 ? 2 : 0;   // a value it returned is a temporary
        default: return 0;
        }
    }
    return 0;
}

// A drop (x.drop(), a 'consume' method, drop(x)) of a place this function
// only borrows: the value's owner (a container, the caller) drops it
// again. Inside a destructor, 'self' and its fields are owned.
void Walker::refDropCheck(const Expr *place, SourceLocation loc) {
    refDropReported_ = false;
    if (quiet_ || unsafe_ || !cur_.live || !place) return;
    bool selfRoot = false;
    int r = dropReach(place, selfRoot);
    if (r != 2 || (selfRoot && isDtor_)) return;
    reportN(loc, "cannot drop a value this function only borrows (it is reached through a reference, a slice or a "
                 "call's result, and whoever owns it will drop it again); take it out with std::replace() and drop "
                 "that, or let its owner drop it (a container's clear()/remove())",
            {});
    refDropReported_ = true;
}

EV Walker::evalCall(CallExpr &e) {
    callOwners_.clear();
    EV r = evalCallInner(e);
    if (!cur_.live) return r;
    FunctionDecl *fd = nullptr;
    if (e.callee && e.callee->kind == ExprKind::Ident) fd = static_cast<IdentExpr &>(*e.callee).resolvedFn;
    // A method on an owned receiver returning a reference into it borrows
    // from it: the callee's summary says its result points into memory
    // reached from its 'self' (p0). A callee without a body here gives no
    // such link (its result might be fresh memory).
    bool intoSelf = false;
    if (!callOwners_.empty() && holdsRefs(drv_.sema, e.type)) {
        FunctionDecl *callee = drv_.bodyOf(fd);
        const Summary *s = callee ? drv_.summaryOf(callee) : nullptr;
        if (s) {
            auto rooted = [](const std::vector<Term> &ts) {
                for (auto &t : ts)
                    if (t.kind == 'E' && t.key.size() >= 2 && t.key.compare(0, 2, "p0") == 0 &&
                        (t.key.size() == 2 || std::strchr("*.[~", t.key[2])))
                        return true;
                return false;
            };
            for (auto &kv : s->ret.vals) if (rooted(kv.second)) intoSelf = true;
            if (rooted(s->ret.any)) intoSelf = true;
            for (auto &kv : s->ret.lazy)
                for (auto &ep : kv.second)
                    if (ep.compare(0, 2, "p0") == 0) intoSelf = true;
            // Elision, as in Rust: a method returning a plain (region-less)
            // reference whose origin the summary can't name (it came out
            // of raw memory: a container's buffer) and taking no other
            // reference borrows from its receiver. '&heap'/'&static'
            // results claim their own lifetime and are left alone.
            if (!intoSelf && callee->isMethod && !callee->isStaticMethod && elidesToSelf(*callee, *s))
                intoSelf = true;
        }
    }
    if (intoSelf) {
        int t = ownToken(e, callOwners_);
        if (r.agg) {
            for (auto &c : r.cells)
                if (objs_[c.obj].kind == OK::Temp) vsAdd(cur_.mem[c.obj].any, t, 0);
        } else {
            vsAdd(r.v, t, 0);
        }
    }
    callOwners_.clear();
    ownTemp(e);
    if (isNoReturnCallee(fd)) cur_.live = false;   // panic/abort paths owe nothing
    return r;
}

EV Walker::evalCallInner(CallExpr &e) {
    EV r;
    if (e.taggedUnionTag >= 0) {
        int t = tempObj(&e, 1);
        clearSubtree(t, "");
        for (size_t i = 0; i < e.args.size(); ++i)
            if (e.args[i]) {
                if (i == 0) writeInit({{t, ".|"}}, *e.args[i], e.args[i]->type);
                else rval(*e.args[i]);
            }
        r.agg = true;
        r.cells.push_back({t, ""});
        return r;
    }
    if (e.nullOp != CallExpr::NullOp::None) {
        EV b = e.nullOpBase ? rval(*e.nullOpBase) : EV{};
        if (e.nullOp == CallExpr::NullOp::Default) {
            EV a = e.args.empty() || !e.args[0] ? EV{} : rval(*e.args[0]);
            if (e.nullOpBase && mustDrop(e.nullOpBase->type)) {
                reportN(e.loc, "'.default()' on a value of Drop type leaks whichever of the two values it "
                               "doesn't return; use 'match' and drop the other one",
                        {});
                ownMove(*e.nullOpBase, b, kOMove);
                if (!e.args.empty() && e.args[0]) ownMove(*e.args[0], a, kOMove);
            }
            r.v = b.v;
            vsUnion(r.v, a.v);
            if (b.agg || a.agg) {
                r.agg = true;
                r.cells = b.cells;
                for (auto &c : a.cells) r.cells.push_back(c);
                for (auto &c : r.cells) c.weak = r.cells.size() > 1;
            }
        } else {
            for (auto &a : e.args) if (a) rval(*a);
        }
        return r;
    }
    FunctionDecl *fd = nullptr;
    if (e.callee && e.callee->kind == ExprKind::Ident) {
        auto &id = static_cast<IdentExpr &>(*e.callee);
        if (id.resolvedFn && !id.resolved) fd = id.resolvedFn;
        else rval(*e.callee);
    } else if (e.callee) {
        rval(*e.callee);
    }
    std::vector<EV> args;
    if (e.methodBase) {
        TypePtr bt = stripNewtype(e.methodBase->type);
        if (fd && fd->isStaticMethod) {
            rval(*e.methodBase);
        } else if (bt && (bt->kind == TypeKind::Reference || bt->kind == TypeKind::Pointer)) {
            args.push_back(rval(*e.methodBase));
            std::vector<CellRef> pointees;
            for (auto &x : args.back().v) pointees.push_back(cellOf(x, ""));
            if (pointees.size() > 1) for (auto &c : pointees) c.weak = true;
            if (drv_.sema.isConsumeMethod(fd)) {
                refDropCheck(e.methodBase.get(), e.loc);
                ownConsumeCells(pointees, kODrop, e.loc, nullptr);
                refDropReported_ = false;
            } else {
                ownUse(pointees, e.loc, false);
            }
        } else {
            EV self;
            auto cells = lval(*e.methodBase);
            for (auto &c : cells) vsAdd(self.v, viewObj(c.obj, c.path), 0);
            args.push_back(self);
            if (drv_.sema.isConsumeMethod(fd)) {
                EV bv;
                bv.agg = true;
                bv.cells = cells;
                refDropCheck(e.methodBase.get(), e.loc);
                if (mustDrop(e.methodBase->type)) ownMove(*e.methodBase, bv, kODrop);
                else ownConsumeCells(cells, kODrop, e.loc, nullptr);
                refDropReported_ = false;
            } else {
                ownUse(cells, e.loc, false);
                if (isPlaceExpr(e.methodBase.get()) && cells.size() == 1 && tracked(cells[0].obj))
                    for (auto it : ownOverlap(cells[0].obj, cells[0].path)) callOwners_.push_back(it->first);
            }
        }
    }
    size_t firstArg = args.size();
    for (auto &a : e.args) {
        args.push_back(a ? rval(*a) : EV{});
        // A value of Drop type passed by value moves into the callee.
        if (a && mustDrop(a->type)) ownMove(*a, args.back(), kOMove);
    }
    if (!cur_.live) return r;

    // spawn_scoped(f, arg): lend arg's referents to the new thread;
    // join(h): get them back.
    if (e.callee && e.callee->kind == ExprKind::Ident && !fd) {
        auto &cn = static_cast<IdentExpr &>(*e.callee).name;
        if (cn == "spawn_scoped" && args.size() >= 2 && e.args.size() >= 2) {
            int tok = intern("S" + ptrKey(&e), objInfo(OK::Token));
            const ReferenceType *rt = refOf(e.args[1]->type);
            bool sync = rt && drv_.sema.threadSafety(rt->base, true, true).empty();
            spawns_[tok] = {e.loc, rt ? rt->mut : true, sync};
            spawnTok_[&e] = tok;
            for (auto &x : args[1].v) {
                int b = canonBase(x.obj);
                const ObjInfo &oi = objs_[b];
                if (oi.kind == OK::Local || oi.kind == OK::Ext || oi.kind == OK::Alloc || oi.kind == OK::Temp)
                    sortedUnion(cur_.lent[b], std::vector<int>{tok});
            }
        } else if ((cn == "chan_send" || cn == "chan_recv") && args.size() >= 2 && e.args.size() >= 2) {
            // chan_send(ch, &v) moves v to the receiver; chan_recv(ch, &out)
            // stores a received value in out.
            const ReferenceType *rt = refOf(e.args[1]->type);
            if (rt && mustDrop(rt->base)) {
                std::vector<CellRef> cells;
                for (auto &x : args[1].v) cells.push_back(cellOf(x, ""));
                if (cells.size() > 1) for (auto &c : cells) c.weak = true;
                if (cn == "chan_send") ownConsumeCells(cells, kOMove, e.loc, e.args[1].get());
                else ownAssign(cells, e.loc);
            }
        } else if (cn == "__safec_join" && !e.args.empty() && e.args[0] &&
                   e.args[0]->kind == ExprKind::Ident && static_cast<IdentExpr &>(*e.args[0]).resolved) {
            auto ht = handleTok_.find(varObj(static_cast<IdentExpr &>(*e.args[0]).resolved));
            if (ht != handleTok_.end())
                for (auto it = cur_.lent.begin(); it != cur_.lent.end();) {
                    auto &v = it->second;
                    v.erase(std::remove(v.begin(), v.end(), ht->second), v.end());
                    it = v.empty() ? cur_.lent.erase(it) : std::next(it);
                }
        }
    }
    freeLoc_ = e.loc;
    switch (e.memOp) {
    case CallExpr::MemOp::Dealloc: {
        if (args.size() > firstArg) {
            const VSet &v = args[firstArg].v;
            const Expr *arg = e.args.empty() ? nullptr : e.args[0].get();
            // A reference argument was already checked when it was read;
            // a raw pointer is checked here.
            if ((vsBits(v) & kFreed) && arg && !refOf(arg->type) && !refOf(
                    arg->kind == ExprKind::Cast ? static_cast<const CastExpr *>(arg)->operand->type : nullptr))
                report(e.loc, "double free: dealloc() of a pointer after std::dealloc() freed it");
            freeObjs(v);
        }
        return r;
    }
    case CallExpr::MemOp::ArenaReset: case CallExpr::MemOp::ArenaDestroy:
        resetRegion(e.memRegion, false);
        return r;
    case CallExpr::MemOp::ArenaMark: {
        int m = markObj("k" + ptrKey(&e), e.memRegion);
        sortedUnion(cur_.marks[e.memRegion], std::vector<int>{m});
        vsAdd(r.v, m, 0);
        return r;
    }
    case CallExpr::MemOp::ArenaFreeTo:
        freeTo(e.memRegion, args.size() > firstArg ? args[firstArg].v : VSet{});
        return r;
    default: break;
    }

    FunctionDecl *callee = drv_.bodyOf(fd);
    const Summary *s = callee ? drv_.summaryOf(callee) : nullptr;
    if (!s) return opaqueResult(e, e.type);
    return applySummary(e, callee, *s, args);
}

// A call's fresh objects are named by the call site plus the callee's own
// name for them, a chain of call sites ending in the allocation site. Within
// a recursive component, and past a length cap, the chain is cut down to its
// allocation site ('rec:<site>', 'deep:<site>') so names stay finite. Keeping
// the site apart, rather than folding every object such a call makes into
// one, keeps a free of one allocation (a map's rehash dropping its old
// arrays) from tainting the unrelated ones the same call made (the buffer of
// a Vec beside the map).
std::string Walker::calleeSite(const Expr &call, FunctionDecl *callee, const std::string &key) {
    auto leaf = [&]() {
        std::string l = key.substr(key.rfind('/') + 1);
        for (const char *pre : {"rec:", "deep:"})
            if (l.compare(0, std::strlen(pre), pre) == 0) l = l.substr(std::strlen(pre));
        return l;
    };
    std::string k = drv_.sameScc(&fn_, callee) ? "rec:" + leaf() : key;
    std::string s = "c" + ptrKey(&call) + "/" + k;
    if (s.size() > 160) s = "c" + ptrKey(&call) + "/deep:" + leaf();
    return s;
}

// The objects reachable from the values stored under 'cells' (everything a
// collapsed '~' entry path stands for), and the entry-path prefixes whose
// not-yet-materialized objects belong to it too.
void Walker::closure(const std::vector<CellRef> &cells, std::vector<CellRef> &outCells,
                     std::vector<std::string> &deep) {
    std::set<int> seen;
    std::vector<int> work;
    auto addObj = [&](int o) {
        o = canonBase(o);
        if (seen.insert(o).second) work.push_back(o);
    };
    auto scanCells = [&](int obj, const std::string &prefix) {
        auto it = cur_.mem.find(obj);
        if (it != cur_.mem.end()) {
            for (auto &kv : it->second.vals)
                if (underPrefix(kv.first, prefix)) for (auto &x : kv.second) addObj(x.obj);
            for (auto &x : it->second.any) addObj(x.obj);
            for (auto &kv : it->second.lazy)
                if (underPrefix(kv.first, prefix)) for (auto &e2 : kv.second) deep.push_back(e2);
        }
        for (auto &e2 : defaultEPs(cur_, obj, prefix)) deep.push_back(canonEP(e2));
    };
    for (auto &c : cells) scanCells(c.obj, c.path);
    while (!work.empty()) {
        int o = work.back();
        work.pop_back();
        const ObjInfo &oi = objs_[o];
        if (oi.kind == OK::Ext && oi.collapsed) deep.push_back(oi.ep.substr(0, oi.ep.size() - 1) + "*");
        scanCells(o, "");
    }
    for (int o : seen) outCells.push_back({o, ""});
    std::sort(deep.begin(), deep.end());
    deep.erase(std::unique(deep.begin(), deep.end()), deep.end());
}

Walker::Res Walker::resolveCells(const std::string &ep, const std::vector<int> &argObjs) {
    Res res;
    size_t i = 0;
    if (ep.size() > 1 && ep[0] == 'p' && isdigit((unsigned char)ep[1])) {
        i = 1;
        while (i < ep.size() && isdigit((unsigned char)ep[i])) ++i;
        int idx = std::stoi(ep.substr(1, i - 1));
        if (idx < 0 || idx >= (int)argObjs.size()) return res;
        res.cells.push_back({argObjs[idx], ""});
    } else if (ep.rfind("g:", 0) == 0) {
        i = 2;
        while (i < ep.size() && !std::strchr(".[*~", ep[i])) ++i;
        ObjInfo oi = objInfo(OK::Global);
        oi.ep = ep.substr(0, i);
        res.cells.push_back({intern("G" + oi.ep, oi), ""});
    } else {
        return res;
    }
    while (i < ep.size() && !res.cells.empty()) {
        char c = ep[i];
        if (c == '.' || c == '[') {
            size_t j = i + 1;
            if (c == '[') j = ep.find(']', i) + 1;
            else while (j < ep.size() && !std::strchr(".[*~", ep[j])) ++j;
            std::string sel = ep.substr(i, j - i);
            for (auto &cell : res.cells) cell.path = normPath(cell.path + sel);
            i = j;
        } else if (c == '*') {
            std::vector<CellRef> next;
            for (auto &cell : res.cells)
                for (auto &x : lookup(cur_, cell.obj, cell.path)) next.push_back(cellOf(x, ""));
            res.cells = std::move(next);
            ++i;
        } else if (c == '~') {
            std::vector<CellRef> next;
            closure(res.cells, next, res.deep);
            res.cells = std::move(next);
            ++i;
        } else break;
    }
    return res;
}

VSet Walker::resolveObjs(const std::string &ep, const std::vector<int> &argObjs, std::vector<std::string> *deep) {
    Res res = resolveCells(ep, argObjs);
    VSet out;
    if (!ep.empty() && ep.back() == '~') {
        for (auto &c : res.cells) vsAdd(out, c.obj, 0);
        if (deep) deep->insert(deep->end(), res.deep.begin(), res.deep.end());
        return out;
    }
    for (auto &c : res.cells) vsUnion(out, lookup(cur_, c.obj, c.path));
    return out;
}

VSet Walker::mapTerms(const std::vector<Term> &terms, const std::vector<int> &argObjs, const Expr &call,
                      FunctionDecl *callee, std::vector<std::string> *deep) {
    VSet out;
    for (auto &t : terms) {
        VSet base;
        if (t.kind == 'E') {
            base = resolveObjs(t.key, argObjs, deep);
        } else if (t.kind == 'G') {
            ObjInfo oi = objInfo(OK::Global);
            oi.ep = t.key;
            vsAdd(base, intern("G" + t.key, oi), 0);
        } else if (t.kind == 'F') {
            std::string site = calleeSite(call, callee, t.key);
            vsAdd(base, siteObj(call, site, 0, t.heap, t.region), t.bits);
        } else if (t.kind == 'M') {
            vsAdd(base, markObj(calleeSite(call, callee, t.key), t.region), 0);
        }
        for (auto &x : base) vsAdd(out, t.prefix.empty() ? x.obj : viewObj(x.obj, t.prefix), x.bits | t.bits);
    }
    return out;
}

EV Walker::applySummary(const Expr &e, FunctionDecl *callee, const Summary &s, std::vector<EV> &args) {
    EV r;
    // Arguments go into temporaries named by parameter index ('p<i>').
    std::vector<int> argObjs;
    for (size_t i = 0; i < args.size(); ++i) {
        int t = tempObj(&e, 100 + (int)i);
        clearSubtree(t, "");
        if (args[i].agg) copyAgg({{t, ""}}, args[i].cells);
        else if (!args[i].v.empty()) cur_.mem[t].vals[""] = args[i].v;
        argObjs.push_back(t);
    }
    std::string calleeName = callee->name;

    // Fresh objects made by this call replace the ones the previous
    // evaluation of it made.
    std::set<std::string> fresh;
    auto collect = [&](const FactCells &fc) {
        for (auto &kv : fc.vals) for (auto &t : kv.second) if (t.kind == 'F') fresh.insert(t.key);
        for (auto &t : fc.any) if (t.kind == 'F') fresh.insert(t.key);
    };
    for (auto &w : s.writes) collect(w.second);
    collect(s.ret);
    std::map<std::string, std::pair<bool, std::string>> freshInfo;
    auto info = [&](const FactCells &fc) {
        for (auto &kv : fc.vals) for (auto &t : kv.second) if (t.kind == 'F') freshInfo[t.key] = {t.heap, t.region};
        for (auto &t : fc.any) if (t.kind == 'F') freshInfo[t.key] = {t.heap, t.region};
    };
    for (auto &w : s.writes) info(w.second);
    info(s.ret);
    for (auto &kv : freshInfo) {
        std::string site = calleeSite(e, callee, kv.first);
        int mr = siteObj(e, site, 0, kv.second.first, kv.second.second);
        freshAlloc(mr, siteObj(e, site, 1, kv.second.first, kv.second.second), kv.second.second);
        clearSubtree(mr, "");
    }

    // Reads of entry memory inside the callee are reads here.
    if (!quiet_ && !unsafe_) {
        for (auto &u : s.uses) {
            if (s.frees.count(u)) continue;   // reported as a double free below
            bool saved = recordUses_;
            recordUses_ = true;
            VSet v;
            Res res = resolveCells(u, argObjs);
            for (auto &c : res.cells) vsUnion(v, lookup(cur_, c.obj, c.path));
            recordUses_ = saved;
            uint8_t b = vsBits(v);
            if (!b) continue;
            if (b & kFreed)
                report(e.loc, "call to '" + calleeName + "' reads a &heap reference after std::dealloc() freed it");
            else {
                std::string region;
                for (auto &x : v) if (region.empty() && (x.bits & kReset)) region = objs_[canonBase(x.obj)].region;
                report(e.loc, "call to '" + calleeName + "' reads an &arena<" + region + "> reference after arena_reset<" +
                       region + ">(), arena_destroy<" + region + ">(), or arena_free_to<" + region +
                       ">() invalidated it");
            }
        }
    } else {
        for (auto &u : s.uses) {
            bool saved = recordUses_;
            recordUses_ = true;
            Res res = resolveCells(u, argObjs);
            for (auto &c : res.cells) lookup(cur_, c.obj, c.path);
            recordUses_ = saved;
        }
    }

    // Resolve everything against the state before the call's effects.
    std::vector<std::string> freedDeep;
    std::vector<VSet> freed;
    for (auto &f : s.frees) freed.push_back(resolveObjs(f, argObjs, &freedDeep));
    std::vector<std::pair<std::string, VSet>> freeTos;
    for (auto &ft : s.freeTos) freeTos.push_back({ft.first, resolveObjs(ft.second, argObjs, nullptr)});
    struct Pending {
        std::vector<CellRef> cells; bool closure;
        std::vector<std::string> deep;
        std::vector<std::pair<std::string, std::vector<CellRef>>> lazy;
        std::vector<std::pair<std::string, VSet>> vals;
        VSet any;
    };
    std::vector<Pending> pend;
    for (auto &w : s.writes) {
        Pending p;
        Res res;
        if (w.first[0] == '!') {
            auto fi = freshInfo.find(w.first.substr(1));
            if (fi == freshInfo.end()) continue;   // unreachable from anything the caller gets
            res.cells.push_back({siteObj(e, calleeSite(e, callee, fi->first), 0, fi->second.first,
                                         fi->second.second), ""});
        } else {
            res = resolveCells(w.first, argObjs);
        }
        p.cells = res.cells;
        p.closure = !w.first.empty() && w.first.back() == '~';
        p.deep = res.deep;
        for (auto &kv : w.second.lazy) {
            std::vector<CellRef> srcs;
            for (auto &ep : kv.second) {
                Res sr = resolveCells(ep, argObjs);
                for (auto &c : sr.cells) srcs.push_back(c);
            }
            p.lazy.push_back({kv.first, srcs});
        }
        std::vector<std::string> dd;
        for (auto &kv : w.second.vals) p.vals.push_back({kv.first, mapTerms(kv.second, argObjs, e, callee, &dd)});
        p.any = mapTerms(w.second.any, argObjs, e, callee, &dd);
        pend.push_back(std::move(p));
    }
    int rt = tempObj(&e, 99);
    clearSubtree(rt, "");
    std::vector<std::pair<std::string, std::vector<CellRef>>> retLazy;
    for (auto &kv : s.ret.lazy) {
        std::vector<CellRef> srcs;
        for (auto &ep : kv.second) {
            Res sr = resolveCells(ep, argObjs);
            for (auto &c : sr.cells) srcs.push_back(c);
        }
        retLazy.push_back({kv.first, srcs});
    }
    std::vector<std::string> dd;
    std::vector<std::pair<std::string, VSet>> retVals;
    for (auto &kv : s.ret.vals) retVals.push_back({kv.first, mapTerms(kv.second, argObjs, e, callee, &dd)});
    VSet retAny = mapTerms(s.ret.any, argObjs, e, callee, &dd);

    // Apply: the call's frees and resets reach every alias the caller has;
    // then what the callee left in memory it can see, which carries its own
    // per-path validity (a buffer freed and replaced is not dangling).
    for (auto &v : freed) {
        if (vsBits(v) & kFreed)
            report(e.loc, "call to '" + calleeName + "' frees memory after std::dealloc() freed it");
        freeObjs(v);
    }
    for (auto &d : freedDeep) {
        if (epRooted(d)) summary.frees.insert(d + "~");
        addDeep(d, kFreed, {});
    }
    for (auto &r2 : s.resets) resetRegion(r2, false);
    for (auto &ft : freeTos) freeTo(ft.first, ft.second);

    for (auto &p : pend) {
        if (p.closure) {
            VSet all = p.any;
            for (auto &kv : p.vals) vsUnion(all, kv.second);
            for (auto &c : p.cells) if (!isTrusted(c.obj)) vsUnion(cur_.mem[c.obj].any, all);
            for (auto &d : p.deep) addDeep(d, 0, all);
            continue;
        }
        bool single = p.cells.size() == 1 && !p.cells[0].weak;
        for (auto &c : p.cells) {
            std::vector<CellRef> dst{{c.obj, c.path, 0, !single}};
            for (auto &l : p.lazy) {
                std::vector<CellRef> d2 = dst;
                for (auto &x : d2) x.path = normPath(x.path + l.first);
                copyAgg(d2, l.second);
            }
            for (auto &kv : p.vals) writeCell(c.obj, c.path + kv.first, kv.second, single);
            if (!p.any.empty() && !isTrusted(c.obj)) vsUnion(cur_.mem[c.obj].any, p.any);
        }
    }
    for (auto &l : retLazy) copyAgg({{rt, l.first}}, l.second);
    for (auto &kv : retVals) writeCell(rt, kv.first, kv.second, true);
    if (!retAny.empty()) vsUnion(cur_.mem[rt].any, retAny);

    for (int t : argObjs) clearSubtree(t, "");
    if (isAgg(e.type)) { r.agg = true; r.cells.push_back({rt, ""}); return r; }
    r.v = lookup(cur_, rt, "");
    clearSubtree(rt, "");
    return r;
}

// ── Ownership of must-drop values ─────────────────────────────────────────────
// A value whose type implements Drop (has 'void drop()'), or an aggregate
// holding one by value, must be dropped or moved exactly once on every path
// before its variable goes out of scope. The checker rides on the walk
// above: each owned *slot* (a local's must-drop leaf: the variable itself
// when its type has drop(), else each must-drop field/element/tuple member,
// recursively) has kO* bits in State::own, joined like everything else, so
// loops reach a fixpoint and break/continue/goto/return/try carry it to
// their targets. A slot is consumed by drop() or a 'consume' method on it,
// by being moved (returned, passed by value, assigned or stored somewhere,
// sent to another thread), or by forget(); a slot still Live when its scope
// ends (block end, break/continue/goto out of the block, return — after the
// defers ran) is a leak. Using a consumed slot, consuming it again, and
// overwriting a Live one are errors too. Must-drop temporaries (call
// results, literals) must be consumed by the full expression that makes
// them. Values reached through references, pointers and containers are
// borrowed: they can be dropped in place (a container's or a state
// machine's own business, decided at run time) but not moved out of.
//
// Methods returning references (or structs holding them) borrow from an
// owned receiver: the result carries a borrow token, and using it after the
// receiver was dropped or moved is an error ('&int p = v.at(0); v.drop();
// *p').

void Walker::ownLeaves(const TypePtr &t0, const std::string &path, std::vector<std::string> &out, int depth) {
    TypePtr t = stripNewtype(t0);
    if (!t || depth > 16 || !mustDrop(t)) return;
    switch (t->kind) {
    case TypeKind::Struct: {
        auto &st = static_cast<const StructType &>(*t);
        if (drv_.sema.hasDropMethod(st) || st.isTaggedUnion || st.fields.empty()) { out.push_back(path); return; }
        for (auto &f : st.fields) {
            if (!mustDrop(f.type)) continue;
            if (f.isAnonymous) { out.push_back(path); return; }   // keep anonymous members whole
            ownLeaves(f.type, normPath(path + "." + f.name), out, depth + 1);
        }
        return;
    }
    case TypeKind::Array: {
        auto &at = static_cast<const ArrayType &>(*t);
        if (at.size > 0 && at.size <= 16 && path.find('[') == std::string::npos) {
            for (int64_t k = 0; k < at.size; ++k)
                ownLeaves(at.element, path + "[" + std::to_string(k) + "]", out, depth + 1);
            return;
        }
        out.push_back(path);
        return;
    }
    case TypeKind::Tuple: {
        auto &tt = static_cast<const TupleType &>(*t);
        for (size_t i = 0; i < tt.elementTypes.size(); ++i)
            ownLeaves(tt.elementTypes[i], path + "." + std::to_string(i), out, depth + 1);
        return;
    }
    default: out.push_back(path); return;
    }
}

std::string Walker::ownDesc(int obj, const std::string &path) const {
    auto it = ownVars_.find(obj);
    std::string n = it != ownVars_.end() ? it->second.name : "value";
    std::string p;
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '|') continue;
        p += path[i];
    }
    return n + p;
}

void Walker::ownDeclare(int obj, const TypePtr &ty, bool init, SourceLocation loc, const std::string &name,
                        const VarDecl *vd) {
    if (!mustDrop(ty)) return;
    OwnVar &ov = ownVars_[obj];
    ov.ty = ty;
    ov.name = name;
    ov.decl = loc;
    ov.vd = vd;
    std::vector<std::string> leaves;
    ownLeaves(ty, "", leaves);
    // Still live from an earlier iteration: its scope end kept it for a
    // defer, which only runs when the function returns.
    for (auto &l : leaves) {
        auto it = cur_.own.find({obj, l});
        if (it != cur_.own.end() && (it->second & kOLive))
            reportN(loc, "'" + ownDesc(obj, l) + "' from the previous iteration is still live here: the "
                         "'defer' that drops it only runs when the function returns, so every iteration "
                         "but the last leaks it; drop it inside the loop instead",
                    {});
    }
    for (auto it = cur_.own.lower_bound({obj, ""}); it != cur_.own.end() && it->first.first == obj;)
        it = cur_.own.erase(it);
    for (auto &l : leaves) cur_.own[{obj, l}] = init ? kOLive : kOUninit;
    if (!frames_.empty()) {
        auto &v = frames_.back().objs;
        if (std::find(v.begin(), v.end(), obj) == v.end()) v.push_back(obj);
    }
}

// The slots of 'obj' at or below 'path', or the one above it.
std::vector<std::map<OwnKey, uint8_t>::iterator> Walker::ownOverlap(int obj, const std::string &path) {
    std::vector<std::map<OwnKey, uint8_t>::iterator> out;
    for (auto it = cur_.own.lower_bound({obj, ""}); it != cur_.own.end() && it->first.first == obj; ++it) {
        const std::string &k = it->first.second;
        if (underPrefix(k, path) || underPrefix(path, k)) { out.push_back(it); continue; }
        if (hasStar(path) || hasStar(k)) {
            std::string ks = starAll(k), ps = starAll(path);
            if (underPrefix(ks, ps) || underPrefix(ps, ks)) out.push_back(it);
        }
    }
    return out;
}

static const char *goneWord(uint8_t b) {
    if ((b & kODrop) && (b & kOMove)) return "dropped or moved";
    return (b & kODrop) ? "dropped" : "moved";
}

void Walker::ownUse(const std::vector<CellRef> &cells, SourceLocation loc, bool write) {
    if (!cur_.live) return;
    for (auto &c : cells) {
        lendCheck(c.obj, loc, write);
        if (!tracked(c.obj)) continue;
        ownCheckObjTokens(c.obj, loc);
        for (auto it : ownOverlap(c.obj, c.path)) {
            uint8_t b = it->second;
            if (b & kOGone) {
                std::vector<std::pair<SourceLocation, std::string>> notes;
                auto g = goneAt_.find(it->first);
                if (g != goneAt_.end()) notes.push_back({g->second, std::string("'") + ownDesc(it->first.first, it->first.second) + "' was " + goneWord(b) + " here"});
                std::string other = ownDesc(it->first.first, it->first.second);
                reportN(loc, std::string(write ? "assignment to part of '" : "use of '") + ownDesc(c.obj, c.path) +
                                 (other == ownDesc(c.obj, c.path) ? std::string("' after it ") : "' after '" + other + "' ") +
                                 ((b & kOLive) ? "may have been " : "was ") + goneWord(b) +
                                 (write ? " (assign the whole value to make it live again)" : ""),
                        notes);
            } else if (b & kOUninit) {
                // Initialized in place: field by field, or through a
                // reference ('init(&x)', 'x.reset()').
                bool partial = c.path != it->first.second && underPrefix(c.path, it->first.second);
                it->second = (write || partial) ? (kOLive | kOBuild) : kOLive;
            }
        }
    }
}

void Walker::ownConsumeKey(int obj, const std::string &path, uint8_t kind, SourceLocation loc) {
    lendCheck(obj, loc, true);
    auto ov = ownVars_.find(obj);
    if (ov != ownVars_.end() && ov->second.pseudo && !path.empty()) {
        // A destructor taking *self apart field by field.
        ownUse({{obj, path}}, loc, false);
        size_t e = path.find_first_of(".[", 1);
        dtorFields_.insert(path.substr(1, e == std::string::npos ? std::string::npos : e - 1));
        return;
    }
    // 'x.f' dropped or moved out of a Drop value 'x': x is unusable as a
    // whole until x.f is assigned again (then x owns a field again).
    for (auto it : ownOverlap(obj, path)) {
        const std::string &k = it->first.second;
        if (k == path || !underPrefix(path, k) || (it->second & kOSub)) continue;
        if (it->second & kOGone) {
            ownUse({{obj, path}}, loc, false);
            return;
        }
        auto sub = cur_.own.find({obj, path});
        if (sub != cur_.own.end() && (sub->second & kOGone)) {
            reportN(loc, std::string(kind == kODrop ? "double drop: '" : "use of '") + ownDesc(obj, path) + "' " +
                             (kind == kODrop ? "" : "after it ") + ((sub->second & kOLive) ? "may have been " : "was ") +
                             goneWord(sub->second) + (kind == kODrop ? " already" : ""),
                    {});
        }
        cur_.own[{obj, path}] = kind | kOSub;
        goneAt_[{obj, path}] = loc;
        return;
    }
    for (auto it : ownOverlap(obj, path)) {
        const std::string &k = it->first.second;
        uint8_t b = it->second;
        if (b & kOSub) {
            if (b & kOGone)
                reportN(loc, "'" + ownDesc(obj, path) + "' is " + (kind == kODrop ? "dropped" : "moved") + " while its field '" +
                                 ownDesc(obj, k) + "' " + ((b & kOLive) ? "may be" : "is") + " " + goneWord(b) +
                                 " (out of it): assign '" + ownDesc(obj, k) + "' a new value first",
                        {});
            it->second = kind | kOSub;
            continue;
        }
        if (b & kOGone) {
            std::vector<std::pair<SourceLocation, std::string>> notes;
            auto g = goneAt_.find(it->first);
            if (g != goneAt_.end()) notes.push_back({g->second, "'" + ownDesc(obj, k) + "' was " + goneWord(b) + " here"});
            std::string what = (kind == kODrop && (b & kODrop) && !(b & kOMove)) ? "double drop: '" + ownDesc(obj, k) + "' "
                               : "use of '" + ownDesc(obj, k) + "' after it ";
            reportN(loc, what + ((b & kOLive) ? "may have been " : "was ") + goneWord(b) +
                             (kind == kODrop && (b & kODrop) && !(b & kOMove) ? " already" : ""),
                    notes);
        }
        it->second = kind;
        goneAt_[it->first] = loc;
    }
    if (isDtor_ && obj == selfObj_) {
        if (path.empty()) dtorFields_.insert("");
        else {
            size_t e = path.find_first_of(".[", 1);
            dtorFields_.insert(path.substr(1, e == std::string::npos ? std::string::npos : e - 1));
        }
    }
}

// Consume what 'cells' name (kind kODrop or kOMove). 'what' is the
// expression being moved, for messages; null for a drop through a call.
void Walker::ownConsumeCells(const std::vector<CellRef> &cells, uint8_t kind, SourceLocation loc, const Expr *what) {
    if (!cur_.live || cells.empty()) return;
    bool anyTracked = false;
    for (auto &c : cells) anyTracked = anyTracked || tracked(c.obj);
    if (anyTracked) {
        if (cells.size() == 1 && !cells[0].weak) {
            ownConsumeKey(cells[0].obj, cells[0].path, kind, loc);
            return;
        }
        reportN(loc, "can't tell which value of Drop type this " + std::string(kind == kODrop ? "drops" : "moves") +
                         " (it may be one of several variables); pick it with if/else",
                {});
        return;
    }
    for (auto &c : cells) {
        const ObjInfo &oi = objs_[c.obj];
        if (oi.kind == OK::Temp) continue;   // part of a temporary: the temporary itself is checked
        // A destructor takes its fields apart.
        if (isDtor_ && oi.kind == OK::Ext && oi.ep == "p0" && !c.path.empty()) {
            size_t e = c.path.find_first_of(".[", 1);
            dtorFields_.insert(c.path.substr(1, e == std::string::npos ? std::string::npos : e - 1));
            continue;
        }
        bool wholeParam = oi.kind == OK::Ext && c.path.empty() && oi.ep.size() > 1 && oi.ep[0] == 'p' &&
                          oi.ep.find_first_not_of("0123456789", 1) == std::string::npos;
        if (kind == kODrop) {
            if (wholeParam && !refDropReported_)
                reportN(loc, "cannot drop a value this function only borrows (it is behind a reference "
                             "parameter); take the parameter by value, or declare the method 'consume'",
                        {});
            continue;   // through any other reference: checked by refDropCheck (syntactic)
        }
        std::string desc = what && what->type ? typeName(what->type) : "?";
        reportN(loc, "cannot move a value of Drop type '" + desc + "' out of memory this function doesn't own "
                     "(it is reached through a reference, a pointer, a global or a container); borrow it with "
                     "'&', or take it out with std::replace()",
                {});
        return;
    }
}

void Walker::ownTakeTemp(const Expr *e) {
    if (!e) return;
    while (e->kind == ExprKind::Cast) e = static_cast<const CastExpr *>(e)->operand.get();
    auto it = std::find(pending_.begin(), pending_.end(), e);
    if (it != pending_.end()) pending_.erase(it);
    if (e->kind == ExprKind::Ternary) {
        auto *t = static_cast<const TernaryExpr *>(e);
        ownTakeTemp(t->then.get());
        ownTakeTemp(t->else_.get());
    }
}

// 'e' (already evaluated to 'v') is moved (kOMove) or dropped (kODrop).
void Walker::ownMove(Expr &e, const EV &v, uint8_t kind) {
    if (!cur_.live || !mustDrop(e.type)) return;
    if (isPlaceExpr(&e)) {
        // A subscript with a run-time index on an owned array.
        for (auto &c : v.cells)
            if (tracked(c.obj) && !ownVars_[c.obj].pseudo && hasStar(c.path)) {
                reportN(e.loc, "ownership of an array element selected at run time can't be tracked; use "
                               "constant indices, or a Vec, for values of Drop type",
                        {});
                return;
            }
        ownConsumeCells(v.cells, kind, e.loc, &e);
        return;
    }
    ownTakeTemp(&e);
}

void Walker::ownAssign(const std::vector<CellRef> &cells, SourceLocation loc) {
    if (!cur_.live) return;
    for (auto &c : cells) {
        lendCheck(c.obj, loc, true);
        if (!tracked(c.obj)) continue;
        // Refilling a field dropped/moved out of a Drop value.
        bool refill = false;
        auto sub = cur_.own.find({c.obj, c.path});
        if (sub != cur_.own.end() && (sub->second & kOSub)) {
            if (sub->second & kOLive)
                reportN(loc, "assigning to '" + ownDesc(c.obj, c.path) + "', which may still hold a value of Drop type "
                             "(it was dropped or moved on some paths only): drop it on every path first",
                        {});
            cur_.own.erase(sub);
            refill = true;
        }
        for (auto it : ownOverlap(c.obj, c.path)) {
            const std::string &k = it->first.second;
            uint8_t b = it->second;
            if (refill) continue;
            // An element picked at run time of an array kept as one unit:
            // which element it overwrites is unknown (not tracked).
            if (hasStar(c.path) && k != c.path) { it->second = kOLive; continue; }
            if (k != c.path && underPrefix(c.path, k)) {
                // A field of a Drop value: the value owns the old contents
                // (unless it is still being built).
                if ((b & kOUninit) || (b & kOBuild)) { it->second = kOLive | kOBuild; continue; }
                if (b & kOLive)
                    reportN(loc, "assigning to '" + ownDesc(c.obj, c.path) + "' overwrites part of '" +
                                     ownDesc(c.obj, k) + "', which owns it: drop it first, or swap it out with "
                                     "std::replace()",
                            {});
                it->second = kOLive;
                continue;
            }
            if (b & kOLive) {
                std::vector<std::pair<SourceLocation, std::string>> notes;
                auto ov = ownVars_.find(c.obj);
                if (ov != ownVars_.end()) notes.push_back({ov->second.decl, "'" + ov->second.name + "' declared here"});
                reportN(loc, std::string("assigning to '") + ownDesc(c.obj, k) + "' overwrites a" +
                                 ((b & kOGone) ? " possibly" : "") + " live value of Drop type, which would leak; "
                                 "drop it first (or std::replace() it)",
                        notes);
            }
            it->second = kOLive;
        }
    }
}

void Walker::ownFlush() {
    for (auto *e : pending_)
        reportN(e->loc, "this temporary of Drop type '" + typeName(e->type) + "' is never dropped (it leaks): "
                        "bind it to a variable and drop it, or pass it on",
                {});
    pending_.clear();
}

bool Walker::keptByDefer(int obj) const {
    auto ov = ownVars_.find(obj);
    if (ov == ownVars_.end() || !ov->second.vd) return false;
    for (size_t i = 0; i < defers_.size(); ++i)
        if (defers_[i].first < curPos_ && deferVars_[i].count(ov->second.vd)) return true;
    return false;
}

void Walker::ownExitFrame(const Frame &f, SourceLocation loc, const char *how) {
    // CodeGen joins the function's scoped threads when it returns, before
    // the function body's own locals are gone.
    if (f.s != fn_.body.get() && !cur_.lent.empty())
        for (int obj : f.locals) lendExit(obj, loc);
    for (int obj : f.objs) {
        bool keep = keptByDefer(obj);
        bool haveLive = false, haveMixed = false;
        std::string live, mixed;
        for (auto it = cur_.own.lower_bound({obj, ""}); it != cur_.own.end() && it->first.first == obj; ++it) {
            if (!(it->second & kOLive) || (it->second & kOSub)) continue;
            if (it->second & kOGone) { if (!haveMixed) { haveMixed = true; mixed = it->first.second; } }
            else if (!haveLive) { haveLive = true; live = it->first.second; }
        }
        if (!keep && cur_.live && (haveLive || haveMixed)) {
            auto &ov = ownVars_[obj];
            bool isMixed = !haveLive;
            std::string path = isMixed ? mixed : live;
            std::string name = ownDesc(obj, path);
            std::string msg = isMixed
                ? "'" + name + "' is dropped or moved on some paths but not on others: it leaks " + how +
                      " on the paths where it is still live"
                : "value of Drop type '" + typeName(ov.ty) + "' leaks " + how + ": '" + name +
                      "' is never dropped; call " + name + ".drop() or add 'defer " + name + ".drop()'";
            if (!isMixed && !path.empty())
                msg = "'" + name + "' (a field of Drop type in '" + ov.name + "') leaks " + how +
                      ": drop it, or move it somewhere that owns it";
            reportN(loc, msg, {{ov.decl, "'" + ov.name + "' declared here"}});
        }
        if (!keep)
            for (auto it = cur_.own.lower_bound({obj, ""}); it != cur_.own.end() && it->first.first == obj;)
                it = cur_.own.erase(it);
    }
}

void Walker::ownExitAbove(size_t depth, SourceLocation loc, const char *how) {
    for (size_t i = frames_.size(); i-- > depth;) ownExitFrame(frames_[i], loc, how);
}

void Walker::ownExitForGoto(const std::string &label, SourceLocation loc) {
    auto it = labelScopes_.find(label);
    if (it == labelScopes_.end()) return;
    for (size_t i = frames_.size(); i-- > 0;) {
        if (frames_[i].s && it->second.count(frames_[i].s)) break;
        ownExitFrame(frames_[i], loc, "when 'goto' leaves its scope");
    }
}

std::string Walker::danglingIn(const VSet &v) const {
    for (auto &e : v) {
        if (objs_[e.obj].kind != OK::Token) continue;
        auto t = cur_.tok.find(e.obj);
        if (t == cur_.tok.end()) continue;
        for (auto &k : t->second) {
            // Pointing straight into the owner: its own use check reports it.
            bool direct = false;
            for (auto &x : v) if (canonBase(x.obj) == k.first) direct = true;
            if (direct) continue;
            auto o = cur_.own.find(k);
            if (o == cur_.own.end() || (o->second & kOGone)) {
                uint8_t b = o == cur_.own.end() ? kOGone : o->second;
                return "'" + ownDesc(k.first, k.second) + "' " +
                       (o == cur_.own.end() ? std::string("went out of scope") : std::string((b & kOLive) ? "may have been " : "was ") + goneWord(b));
            }
        }
    }
    return "";
}

void Walker::ownCheckVal(const Expr &e, const VSet &v) {
    if (quiet_ || unsafe_ || !cur_.live || cur_.tok.empty()) return;
    std::string why = danglingIn(v);
    if (!why.empty())
        reportN(e.loc, "use of a reference borrowed from " + why.substr(0, why.find('\'', 1) + 1) + " after " +
                           why + " (it points into that value)",
                {});
}

void Walker::ownCheckObjTokens(int obj, SourceLocation loc) {
    if (quiet_ || unsafe_ || cur_.tok.empty()) return;
    auto it = cur_.mem.find(obj);
    if (it == cur_.mem.end()) return;
    std::string why = danglingIn(it->second.any);
    for (auto &kv : it->second.vals) if (why.empty()) why = danglingIn(kv.second);
    if (!why.empty())
        reportN(loc, "use of '" + ownDesc(obj, "") + "', which borrows from " + why.substr(0, why.find('\'', 1) + 1) +
                         ", after " + why,
                {});
}

int Walker::ownToken(const Expr &site, const std::vector<OwnKey> &owners) {
    int t = intern("K" + ptrKey(&site), objInfo(OK::Token));
    cur_.tok[t] = owners;
    return t;
}

// At each return (after the defers ran): every owned slot still in scope
// must be gone.
void Walker::ownEndFunction() {
    // Every owned slot left, including ones a 'defer' was expected to drop.
    Frame all{nullptr, {}};
    for (auto &kv : cur_.own) {
        auto ov = ownVars_.find(kv.first.first);
        if (ov == ownVars_.end() || ov->second.pseudo) continue;
        if (all.objs.empty() || all.objs.back() != kv.first.first) all.objs.push_back(kv.first.first);
    }
    int saved = curPos_;
    curPos_ = -1;   // no defer is pending any more
    ownExitFrame(all, endLoc_, "when the function returns");
    curPos_ = saved;
}


// A match on a value of Drop type that binds a Drop payload moves the
// subject into the binding (in every arm: an arm without a payload has
// nothing left to drop). An arm that matches a Drop payload without
// binding it would leak it.
bool Walker::ownMatchSubject(Expr *subject, const EV &subj, const std::vector<const std::vector<MatchPattern> *> &arms) {
    if (!subject || !cur_.live || !mustDrop(subject->type)) return false;
    bool binds = false;
    for (auto *pats : arms)
        for (auto &p : *pats) if (!p.bindName.empty() && p.payloadType && mustDrop(p.payloadType)) binds = true;
    if (!binds) return false;
    ownMove(*subject, subj, kOMove);
    for (auto *pats : arms)
        for (auto &p : *pats) {
            if (!p.bindName.empty()) continue;
            if ((p.payloadType && mustDrop(p.payloadType)) || p.kind == PatternKind::Wildcard)
                reportN(subject->loc, std::string("this match moves its value of Drop type '") + typeName(subject->type) +
                                          "', but the arm '" + (p.kind == PatternKind::Wildcard ? std::string("default") : "case " + p.ident) +
                                          "' doesn't bind what it matches, which leaks it; bind it ('case " +
                                          (p.ident.empty() ? std::string("some") : p.ident) + "(v)') and drop it",
                        {});
        }
    return true;
}

void Walker::ownBind(const MatchPattern &p, SourceLocation loc) {
    if (p.bindName.empty() || !p.payloadType || !mustDrop(p.payloadType) || binds_.empty()) return;
    ownDeclare(binds_.back().second, p.payloadType, true, loc, p.bindName, nullptr);
}


// ── Scoped threads ──────────────────────────────────────────────────────────
// spawn_scoped(f, &x) lends x to the new thread until join(h) (or the
// function's return, where CodeGen joins every scoped thread): meanwhile
// this thread may not use x (only read it, for a const lend), drop it,
// free it, or let it go out of scope.
std::string Walker::objName(int obj) const {
    auto l = locals_.find(obj);
    if (l != locals_.end()) return "'" + l->second.name + "'";
    auto o = ownVars_.find(obj);
    if (o != ownVars_.end()) return "'" + o->second.name + "'";
    const ObjInfo &oi = objs_[obj];
    if (oi.kind == OK::Alloc) return "a heap/arena allocation";
    return "memory";
}

void Walker::lendCheck(int obj, SourceLocation loc, bool write) {
    if (cur_.lent.empty()) return;
    auto it = cur_.lent.find(canonBase(obj));
    if (it == cur_.lent.end()) return;
    for (int t : it->second) {
        auto sp = spawns_.find(t);
        if (sp == spawns_.end()) continue;
        if (sp->second.sync) continue;              // a Mutex, atomics: made to be shared
        if (!write && !sp->second.mut) continue;   // reading what is shared for reading
        reportN(loc, objName(canonBase(obj)) + " is lent to the thread started here by spawn_scoped, which may still "
                     "be using it: join that thread (join(h)) before touching it again",
                {{sp->second.loc, "the thread was started here"}});
        return;
    }
}

void Walker::lendExit(int obj, SourceLocation loc) {
    auto it = cur_.lent.find(obj);
    if (it == cur_.lent.end()) return;
    for (int t : it->second) {
        auto sp = spawns_.find(t);
        if (sp == spawns_.end()) continue;
        reportN(loc, objName(obj) + " goes away while the thread started by spawn_scoped may still be using it: "
                     "join that thread (join(h)) first",
                {{sp->second.loc, "the thread was started here"}});
        break;
    }
    cur_.lent.erase(it);
}

// ── Statements ─────────────────────────────────────────────────────────────────
void Walker::jumpTo(bool isBreak, const std::string &label, SourceLocation loc) {
    for (auto it = targets_.rbegin(); it != targets_.rend(); ++it) {
        if (!label.empty() ? it->label == label : (isBreak || !it->isSwitch)) {
            if (!isBreak && it->isSwitch) continue;
            ownFlush();
            ownExitAbove(it->depth, loc, isBreak ? "when 'break' leaves its scope" : "when 'continue' leaves its scope");
            State &t = isBreak ? it->brk : it->cont;
            t = join(t, cur_);
            cur_.live = false;
            return;
        }
    }
}

void Walker::doReturn(bool isError) {
    if (inDefer_) { cur_.live = false; return; }
    ownFlush();
    State saved = cur_;
    ++inDefer_;
    for (size_t i = defers_.size(); i-- > 0;) {
        if (defers_[i].first >= curPos_) continue;
        const DeferStmt *ds = defers_[i].second;
        if (ds->isErrDefer && !isError) continue;
        int savedPos = curPos_;
        if (ds->body) stmt(*ds->body);
        curPos_ = savedPos;
    }
    --inDefer_;
    if (cur_.live) { ownFlush(); ownEndFunction(); }
    cur_.lent.clear();   // CodeGen joins every scoped thread on the way out
    exit_ = join(exit_, cur_);
    cur_ = std::move(saved);
    cur_.live = false;
}

void Walker::loopOnce(Expr *cond, Stmt *body, Expr *incr, bool doWhile, const std::string &label,
                      bool alwaysTrue, State &back, State &exitS) {
    targets_.push_back({label, false, State{}, State{}, frames_.size()});
    // The condition and increment are full expressions of their own.
    if (!doWhile) {
        ownFlush();
        dropTemps(cur_);
        State t, f;
        this->cond(cond, t, f);
        dropTemps(t);
        dropTemps(f);
        exitS = alwaysTrue ? State{} : std::move(f);
        cur_ = std::move(t);
        if (body) stmt(*body);
        cur_ = join(cur_, targets_.back().cont);
        if (incr) { ownFlush(); dropTemps(cur_); rval(*incr); ownFlush(); dropTemps(cur_); }
        back = cur_;
    } else {
        if (body) stmt(*body);
        cur_ = join(cur_, targets_.back().cont);
        ownFlush();
        dropTemps(cur_);
        State t, f;
        this->cond(cond, t, f);
        dropTemps(t);
        dropTemps(f);
        exitS = alwaysTrue ? State{} : std::move(f);
        back = std::move(t);
    }
    ownFlush();
    exitS = join(exitS, targets_.back().brk);
    targets_.pop_back();
}

void Walker::loop(const Stmt &s, Expr *cond, Stmt *body, Expr *incr, bool doWhile, const std::string &label) {
    bool alwaysTrue = !cond;
    if (cond && cond->kind == ExprKind::IntLit) alwaysTrue = static_cast<IntLitExpr *>(cond)->value != 0;
    if (cond && cond->kind == ExprKind::BoolLit) alwaysTrue = static_cast<BoolLitExpr *>(cond)->value;
    State head = cur_;
    State back, exitS;
    ++quiet_;
    for (int i = 0; i < kMaxIters; ++i) {
        cur_ = head;
        loopOnce(cond, body, incr, doWhile, label, alwaysTrue, back, exitS);
        foldIteration(back, &s);
        State next = join(head, back);
        if (next == head) break;
        head = std::move(next);
    }
    --quiet_;
    cur_ = head;
    loopOnce(cond, body, incr, doWhile, label, alwaysTrue, back, exitS);
    cur_ = std::move(exitS);
}

void Walker::stmt(Stmt &s) {
    if (s.kind != StmtKind::Compound) { ownFlush(); dropTemps(cur_); }
    int savedPos = curPos_;
    auto po = order_.find(&s);
    if (po != order_.end()) curPos_ = po->second;
    switch (s.kind) {
    case StmtKind::Compound: {
        auto &cs = static_cast<CompoundStmt &>(s);
        frames_.push_back({&s, {}});
        for (auto &c : cs.body) if (c) stmt(*c);
        if (cur_.live) {
            ownFlush();
            int saved = curPos_;
            auto be = blockEnd_.find(&s);
            if (be != blockEnd_.end()) curPos_ = be->second + 1;
            ownExitFrame(frames_.back(), cs.endLoc, "at the end of its scope");
            curPos_ = saved;
        }
        frames_.pop_back();
        break;
    }
    case StmtKind::Expr: {
        auto &es = static_cast<ExprStmt &>(s);
        if (es.expr && cur_.live) rval(*es.expr);
        break;
    }
    case StmtKind::If: {
        auto &is = static_cast<IfStmt &>(s);
        State ts, fs;
        cond(is.cond.get(), ts, fs);
        cur_ = std::move(ts);
        if (is.then) stmt(*is.then);
        State t = std::move(cur_);
        cur_ = std::move(fs);
        if (is.else_) stmt(*is.else_);
        cur_ = join(t, cur_);
        break;
    }
    case StmtKind::IfConst: {
        auto &is = static_cast<IfConstStmt &>(s);
        if (is.constResult) {
            if (*is.constResult) { if (is.then) stmt(*is.then); }
            else if (is.else_) stmt(*is.else_);
            break;
        }
        State base = cur_;
        if (is.then) stmt(*is.then);
        State t = std::move(cur_);
        cur_ = std::move(base);
        if (is.else_) stmt(*is.else_);
        cur_ = join(t, cur_);
        break;
    }
    case StmtKind::While: case StmtKind::DoWhile: {
        auto &ws = static_cast<WhileStmt &>(s);
        loop(s, ws.cond.get(), ws.body.get(), nullptr, ws.isDoWhile, ws.label);
        break;
    }
    case StmtKind::For: {
        auto &fs = static_cast<ForStmt &>(s);
        frames_.push_back({&s, {}});
        if (fs.init) stmt(*fs.init);
        loop(s, fs.cond.get(), fs.body.get(), fs.incr.get(), false, fs.label);
        if (cur_.live) ownExitFrame(frames_.back(), fs.body ? fs.body->loc : s.loc, "after its loop");
        frames_.pop_back();
        break;
    }
    case StmtKind::Return: {
        auto &rs = static_cast<ReturnStmt &>(s);
        if (!cur_.live) break;
        if (rs.value) {
            if (rs.value->kind == ExprKind::Compound) {
                clearSubtree(retObj_, "");
                writeInit({{retObj_, ""}}, *rs.value, fn_.returnType);
            } else {
                EV v = rval(*rs.value);
                ownMove(*rs.value, v, kOMove);
                clearSubtree(retObj_, "");
                if (v.agg) copyAgg({{retObj_, ""}}, v.cells);
                else if (!v.v.empty()) cur_.mem[retObj_].vals[""] = v.v;
            }
        }
        if (cur_.live) { endLoc_ = s.loc; doReturn(false); }
        break;
    }
    case StmtKind::Break:
        if (cur_.live) jumpTo(true, static_cast<BreakStmt &>(s).label, s.loc);
        break;
    case StmtKind::Continue:
        if (cur_.live) jumpTo(false, static_cast<ContinueStmt &>(s).label, s.loc);
        break;
    case StmtKind::Goto: {
        auto &gs = static_cast<GotoStmt &>(s);
        if (cur_.live) {
            ownFlush();
            ownExitForGoto(gs.label, gs.loc);
            labels_[gs.label] = join(labels_[gs.label], cur_);
            cur_.live = false;
        }
        break;
    }
    case StmtKind::Label: {
        auto &ls = static_cast<LabelStmt &>(s);
        auto it = labels_.find(ls.label);
        if (it != labels_.end()) cur_ = join(cur_, it->second);
        if (ls.body) stmt(*ls.body);
        break;
    }
    case StmtKind::VarDecl: {
        auto &vs = static_cast<VarDeclStmt &>(s);
        if (!cur_.live || !vs.decl) break;
        int o = varObj(vs.decl);
        if (objs_[o].kind == OK::Global) break;   // a static local's initializer is a constant
        clearSubtree(o, "");
        if (vs.init) writeInit({{o, ""}}, *vs.init, vs.resolvedType);
        if (cur_.live) ownDeclare(o, vs.resolvedType, vs.init != nullptr, vs.loc, vs.name, vs.decl);
        locals_[o] = {vs.name, vs.loc};
        if (!frames_.empty()) frames_.back().locals.push_back(o);
        if (vs.init) {
            const Expr *ie = vs.init.get();
            while (ie && ie->kind == ExprKind::Cast) ie = static_cast<const CastExpr *>(ie)->operand.get();
            auto st = spawnTok_.find(ie);
            if (st != spawnTok_.end()) handleTok_[o] = st->second;
        }
        break;
    }
    case StmtKind::MultiVarDecl:
        for (auto &d : static_cast<MultiVarDeclStmt &>(s).decls) if (d) stmt(*d);
        break;
    case StmtKind::Unsafe:
        ++unsafe_;
        if (auto *b = static_cast<UnsafeStmt &>(s).body.get()) stmt(*b);
        --unsafe_;
        break;
    case StmtKind::Defer:
        break;   // registered up front; runs at each return (see doReturn)
    case StmtKind::Match: {
        auto &ms = static_cast<MatchStmt &>(s);
        EV subj = cur_.live && ms.subject ? rval(*ms.subject) : EV{};
        std::vector<const std::vector<MatchPattern> *> armPats;
        for (auto &arm : ms.arms) armPats.push_back(&arm.patterns);
        bool moving = ownMatchSubject(ms.subject.get(), subj, armPats);
        ownFlush();
        State base = cur_;
        State acc;
        bool exhaustive = false, sawNone = false, sawSome = false;
        for (auto &arm : ms.arms) {
            cur_ = base;
            size_t nb = binds_.size();
            frames_.push_back({nullptr, {}});
            for (auto &p : arm.patterns) {
                if (p.kind == PatternKind::Wildcard && p.ident.empty()) exhaustive = true;
                if (p.resolvedTag == 0) sawNone = true;
                if (p.resolvedTag == 1) sawSome = true;
                bindPattern(p, subj, ms.subject ? ms.subject->type : nullptr);
                if (moving) ownBind(p, ms.subject->loc);
            }
            if (arm.body) stmt(*arm.body);
            if (cur_.live) ownExitFrame(frames_.back(), arm.body ? arm.body->loc : s.loc, "at the end of its match arm");
            frames_.pop_back();
            binds_.resize(nb);
            acc = join(acc, cur_);
        }
        TypePtr st = ms.subject ? stripNewtype(ms.subject->type) : nullptr;
        bool nullLike = st && (st->kind == TypeKind::Optional || st->kind == TypeKind::Pointer ||
                               (st->kind == TypeKind::Reference && static_cast<ReferenceType &>(*st).nullable));
        if (nullLike && sawNone && sawSome) exhaustive = true;
        cur_ = exhaustive ? acc : join(acc, base);
        break;
    }
    case StmtKind::Switch: {
        auto &sw = static_cast<SwitchStmt &>(s);
        if (cur_.live && sw.controlling) rval(*sw.controlling);
        State dispatch = cur_;
        bool hasDefault = false;
        targets_.push_back({"", true, State{}, State{}, frames_.size()});
        frames_.push_back({&s, {}});
        State fall;
        for (auto &c : sw.cases) {
            if (c.isDefault) hasDefault = true;
            cur_ = join(dispatch, fall);
            for (auto &b : c.body) if (b) stmt(*b);
            fall = cur_;
        }
        cur_ = fall;
        if (cur_.live) ownExitFrame(frames_.back(), s.loc, "at the end of the switch");
        fall = cur_;
        frames_.pop_back();
        State out = join(fall, targets_.back().brk);
        targets_.pop_back();
        cur_ = hasDefault ? out : join(out, dispatch);
        break;
    }
    case StmtKind::Asm: {
        auto &as = static_cast<AsmStmt &>(s);
        if (!cur_.live) break;
        for (auto &x : as.inputExprs) if (x) rval(*x);
        for (auto &x : as.outputExprs)
            if (x) {
                auto cells = lval(*x);
                if (isTracked(x->type)) write(cells, {});
            }
        break;
    }
    default: break;
    }
    curPos_ = savedPos;
}

void Walker::walkBody() {
    cur_ = State{};
    cur_.live = true;
    exit_ = State{};
    targets_.clear();
    binds_.clear();
    frames_.clear();
    pending_.clear();
    curPos_ = 0;
    // The function's own frame: by-value parameters of Drop type are owned
    // here; a destructor's *self is tracked for use after it was dropped.
    frames_.push_back({fn_.body.get(), {}});
    for (size_t i = 0; i < fn_.paramVars.size() && i < fn_.params.size(); ++i) {
        const VarDecl *pv = fn_.paramVars[i];
        if (pv && mustDrop(fn_.params[i].type))
            ownDeclare(varObj(pv), fn_.params[i].type, true, fn_.params[i].loc, fn_.params[i].name, pv);
    }
    if (isDtor_ && selfObj_ >= 0) cur_.own[{selfObj_, ""}] = kOLive;
    stmt(*fn_.body);
    if (cur_.live) {
        curPos_ = 1 << 30;
        endLoc_ = fn_.body->endLoc;
        doReturn(false);
    }
    frames_.clear();
}

void Walker::run() {
    for (size_t i = 0; i < fn_.paramVars.size(); ++i)
        if (fn_.paramVars[i]) paramIdx_[fn_.paramVars[i]] = (int)i;
    retObj_ = intern("R", objInfo(OK::Ret));
    // Preorder statement numbering: a return runs exactly the defers
    // registered textually before it (CodeGen's order).
    int n = 0;
    bool hasLabels = false;
    visitStmt(fn_.body.get(), [&](Stmt &s) {
        order_[&s] = ++n;
        if (s.kind == StmtKind::Defer) defers_.push_back({n, static_cast<DeferStmt *>(&s)});
        if (s.kind == StmtKind::Label) hasLabels = true;
    }, [](Expr &) {});
    for (auto &kv : order_) {
        if (kv.first->kind != StmtKind::Compound) continue;
        int mx = kv.second;
        visitStmt(const_cast<Stmt *>(kv.first), [&](Stmt &x) { auto o = order_.find(&x); if (o != order_.end()) mx = std::max(mx, o->second); }, [](Expr &) {});
        blockEnd_[kv.first] = mx;
    }

    // Each allocation site's outermost enclosing loop (see foldIteration).
    std::function<void(Stmt *, const Stmt *)> mapLoops = [&](Stmt *st, const Stmt *outer) {
        if (!st) return;
        bool isLoop = st->kind == StmtKind::While || st->kind == StmtKind::DoWhile || st->kind == StmtKind::For;
        if (isLoop && !outer) {
            visitStmt(st, [](Stmt &) {}, [&](Expr &e) { outerLoop_[&e] = st; });
            return;
        }
        switch (st->kind) {
        case StmtKind::Compound: for (auto &c : static_cast<CompoundStmt *>(st)->body) mapLoops(c.get(), outer); break;
        case StmtKind::If: mapLoops(static_cast<IfStmt *>(st)->then.get(), outer); mapLoops(static_cast<IfStmt *>(st)->else_.get(), outer); break;
        case StmtKind::IfConst: mapLoops(static_cast<IfConstStmt *>(st)->then.get(), outer); mapLoops(static_cast<IfConstStmt *>(st)->else_.get(), outer); break;
        case StmtKind::Label: mapLoops(static_cast<LabelStmt *>(st)->body.get(), outer); break;
        case StmtKind::Unsafe: mapLoops(static_cast<UnsafeStmt *>(st)->body.get(), outer); break;
        case StmtKind::Defer: mapLoops(static_cast<DeferStmt *>(st)->body.get(), outer); break;
        case StmtKind::Match: for (auto &a : static_cast<MatchStmt *>(st)->arms) mapLoops(a.body.get(), outer); break;
        case StmtKind::Switch: for (auto &c : static_cast<SwitchStmt *>(st)->cases) for (auto &b : c.body) mapLoops(b.get(), outer); break;
        case StmtKind::MultiVarDecl: for (auto &d : static_cast<MultiVarDeclStmt *>(st)->decls) mapLoops(d.get(), outer); break;
        default: break;
        }
    };
    mapLoops(fn_.body.get(), nullptr);

    // Ownership: what each defer names, each label's enclosing blocks, and
    // whether this is a destructor (drop() or a 'consume' method).
    for (auto &d : defers_) {
        std::set<const VarDecl *> vars;
        visitStmt(d.second->body.get(), [](Stmt &) {}, [&](Expr &e) {
            if (e.kind == ExprKind::Ident && static_cast<IdentExpr &>(e).resolved)
                vars.insert(static_cast<IdentExpr &>(e).resolved);
        });
        deferVars_.push_back(std::move(vars));
    }
    {
        std::vector<const Stmt *> stack;
        std::function<void(Stmt *)> scan = [&](Stmt *st) {
            if (!st) return;
            bool block = st->kind == StmtKind::Compound || st->kind == StmtKind::For || st->kind == StmtKind::Switch;
            if (block) stack.push_back(st);
            if (st->kind == StmtKind::Label)
                labelScopes_[static_cast<LabelStmt *>(st)->label] = std::set<const Stmt *>(stack.begin(), stack.end());
            switch (st->kind) {
            case StmtKind::Compound: for (auto &c : static_cast<CompoundStmt *>(st)->body) scan(c.get()); break;
            case StmtKind::If: scan(static_cast<IfStmt *>(st)->then.get()); scan(static_cast<IfStmt *>(st)->else_.get()); break;
            case StmtKind::IfConst: scan(static_cast<IfConstStmt *>(st)->then.get()); scan(static_cast<IfConstStmt *>(st)->else_.get()); break;
            case StmtKind::While: case StmtKind::DoWhile: scan(static_cast<WhileStmt *>(st)->body.get()); break;
            case StmtKind::For: scan(static_cast<ForStmt *>(st)->init.get()); scan(static_cast<ForStmt *>(st)->body.get()); break;
            case StmtKind::Label: scan(static_cast<LabelStmt *>(st)->body.get()); break;
            case StmtKind::Unsafe: scan(static_cast<UnsafeStmt *>(st)->body.get()); break;
            case StmtKind::Match: for (auto &a : static_cast<MatchStmt *>(st)->arms) scan(a.body.get()); break;
            case StmtKind::Switch: for (auto &c : static_cast<SwitchStmt *>(st)->cases) for (auto &b : c.body) scan(b.get()); break;
            default: break;
            }
            if (block) stack.pop_back();
        };
        scan(fn_.body.get());
    }
    isDtor_ = drv_.sema.isConsumeMethod(&fn_) && !fn_.params.empty();
    if (isDtor_) {
        selfObj_ = extObj("p0");
        OwnVar &ov = ownVars_[selfObj_];
        const ReferenceType *rt = refOf(fn_.params[0].type);
        ov.ty = rt ? rt->base : nullptr;
        ov.name = "self";
        ov.decl = fn_.loc;
        ov.pseudo = true;
    }

    if (hasLabels) {
        ++quiet_;
        for (int i = 0; i < kMaxIters; ++i) {
            auto before = labels_;
            walkBody();
            if (labels_ == before) break;
        }
        --quiet_;
    }
    walkBody();
    // A destructor must take apart every field of Drop type it owns (on
    // some path: which fields still hold a value can depend on run-time
    // state, like a coroutine's resume point).
    if (isDtor_ && !dtorFields_.count("")) {
        const StructType *st = structOf(fn_.params[0].type);
        if (st)
            for (auto &f : st->fields)
                if (mustDrop(f.type) && !dtorFields_.count(f.name)) {
                    std::string owner = st->name, m = fn_.name.size() > owner.size() + 1 ? fn_.name.substr(owner.size() + 1) : fn_.name;
                    owner = typeName(refOf(fn_.params[0].type)->base);
                    reportN(fn_.loc, "'" + owner + "::" + m + "()' never drops field '" + f.name + "' (of Drop type '" +
                                         typeName(f.type) + "'): a destructor must drop or move every field it owns "
                                         "(self." + f.name + ".drop())",
                            {});
                }
    }
    buildSummary();
}

Term Walker::termOf(const VEnt &e) const {
    Term t;
    t.kind = 0;
    int b = e.obj;
    if (objs_[b].kind == OK::View) { t.prefix = objs_[b].prefix; b = objs_[b].base; }
    const ObjInfo &oi = objs_[b];
    switch (oi.kind) {
    case OK::Ext:    t.kind = 'E'; t.key = oi.ep; t.bits = e.bits; break;
    case OK::Global: t.kind = 'G'; t.key = oi.ep; t.bits = e.bits; break;
    case OK::Alloc:  t.kind = 'F'; t.key = oi.site; t.bits = e.bits; t.heap = oi.heap; t.region = oi.region; break;
    case OK::Mark:   t.kind = 'M'; t.key = oi.site; t.region = oi.region; break;
    default: break;
    }
    return t;
}

void Walker::buildSummary() {
    auto toTerms = [&](const VSet &v) {
        std::vector<Term> out;
        for (auto &e : v) {
            Term t = termOf(e);
            if (!t.kind) continue;
            auto it = std::lower_bound(out.begin(), out.end(), t);
            if (it == out.end() || *it != t) out.insert(it, t);
        }
        return out;
    };
    auto toFacts = [&](const ObjCells &oc) {
        FactCells fc;
        for (auto &kv : oc.vals) {
            auto t = toTerms(kv.second);
            fc.vals[kv.first] = std::move(t);
        }
        for (auto &kv : oc.lazy) {
            std::vector<std::string> eps;
            for (auto &e : kv.second) if (epRooted(e)) eps.push_back(e);
            if (!eps.empty()) fc.lazy[kv.first] = eps;
        }
        fc.any = toTerms(oc.any);
        return fc;
    };
    if (!exit_.live) return;
    for (auto &om : exit_.mem) {
        const ObjInfo &oi = objs_[om.first];
        if (oi.kind == OK::Ext && epRooted(oi.ep))
            summary.writes[oi.collapsed ? oi.ep : oi.ep + "*"] = toFacts(om.second);
        else if (oi.kind == OK::Global)
            summary.writes[oi.ep] = toFacts(om.second);
        else if (oi.kind == OK::Ret)
            summary.ret = toFacts(om.second);
        else if (oi.kind == OK::Alloc) {
            // What the callee stored into memory it allocated ('!site'),
            // for the caller's copy of that allocation.
            FactCells f = toFacts(om.second);
            FactCells &w = summary.writes["!" + oi.site];
            for (auto &kv : f.vals) sortedUnion(w.vals[kv.first], kv.second);
            for (auto &kv : f.lazy) sortedUnion(w.lazy[kv.first], kv.second);
            sortedUnion(w.any, f.any);
        }
    }
    for (auto &d : exit_.deep) {
        if (!epRooted(d.first)) continue;
        if (d.second.bits & kFreed) summary.frees.insert(d.first + "~");
        if (!d.second.vals.empty()) {
            auto t = toTerms(d.second.vals);
            summary.writes[d.first + "~"].any = t;
        }
    }
}

// ── Summary join (monotone iteration within a recursive component) ────────────
std::vector<Term> identityTerm(const std::string &designator, const std::string &path) {
    if (designator.empty() || designator[0] == '!') return {};
    std::string ep = designator.back() == '~' ? designator : designator + path;
    Term t;
    t.kind = 'E';
    t.key = canonEP(ep);
    return {t};
}

void joinFacts(FactCells &into, const FactCells &other, const std::string &designator) {
    std::set<std::string> paths;
    for (auto &kv : into.vals) paths.insert(kv.first);
    for (auto &kv : other.vals) paths.insert(kv.first);
    for (auto &p : paths) {
        auto a = into.vals.find(p);
        auto b = other.vals.find(p);
        std::vector<Term> v = a != into.vals.end() ? a->second : identityTerm(designator, p);
        sortedUnion(v, b != other.vals.end() ? b->second : identityTerm(designator, p));
        into.vals[p] = v;
    }
    std::set<std::string> pre;
    for (auto &kv : into.lazy) pre.insert(kv.first);
    for (auto &kv : other.lazy) pre.insert(kv.first);
    for (auto &p : pre) {
        auto a = into.lazy.find(p);
        auto b = other.lazy.find(p);
        bool noEntry = designator.empty() || designator[0] == '!';
        std::vector<std::string> v = a != into.lazy.end() ? a->second
                                     : noEntry ? std::vector<std::string>{} : std::vector<std::string>{designator + p};
        std::vector<std::string> w = b != other.lazy.end() ? b->second
                                     : noEntry ? std::vector<std::string>{} : std::vector<std::string>{designator + p};
        sortedUnion(v, w);
        into.lazy[p] = v;
    }
    sortedUnion(into.any, other.any);
}

Summary joinSummary(const Summary &a, const Summary &b) {
    Summary r = a;
    r.frees.insert(b.frees.begin(), b.frees.end());
    r.resets.insert(b.resets.begin(), b.resets.end());
    r.freeTos.insert(b.freeTos.begin(), b.freeTos.end());
    r.uses.insert(b.uses.begin(), b.uses.end());
    for (auto &w : b.writes) {
        auto it = r.writes.find(w.first);
        if (it == r.writes.end()) r.writes[w.first] = w.second;
        else joinFacts(it->second, w.second, w.first);
    }
    joinFacts(r.ret, b.ret, "");
    return r;
}

// ── Driver ─────────────────────────────────────────────────────────────────────
void FlowDriver::run() {
    for (auto *fn : fns_) {
        checked_.insert(fn);
        byName_.emplace(fn->name, fn);
    }
    // Call graph.
    std::unordered_map<FunctionDecl *, std::vector<FunctionDecl *>> calls;
    for (auto *fn : fns_) {
        auto &out = calls[fn];
        visitStmt(fn->body.get(), [](Stmt &) {}, [&](Expr &e) {
            FunctionDecl *t = nullptr;
            if (e.kind == ExprKind::Call) {
                auto &c = static_cast<CallExpr &>(e);
                if (c.callee && c.callee->kind == ExprKind::Ident)
                    t = static_cast<IdentExpr &>(*c.callee).resolvedFn;
            } else if (e.kind == ExprKind::Binary) {
                t = static_cast<BinaryExpr &>(e).resolvedOperator;
            }
            if ((t = bodyOf(t))) out.push_back(t);
        });
    }
    // Tarjan's SCCs, emitted callees first.
    std::unordered_map<FunctionDecl *, int> index, low;
    std::unordered_set<FunctionDecl *> onStack;
    std::vector<FunctionDecl *> stack;
    std::vector<std::vector<FunctionDecl *>> sccs;
    int counter = 0;
    std::function<void(FunctionDecl *)> strong = [&](FunctionDecl *v) {
        index[v] = low[v] = counter++;
        stack.push_back(v);
        onStack.insert(v);
        for (auto *w : calls[v]) {
            if (!index.count(w)) { strong(w); low[v] = std::min(low[v], low[w]); }
            else if (onStack.count(w)) low[v] = std::min(low[v], index[w]);
        }
        if (low[v] == index[v]) {
            sccs.emplace_back();
            FunctionDecl *w;
            do {
                w = stack.back();
                stack.pop_back();
                onStack.erase(w);
                scc_[w] = (int)sccs.size() - 1;
                sccs.back().push_back(w);
            } while (w != v);
        }
    };
    for (auto *fn : fns_) if (!index.count(fn)) strong(fn);

    std::vector<Diag> all;
    for (auto &comp : sccs) {
        bool recursive = comp.size() > 1;
        if (!recursive)
            for (auto *t : calls[comp[0]]) if (t == comp[0]) recursive = true;
        std::unordered_map<FunctionDecl *, std::vector<Diag>> diags;
        if (recursive) for (auto *fn : comp) sums_[fn] = Summary{};
        for (int iter = 0; iter < kMaxIters; ++iter) {
            bool changed = false;
            for (auto *fn : comp) {
                Walker w(*this, *fn);
                w.run();
                diags[fn] = std::move(w.diags);
                auto it = sums_.find(fn);
                Summary next = it == sums_.end() ? w.summary : joinSummary(it->second, w.summary);
                if (it == sums_.end() || !(it->second == next)) { changed = true; sums_[fn] = std::move(next); }
            }
            if (!recursive || !changed) break;
        }
        for (auto *fn : comp) for (auto &d : diags[fn]) all.push_back(std::move(d));
    }
    std::set<std::tuple<const char *, unsigned, unsigned, std::string>> seen;
    for (auto &d : all)
        if (seen.insert({d.loc.file, d.loc.line, d.loc.col, d.msg}).second) {
            diag_.error(d.loc, d.msg);
            for (auto &n : d.notes) diag_.note(n.first, n.second);
        }
}

} // namespace

void Sema::checkMemoryFlow() {
    FlowDriver(diag_, checkedFns_, *this).run();
}

} // namespace safec
