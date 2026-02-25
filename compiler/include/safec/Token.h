#pragma once
#include "safec/Diagnostic.h"
#include <string>
#include <cstdint>

namespace safec {

// ── Token kinds ───────────────────────────────────────────────────────────────
enum class TK {
    // Literals
    IntLit, FloatLit, StringLit, CharLit,

    // Identifier
    Ident,

    // ── C keywords ────────────────────────────────────────────────────────────
    KW_auto, KW_break, KW_case, KW_char, KW_const,
    KW_continue, KW_default, KW_do, KW_double, KW_else,
    KW_enum, KW_extern, KW_float, KW_for, KW_goto,
    KW_if, KW_inline, KW_int, KW_long, KW_register,
    KW_restrict, KW_return, KW_short, KW_signed,
    KW_sizeof, KW_static, KW_struct, KW_switch,
    KW_typedef, KW_union, KW_unsigned, KW_void,
    KW_volatile, KW_while, KW_bool, KW_true, KW_false,
    KW_null,

    // ── SafeC extension keywords ──────────────────────────────────────────────
    KW_region,        // region AudioPool { capacity: 4096 }
    KW_unsafe,        // unsafe { ... }
    KW_consteval,     // consteval void generate_table()
    KW_generic,       // generic<T: Numeric>
    KW_static_assert, // static_assert(cond, "msg")
    KW_stack,         // &stack T   (contextual keyword / also valid ident)
    KW_heap,          // &heap T
    KW_arena,         // &arena<R> T
    KW_capacity,      // region field keyword
    KW_self,          // self — implicit receiver in methods
    KW_operator,      // operator overloading
    KW_new,           // new<R> T  (arena allocation)
    KW_arena_reset,   // arena_reset<R>()
    KW_tuple,         // tuple(T1, T2, ...)
    KW_spawn,         // spawn<R> closure
    KW_join,          // join(handle)
    KW_defer,         // defer stmt
    KW_errdefer,      // errdefer stmt (deferred only on error path)
    KW_match,         // match (expr) { pattern => stmt, ... }
    KW_packed,        // packed struct { ... }
    KW_try,           // try expr  (propagate null optional)
    KW_must_use,      // must_use fn — warn on discarded return value
    KW_fn,            // fn ReturnType(Params) name — function pointer type
    KW_alignof,       // alignof(T)
    KW_typeof,        // typeof(expr) — type position
    KW_fieldcount,    // fieldcount(T) — struct field count

    // ── Operators ─────────────────────────────────────────────────────────────
    Plus, Minus, Star, Slash, Percent,
    Amp, Pipe, Caret, Tilde,
    LShift, RShift,
    PlusPlus, MinusMinus,
    PlusEq, MinusEq, StarEq, SlashEq, PercentEq,
    AmpEq, PipeEq, CaretEq, LShiftEq, RShiftEq,
    AmpAmp, PipePipe, Bang,
    EqEq, BangEq,
    Lt, Gt, LtEq, GtEq,
    Eq,
    Arrow, Dot, DotDotDot,
    Question,    // ?
    QuestionAmp, // ?&  (nullable reference)
    Colon,
    ColonColon,  // ::  (method scope resolution)
    FatArrow,    // =>  (match arm separator)

    // ── Punctuation ───────────────────────────────────────────────────────────
    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,
    Semicolon, Comma, Hash,

    // ── Special ───────────────────────────────────────────────────────────────
    Eof, Invalid
};

struct Token {
    TK             kind   = TK::Invalid;
    std::string    text;
    SourceLocation loc;

    // Numeric payload
    int64_t  intVal      = 0;
    double   floatVal    = 0.0;
    bool     isLongLong  = false;  // LL/ll suffix
    bool     isUnsigned  = false;  // U/u suffix

    Token() = default;
    Token(TK k, std::string t, SourceLocation l)
        : kind(k), text(std::move(t)), loc(l) {}

    bool is (TK k) const { return kind == k; }
    bool isNot(TK k) const { return kind != k; }
    bool isIdent(const char *name) const {
        return kind == TK::Ident && text == name;
    }
    bool isEof() const { return kind == TK::Eof; }

    // Contextual keyword check: identifiers that serve as keywords in context
    bool isContextualKW(const char *name) const {
        return (kind == TK::Ident || isContextualKindOf(kind)) && text == name;
    }

    const char *kindName() const;

private:
    static bool isContextualKindOf(TK k) {
        return k == TK::KW_stack || k == TK::KW_heap || k == TK::KW_arena || k == TK::KW_capacity;
    }
};

} // namespace safec
