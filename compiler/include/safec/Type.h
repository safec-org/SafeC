#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include <cstdint>

namespace safec {

// ── Region kinds ──────────────────────────────────────────────────────────────
enum class Region {
    Unknown,
    Stack,   // &stack T  — lexical-scope lifetime
    Static,  // &static T — program lifetime
    Heap,    // &heap T   — explicit free required
    Arena,   // &arena<R> T — user-defined arena
    // ?&T — an "outliving reference": no region is declared or tracked at
    // all. Exists for pointers that cross the extern-function boundary
    // (an 'extern' — SafeC's only linkage form, always C-ABI — function's
    // T* parameter/return, or a value handed to one) and may be retained
    // by the extern side past the current function's return (a stored
    // callback userData pointer, an opaque handle, etc.), so neither
    // '&stack' (dies with this scope) nor '&heap'/'&static' (claims an
    // ownership/lifetime guarantee SafeC has no way to verify across the
    // ABI boundary) fit. A raw T* converts to '?&T' implicitly with no
    // 'unsafe' (see Sema::canImplicitlyConvert's Pointer→Reference rule,
    // which every region already takes this same path); reading the
    // value out requires 'match'/'is_null()'/'.default(value)' or an
    // 'unsafe'-gated '!', identical to any other nullable reference (see
    // Sema::checkNullabilityDeref) — those all already exist and don't
    // special-case Extern. checkAssign/checkCallRegions/checkRegionEscape
    // only ever test the specific Stack/Static/Heap/Arena cases above, so
    // Extern references skip every escape/lifetime check by construction
    // — "it doesn't have to determine region" falls out of the existing
    // design rather than needing new logic. Printed without a region
    // keyword at all (ReferenceType::str() special-cases this), so it
    // round-trips as exactly '?&T' (or '&T' once unwrapped to non-
    // nullable — though in practice '!'/'match'/'try' all unwrap straight
    // to the plain base type T, not a reference, so that form rarely
    // appears in practice; see those in Sema.cpp).
    Extern,
};

std::string regionName(Region r);

// ── Type kinds ────────────────────────────────────────────────────────────────
enum class TypeKind {
    Void,
    Bool,
    Char,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Pointer,    // raw C pointer (unsafe territory)
    Reference,  // SafeC region-qualified reference
    Array,
    Struct,
    Enum,
    Function,
    Generic,    // instantiated generic parameter
    Error,      // sentinel for error recovery
    Tuple,      // (T1, T2, ...) product type
    Optional,   // ?T — {T, i1} nullable wrapper
    Slice,      // []T — fat pointer {T*, i64}
    Typeof,     // typeof(expr) — resolved by Sema to concrete type
    Newtype,    // newtype Name = BaseType; — distinct type wrapper
    Vector,     // vec<T, N> — fixed-width SIMD vector, lowers to LLVM's
                // native <N x T> IR vector type (see std::simd)
};

// Forward declarations
struct Type;
using TypePtr = std::shared_ptr<Type>;

// ── Base type ─────────────────────────────────────────────────────────────────
struct Type {
    TypeKind kind;
    explicit Type(TypeKind k) : kind(k) {}
    virtual ~Type() = default;

    bool isVoid()      const { return kind == TypeKind::Void; }
    bool isBool()      const { return kind == TypeKind::Bool; }
    bool isInteger()   const;
    bool isFloat()     const { return kind == TypeKind::Float32 || kind == TypeKind::Float64; }
    bool isArithmetic() const { return isInteger() || isFloat(); }
    bool isPointer()   const { return kind == TypeKind::Pointer; }
    bool isReference() const { return kind == TypeKind::Reference; }
    bool isAggregate() const { return kind == TypeKind::Struct || kind == TypeKind::Array; }
    bool isError()     const { return kind == TypeKind::Error; }

    // Pretty-print
    virtual std::string str() const = 0;
    // Are two types the same structural type?
    virtual bool equals(const Type &other) const { return kind == other.kind; }
};

// ── Primitive types ───────────────────────────────────────────────────────────
struct PrimType : Type {
    explicit PrimType(TypeKind k) : Type(k) {
        assert(k == TypeKind::Void   || k == TypeKind::Bool   ||
               k == TypeKind::Char   ||
               k == TypeKind::Int8   || k == TypeKind::Int16  ||
               k == TypeKind::Int32  || k == TypeKind::Int64  ||
               k == TypeKind::UInt8  || k == TypeKind::UInt16 ||
               k == TypeKind::UInt32 || k == TypeKind::UInt64 ||
               k == TypeKind::Float32|| k == TypeKind::Float64||
               k == TypeKind::Error);
    }
    std::string str() const override;
    bool equals(const Type &o) const override { return kind == o.kind; }

    unsigned bitWidth() const;
    bool     isSigned() const;
};

// ── Pointer type (raw, unsafe) ────────────────────────────────────────────────
struct PointerType : Type {
    TypePtr base;
    bool    isConst = false;

    explicit PointerType(TypePtr b, bool c = false)
        : Type(TypeKind::Pointer), base(std::move(b)), isConst(c) {}

    std::string str() const override;
    bool equals(const Type &o) const override;
};

// ── Reference type (SafeC, region-qualified) ──────────────────────────────────
// &stack T, &heap T, &static T, &arena<R> T
// Nullable: ?&stack T
// Lowers to: T* + nonnull/noalias/dereferenceable attributes
struct ReferenceType : Type {
    TypePtr  base;
    Region   region  = Region::Unknown;
    bool     nullable = false;  // true → ?&T
    bool     mut      = true;   // mutable by default
    std::string arenaName;      // if region == Arena: the region identifier

    ReferenceType(TypePtr b, Region r, bool nullable = false,
                  bool mut = true, std::string arena = {})
        : Type(TypeKind::Reference), base(std::move(b)), region(r),
          nullable(nullable), mut(mut), arenaName(std::move(arena)) {}

    std::string str() const override;
    bool equals(const Type &o) const override;
};

// ── Array type ────────────────────────────────────────────────────────────────
struct ArrayType : Type {
    TypePtr  element;
    int64_t  size = -1;  // -1 = unsized (pointer decay compatible), or unresolved
                         // (see sizeExpr) until the post-parse const-eval pass runs

    // Set when the array-size expression in source wasn't a plain literal
    // (e.g. a named constant or a consteval function call like 'square(3)').
    // Untyped (Expr*) to avoid a circular header dependency with AST.h, same
    // pattern as TypeofType::expr. Non-owning: the TranslationUnit's
    // arraySizeExprs list owns the node. Resolved into 'size' by
    // resolveArraySizes() (see main.cpp) before Sema runs; nulled out once resolved.
    void *sizeExpr = nullptr;

    ArrayType(TypePtr elem, int64_t sz = -1)
        : Type(TypeKind::Array), element(std::move(elem)), size(sz) {}

    std::string str() const override;
    bool equals(const Type &o) const override;
};

// ── Struct field ──────────────────────────────────────────────────────────────
struct FieldDecl {
    std::string name;
    TypePtr     type;
    int         index = 0;
    // Bitfield support ('unsigned x : 4;'): bitWidth == -1 means "not a
    // bitfield" (the common case). When >= 0, this field shares LLVM struct
    // slot 'index' with any other bitfields packed into the same storage
    // unit (consecutive same-typed bitfields are packed together, LSB
    // first — see Sema::collectStruct), and 'bitOffset' is this field's bit
    // position within that shared unit. CodeGen reads/writes it via
    // shift+mask on the whole unit rather than a plain GEP+load/store.
    int         bitWidth  = -1;
    int         bitOffset = 0;
    // Anonymous struct/union member ('struct S { union { int a; float b; }; };')
    // — 'type' is the anonymous inline struct/union's (synthesized-name)
    // StructType, and unqualified access to its members ('s.a', 's.b')
    // should transparently reach through this field. See
    // StructType::findFieldPath, which does the recursive search.
    bool        isAnonymous = false;
};

// ── Struct type ───────────────────────────────────────────────────────────────
struct StructType : Type {
    std::string            name;
    std::vector<FieldDecl> fields;
    bool                   isUnion   = false;   // tagged union
    bool                   isDefined = false;   // forward-declared?
    bool                   isPacked  = false;   // packed: no alignment padding
    bool                   isTaggedUnion = false; // true for union decls (tag + payload)
    int                    maxPayloadSize = 0;    // max(sizeof each variant) in bytes
    // Non-empty only for an *unresolved* reference to a generic struct
    // template, e.g. 'struct Box<int>' parses to a StructType named "Box"
    // with typeArgs=[int] — Sema::resolveType monomorphizes it (see
    // Sema::instantiateGenericStruct) into a concrete StructType (name
    // becomes the mangled "Box_int") whose own typeArgs is empty, same
    // as any ordinary struct. Never set on a StructDecl's own '.type'.
    std::vector<TypePtr>  typeArgs;

    explicit StructType(std::string n, bool isUnion = false)
        : Type(TypeKind::Struct), name(std::move(n)), isUnion(isUnion) {}

    const FieldDecl *findField(const std::string &fname) const;
    // Recursive lookup reaching through anonymous struct/union members
    // ('struct S { union { int a; }; };' — 's.a' isn't a direct field of
    // S, it's a field of S's anonymous union member). Returns the matched
    // FieldDecl (its type is the *innermost* field's real type) and fills
    // 'outPath' with the GEP index chain from S down to it — [outer field's
    // index] for a direct field, or [anon field's index, ..., inner field's
    // index] when it's reached through one or more anonymous members.
    // Returns nullptr (path unspecified) if not found anywhere.
    const FieldDecl *findFieldPath(const std::string &fname,
                                   std::vector<int> &outPath) const;
    std::string str() const override { return name; }
    bool equals(const Type &o) const override;
};

// ── Enum type ─────────────────────────────────────────────────────────────────
struct EnumType : Type {
    std::string name;
    std::vector<std::pair<std::string, int64_t>> enumerators;
    int bitWidth = 32;   // underlying type width: 8/16/32/64
    bool isSigned = true; // signed or unsigned

    explicit EnumType(std::string n)
        : Type(TypeKind::Enum), name(std::move(n)) {}

    std::string str() const override { return name; }
    bool equals(const Type &o) const override;
};

// ── Function type ─────────────────────────────────────────────────────────────
struct FunctionType : Type {
    TypePtr              returnType;
    std::vector<TypePtr> paramTypes;
    bool                 variadic = false;

    FunctionType(TypePtr ret, std::vector<TypePtr> params, bool va = false)
        : Type(TypeKind::Function), returnType(std::move(ret)),
          paramTypes(std::move(params)), variadic(va) {}

    std::string str() const override;
    bool equals(const Type &o) const override;
};

// ── Generic placeholder ───────────────────────────────────────────────────────
struct GenericType : Type {
    std::string name;                     // e.g., "T"
    std::vector<std::string> constraints; // e.g., {"Numeric", "Indexed"}

    GenericType(std::string n, std::vector<std::string> c = {})
        : Type(TypeKind::Generic), name(std::move(n)), constraints(std::move(c)) {}

    std::string str() const override { return name; }
    bool equals(const Type &o) const override;
};

// ── Tuple type ────────────────────────────────────────────────────────────────
struct TupleType : Type {
    std::vector<TypePtr> elementTypes;
    explicit TupleType(std::vector<TypePtr> e)
        : Type(TypeKind::Tuple), elementTypes(std::move(e)) {}
    std::string str() const override;
    bool equals(const Type &o) const override;
    int bitWidth() const { return 0; }
};
TypePtr makeTuple(std::vector<TypePtr> elems);

// ── SIMD vector type: vec<T, N> ────────────────────────────────────────────────
// A fixed-width homogeneous vector of N scalar elements, lowering directly to
// LLVM's native <N x T> vector IR type — the actual per-ISA instruction
// selection (SSE/AVX on x86_64, NEON on AArch64, the V extension on RISC-V,
// SIMD128 on WebAssembly) is entirely LLVM's job once the operations are
// expressed as ordinary vector-typed arithmetic; std::simd is a thin,
// portable naming layer over this, not per-architecture hand-written code.
struct VectorType : Type {
    TypePtr element;
    int     width;
    VectorType(TypePtr e, int w)
        : Type(TypeKind::Vector), element(std::move(e)), width(w) {}
    std::string str() const override;
    bool equals(const Type &o) const override;
};
TypePtr makeVector(TypePtr element, int width);

// ── Optional type ?T ──────────────────────────────────────────────────────────
// Lowers to LLVM struct { T, i1 }  (value, has_value)
struct OptionalType : Type {
    TypePtr inner;
    explicit OptionalType(TypePtr t)
        : Type(TypeKind::Optional), inner(std::move(t)) {}
    std::string str() const override { return "?" + inner->str(); }
    bool equals(const Type &o) const override;
};

// ── Slice type []T ────────────────────────────────────────────────────────────
// Lowers to { T*, i64 }  (parse-only for now; codegen to be added)
struct SliceType : Type {
    TypePtr element;
    explicit SliceType(TypePtr elem)
        : Type(TypeKind::Slice), element(std::move(elem)) {}
    std::string str() const override { return "[]" + element->str(); }
    bool equals(const Type &o) const override;
};

// ── Newtype (distinct type wrapper) ──────────────────────────────────────
struct NewtypeType : Type {
    std::string name;
    TypePtr     base;
    explicit NewtypeType(std::string n, TypePtr b)
        : Type(TypeKind::Newtype), name(std::move(n)), base(std::move(b)) {}
    std::string str() const override { return name; }
    bool equals(const Type &o) const override;
};

// ── Typeof type (resolved by Sema to concrete type) ──────────────────────
// Stores a raw pointer to the Expr node (owned by the AST).
// Sema resolves this to a concrete type; codegen never sees it.
struct TypeofType : Type {
    void *expr = nullptr;  // Expr* — untyped to avoid circular header dependency
    explicit TypeofType(void *e)
        : Type(TypeKind::Typeof), expr(e) {}
    std::string str() const override { return "typeof(...)"; }
    bool equals(const Type &o) const override { return false; }
};

// ── Type factory / interning helpers ──────────────────────────────────────────
TypePtr makeVoid();
TypePtr makeBool();
TypePtr makeChar();
TypePtr makeInt(unsigned bits, bool isSigned = true);
TypePtr makeFloat(unsigned bits);
TypePtr makeError();
TypePtr makePointer(TypePtr base, bool isConst = false);
TypePtr makeReference(TypePtr base, Region r, bool nullable = false,
                      bool mut = true, std::string arena = {});
// 'sizeExpr' is non-null when the source size wasn't a plain literal (see
// ArrayType::sizeExpr); it's resolved into a concrete 'size' by
// resolveArraySizes() before Sema runs.
TypePtr makeArray(TypePtr elem, int64_t sz = -1, void *sizeExpr = nullptr);
TypePtr makeFunction(TypePtr ret, std::vector<TypePtr> params, bool va = false);
TypePtr makeOptional(TypePtr inner);
TypePtr makeSlice(TypePtr element);

// Is 'src' implicitly convertible to 'dst' under SafeC rules?
// (Very strict — only same-type or numeric widening in explicit cast context)
bool typeEqual(const TypePtr &a, const TypePtr &b);
bool typeCompatibleAssign(const TypePtr &dst, const TypePtr &src);

} // namespace safec
