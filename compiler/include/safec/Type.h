#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cassert>

namespace safec {

// ── Region kinds ──────────────────────────────────────────────────────────────
enum class Region {
    Unknown,
    Stack,   // &stack T  — lexical-scope lifetime
    Static,  // &static T — program lifetime
    Heap,    // &heap T   — explicit free required
    Arena,   // &arena<R> T — user-defined arena
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
    int64_t  size = -1;  // -1 = unsized (pointer decay compatible)

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

    explicit StructType(std::string n, bool isUnion = false)
        : Type(TypeKind::Struct), name(std::move(n)), isUnion(isUnion) {}

    const FieldDecl *findField(const std::string &fname) const;
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
    std::string name;       // e.g., "T"
    std::string constraint; // e.g., "Numeric"

    GenericType(std::string n, std::string c = {})
        : Type(TypeKind::Generic), name(std::move(n)), constraint(std::move(c)) {}

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
TypePtr makeArray(TypePtr elem, int64_t sz = -1);
TypePtr makeFunction(TypePtr ret, std::vector<TypePtr> params, bool va = false);
TypePtr makeOptional(TypePtr inner);
TypePtr makeSlice(TypePtr element);

// Is 'src' implicitly convertible to 'dst' under SafeC rules?
// (Very strict — only same-type or numeric widening in explicit cast context)
bool typeEqual(const TypePtr &a, const TypePtr &b);
bool typeCompatibleAssign(const TypePtr &dst, const TypePtr &src);

} // namespace safec
