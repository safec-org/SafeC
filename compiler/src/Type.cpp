#include "safec/Type.h"
#include <sstream>
#include <cassert>

namespace safec {

// ── Region name ───────────────────────────────────────────────────────────────
std::string regionName(Region r) {
    switch (r) {
    case Region::Stack:  return "stack";
    case Region::Static: return "static";
    case Region::Heap:   return "heap";
    case Region::Arena:  return "arena";
    default:             return "?";
    }
}

// ── PrimType ──────────────────────────────────────────────────────────────────
bool Type::isInteger() const {
    return kind == TypeKind::Bool   || kind == TypeKind::Char   ||
           kind == TypeKind::Int8   || kind == TypeKind::Int16  ||
           kind == TypeKind::Int32  || kind == TypeKind::Int64  ||
           kind == TypeKind::UInt8  || kind == TypeKind::UInt16 ||
           kind == TypeKind::UInt32 || kind == TypeKind::UInt64;
}

std::string PrimType::str() const {
    switch (kind) {
    case TypeKind::Void:    return "void";
    case TypeKind::Bool:    return "bool";
    case TypeKind::Char:    return "char";
    case TypeKind::Int8:    return "int8";
    case TypeKind::Int16:   return "int16";
    case TypeKind::Int32:   return "int";
    case TypeKind::Int64:   return "long";
    case TypeKind::UInt8:   return "uint8";
    case TypeKind::UInt16:  return "uint16";
    case TypeKind::UInt32:  return "unsigned int";
    case TypeKind::UInt64:  return "unsigned long";
    case TypeKind::Float32: return "float";
    case TypeKind::Float64: return "double";
    case TypeKind::Error:   return "<error>";
    default:                return "?";
    }
}

unsigned PrimType::bitWidth() const {
    switch (kind) {
    case TypeKind::Bool:
    case TypeKind::Int8:
    case TypeKind::UInt8:   return 8;
    case TypeKind::Int16:
    case TypeKind::UInt16:  return 16;
    case TypeKind::Int32:
    case TypeKind::UInt32:
    case TypeKind::Float32: return 32;
    case TypeKind::Char:    return 8;
    case TypeKind::Int64:
    case TypeKind::UInt64:
    case TypeKind::Float64: return 64;
    default:                return 0;
    }
}

bool PrimType::isSigned() const {
    return kind == TypeKind::Int8 || kind == TypeKind::Int16 ||
           kind == TypeKind::Int32 || kind == TypeKind::Int64 ||
           kind == TypeKind::Char;
}

// ── PointerType ───────────────────────────────────────────────────────────────
std::string PointerType::str() const {
    return (isConst ? "const " : "") + base->str() + "*";
}

bool PointerType::equals(const Type &o) const {
    if (o.kind != TypeKind::Pointer) return false;
    auto &p = static_cast<const PointerType &>(o);
    return typeEqual(base, p.base);
}

// ── ReferenceType ─────────────────────────────────────────────────────────────
std::string ReferenceType::str() const {
    std::string s = nullable ? "?&" : "&";
    if (region == Region::Arena)
        s += "arena<" + arenaName + "> ";
    else
        s += regionName(region) + " ";
    if (!mut) s += "const ";
    s += base->str();
    return s;
}

bool ReferenceType::equals(const Type &o) const {
    if (o.kind != TypeKind::Reference) return false;
    auto &r = static_cast<const ReferenceType &>(o);
    return typeEqual(base, r.base) && region == r.region &&
           nullable == r.nullable  && mut == r.mut        &&
           arenaName == r.arenaName;
}

// ── ArrayType ─────────────────────────────────────────────────────────────────
std::string ArrayType::str() const {
    if (size < 0) return element->str() + "[]";
    return element->str() + "[" + std::to_string(size) + "]";
}

bool ArrayType::equals(const Type &o) const {
    if (o.kind != TypeKind::Array) return false;
    auto &a = static_cast<const ArrayType &>(o);
    return size == a.size && typeEqual(element, a.element);
}

// ── StructType ────────────────────────────────────────────────────────────────
const FieldDecl *StructType::findField(const std::string &fname) const {
    for (auto &f : fields) if (f.name == fname) return &f;
    return nullptr;
}

bool StructType::equals(const Type &o) const {
    if (o.kind != TypeKind::Struct) return false;
    return name == static_cast<const StructType &>(o).name;
}

// ── EnumType ──────────────────────────────────────────────────────────────────
bool EnumType::equals(const Type &o) const {
    if (o.kind != TypeKind::Enum) return false;
    return name == static_cast<const EnumType &>(o).name;
}

// ── FunctionType ──────────────────────────────────────────────────────────────
std::string FunctionType::str() const {
    std::string s = returnType->str() + "(";
    for (size_t i = 0; i < paramTypes.size(); ++i) {
        if (i) s += ", ";
        s += paramTypes[i]->str();
    }
    if (variadic) { if (!paramTypes.empty()) s += ", "; s += "..."; }
    return s + ")";
}

bool FunctionType::equals(const Type &o) const {
    if (o.kind != TypeKind::Function) return false;
    auto &f = static_cast<const FunctionType &>(o);
    if (!typeEqual(returnType, f.returnType)) return false;
    if (paramTypes.size() != f.paramTypes.size()) return false;
    for (size_t i = 0; i < paramTypes.size(); ++i)
        if (!typeEqual(paramTypes[i], f.paramTypes[i])) return false;
    return variadic == f.variadic;
}

// ── GenericType ───────────────────────────────────────────────────────────────
bool GenericType::equals(const Type &o) const {
    if (o.kind != TypeKind::Generic) return false;
    return name == static_cast<const GenericType &>(o).name;
}

// ── TupleType ─────────────────────────────────────────────────────────────────
std::string TupleType::str() const {
    std::string s = "(";
    for (size_t i = 0; i < elementTypes.size(); ++i) {
        if (i) s += ", ";
        s += elementTypes[i]->str();
    }
    return s + ")";
}

bool TupleType::equals(const Type &o) const {
    if (o.kind != TypeKind::Tuple) return false;
    auto &t = static_cast<const TupleType &>(o);
    if (elementTypes.size() != t.elementTypes.size()) return false;
    for (size_t i = 0; i < elementTypes.size(); ++i)
        if (!typeEqual(elementTypes[i], t.elementTypes[i])) return false;
    return true;
}

bool OptionalType::equals(const Type &o) const {
    if (o.kind != TypeKind::Optional) return false;
    return typeEqual(inner, static_cast<const OptionalType &>(o).inner);
}

bool SliceType::equals(const Type &o) const {
    if (o.kind != TypeKind::Slice) return false;
    return typeEqual(element, static_cast<const SliceType &>(o).element);
}

// ── Comparison helpers ────────────────────────────────────────────────────────
bool typeEqual(const TypePtr &a, const TypePtr &b) {
    if (!a || !b) return a == b;
    return a->equals(*b);
}

bool typeCompatibleAssign(const TypePtr &dst, const TypePtr &src) {
    // Exact match
    if (typeEqual(dst, src)) return true;
    // Error sentinel propagates silently
    if (dst->isError() || src->isError()) return true;
    // Char ↔ Int8 ↔ UInt8: 8-bit types are mutually assignable
    // (char, signed char, unsigned char are all 8-bit — compatible in practice)
    auto is8bit = [](const TypePtr &t) {
        return t->kind == TypeKind::Char  || t->kind == TypeKind::Int8 ||
               t->kind == TypeKind::UInt8 || t->kind == TypeKind::Bool;
    };
    if (is8bit(dst) && is8bit(src)) return true;
    // Integer widening is NOT implicit in SafeC — explicit cast required
    // Reference → pointer decay allowed only in unsafe (enforced by sema)
    return false;
}

// ── Factory ───────────────────────────────────────────────────────────────────
TypePtr makeVoid()               { return std::make_shared<PrimType>(TypeKind::Void); }
TypePtr makeBool()               { return std::make_shared<PrimType>(TypeKind::Bool); }
TypePtr makeChar()               { return std::make_shared<PrimType>(TypeKind::Char); }
TypePtr makeError()              { return std::make_shared<PrimType>(TypeKind::Error); }

TypePtr makeInt(unsigned bits, bool isSigned) {
    TypeKind k;
    if (isSigned) {
        switch (bits) {
        case 8:  k = TypeKind::Int8;  break;
        case 16: k = TypeKind::Int16; break;
        case 32: k = TypeKind::Int32; break;
        case 64: k = TypeKind::Int64; break;
        default: k = TypeKind::Int32;
        }
    } else {
        switch (bits) {
        case 8:  k = TypeKind::UInt8;  break;
        case 16: k = TypeKind::UInt16; break;
        case 32: k = TypeKind::UInt32; break;
        case 64: k = TypeKind::UInt64; break;
        default: k = TypeKind::UInt32;
        }
    }
    return std::make_shared<PrimType>(k);
}

TypePtr makeFloat(unsigned bits) {
    return std::make_shared<PrimType>(bits == 32 ? TypeKind::Float32 : TypeKind::Float64);
}

TypePtr makePointer(TypePtr base, bool isConst) {
    return std::make_shared<PointerType>(std::move(base), isConst);
}

TypePtr makeReference(TypePtr base, Region r, bool nullable, bool mut, std::string arena) {
    return std::make_shared<ReferenceType>(std::move(base), r, nullable, mut, std::move(arena));
}

TypePtr makeArray(TypePtr elem, int64_t sz) {
    return std::make_shared<ArrayType>(std::move(elem), sz);
}

TypePtr makeFunction(TypePtr ret, std::vector<TypePtr> params, bool va) {
    return std::make_shared<FunctionType>(std::move(ret), std::move(params), va);
}

TypePtr makeTuple(std::vector<TypePtr> elems) {
    return std::make_shared<TupleType>(std::move(elems));
}

TypePtr makeOptional(TypePtr inner) {
    return std::make_shared<OptionalType>(std::move(inner));
}

TypePtr makeSlice(TypePtr element) {
    return std::make_shared<SliceType>(std::move(element));
}

} // namespace safec
