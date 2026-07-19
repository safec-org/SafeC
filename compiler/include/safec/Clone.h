#pragma once
#include "safec/AST.h"
#include "safec/Type.h"
#include <unordered_map>
#include <memory>
#include <string>

namespace safec {

// ── Type substitution map: generic name → concrete type ──────────────────────
using TypeSubst = std::unordered_map<std::string, TypePtr>;

// Recursively substitute GenericType nodes in a type tree.
TypePtr substituteType(const TypePtr &ty, const TypeSubst &subs);

// Deep-clone an expression / statement, substituting generic types.
ExprPtr cloneExpr(const ExprPtr &e, const TypeSubst &subs);
StmtPtr cloneStmt(const StmtPtr &s, const TypeSubst &subs);

// Clone a FunctionDecl, substituting all generic type references.
// The caller must clear genericParams and rename the result.
std::unique_ptr<FunctionDecl> cloneFunctionDecl(const FunctionDecl &fn,
                                                  const TypeSubst &subs);

// Clone a StructDecl, substituting all generic type references in its
// fields and struct-internal method forward-declarations. Assumes the
// template's field/method types have already been resolved against its
// own genericParams (GenericType nodes in place of bare names — see
// Sema::collectStruct's generic-template branch, which does this once
// when the template is first collected). The caller must clear
// genericParams and rename the result (same contract as
// cloneFunctionDecl).
std::unique_ptr<StructDecl> cloneStructDecl(const StructDecl &sd,
                                             const TypeSubst &subs);

} // namespace safec
