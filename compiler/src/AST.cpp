#include "safec/AST.h"
#include <sstream>

namespace safec {

// ── FunctionDecl::funcType() ──────────────────────────────────────────────────
TypePtr FunctionDecl::funcType() const {
    std::vector<TypePtr> paramTys;
    for (auto &p : params) paramTys.push_back(p.type);
    return makeFunction(returnType, std::move(paramTys), isVariadic);
}

} // namespace safec
