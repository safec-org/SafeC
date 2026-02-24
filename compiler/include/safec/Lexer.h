#pragma once
#include "safec/Token.h"
#include "safec/Diagnostic.h"
#include <string>
#include <vector>

namespace safec {

class Lexer {
public:
    Lexer(std::string source, const char *filename, DiagEngine &diag);

    // Lex all tokens (including EOF)
    std::vector<Token> lexAll();

    // Single token advance (used by the streaming interface)
    Token next();

private:
    // Character helpers
    char  peek(int offset = 0) const;
    char  advance();
    bool  atEnd() const { return pos_ >= src_.size(); }
    void  skipWhitespaceAndComments();
    bool  skipLineComment();
    bool  skipBlockComment();

    // Lexing sub-routines
    Token lexNumber();
    Token lexString();
    Token lexChar();
    Token lexIdOrKeyword();
    Token lexPunct();

    // Helpers
    SourceLocation curLoc() const { return {line_, col_, filename_}; }
    Token make(TK k, std::string text) { return {k, std::move(text), startLoc_}; }
    Token makeSimple(TK k, std::string text) {
        auto t = Token{k, std::move(text), curLoc()};
        return t;
    }

    static TK keywordKind(const std::string &word);

    std::string    src_;
    const char    *filename_;
    DiagEngine    &diag_;
    size_t         pos_  = 0;
    unsigned       line_ = 1;
    unsigned       col_  = 1;
    SourceLocation startLoc_;
};

} // namespace safec
