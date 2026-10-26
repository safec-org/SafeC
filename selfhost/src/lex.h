// lex.h — lexer internals shared with the preprocessor.
#pragma once
#include "scc.h"

extern struct Tok g_ptok[RAW_MAX];
extern int g_nptok;

int lex_file(int file);
long f64_bits(double d);
int is_ident_start(int c);
int is_ident_char(int c);
int is_digit(int c);
void tmp_punct(int k);
void tmp_tok_spelling(int kind, int sid);
int push_ptok(int kind, int sid, int line, int col, int flags, long ival);
int punct1(int c);
int punct2(int c0, int c1);
int punct3(int c0, int c1, int c2);
long f32_bits_of_f64(long b);
