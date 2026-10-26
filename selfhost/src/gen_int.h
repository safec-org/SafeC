// gen_int.h — backend functions shared by gen.sc and genx.sc.
#pragma once
#include "scc.h"

#define LS_WASM     1
#define LS_FRAME    2
#define LS_STATIC   3
#define LS_INDIRECT 4

extern int g_pw;
extern int g_fp;
extern int g_sret;
extern int g_vaarg;
extern int g_cdepth;
extern int g_gen_ret;
extern int g_lst_kind[NLOCAL_MAX];
extern int g_lst_val[NLOCAL_MAX];
extern int g_ndf;
extern int g_rt_trap;

int new_wl(int vt);
int alloc_frame(int size, int al);
int alloc_tmp(int t);
void emit_frame_addr(int off);
int base_scalar(int t);
int vt_of(int t);
int is_agg(int t);
int is_ptr_opt(int t);
int pw_of(int t);
int ty_unsigned(int t);
int ty_binsigned(int t);
void emit_load(int t, int off);
void emit_store(int t, int off);
void emit_zero(int t);
void emit_drop_ty(int t);
void emit_trap(int code);
int func_index(int f);
int table_index(int f);
int sig_of_fn_type(int ft);
void push_ftparam(int vt);
int str_addr(int sid);
int region_buf(int r);
int region_off(int r);
int glob_addr(int g);
void setup_local(int l);
void emit_local_addr(int l);
void gen_init_at(int al, int off, int t, int e);
void ctl_open(int op);
void ctl_close();
void emit_br_to(int depth);
void push_loop(int brk, int cont, int label, int ismatch);
void gen_stmt(int s);
void gen_try_fail();
void cs_b(int b);
void cs_u(long v);
void cs_s(long v);
void add_import_sid(int sid, int f);

void gen_expr(int e);
void gen_value_as(int e, int t);
void gen_expr_discard(int e);
void gen_cond(int e);
void gen_addr(int e);
void gen_match(int s, int is_expr);
void conv(int from, int pw, int to);
long f32_bits_of_f64(long b);
long f64_bits(double d);
extern int g_nlp;
