// gen.h — WebAssembly backend internals.
#pragma once
#include "scc.h"

#define VT_I32 127
#define VT_I64 126
#define VT_F32 125
#define VT_F64 124

// wasm opcodes used by the backend
#define OP_UNREACHABLE 0
#define OP_NOP 1
#define OP_BLOCK 2
#define OP_LOOP 3
#define OP_IF 4
#define OP_ELSE 5
#define OP_END 11
#define OP_BR 12
#define OP_BR_IF 13
#define OP_RETURN 15
#define OP_CALL 16
#define OP_CALL_INDIRECT 17
#define OP_DROP 26
#define OP_SELECT 27
#define OP_LOCAL_GET 32
#define OP_LOCAL_SET 33
#define OP_LOCAL_TEE 34
#define OP_GLOBAL_GET 35
#define OP_GLOBAL_SET 36
#define OP_I32_LOAD 40
#define OP_I64_LOAD 41
#define OP_F32_LOAD 42
#define OP_F64_LOAD 43
#define OP_I32_LOAD8_S 44
#define OP_I32_LOAD8_U 45
#define OP_I32_LOAD16_S 46
#define OP_I32_LOAD16_U 47
#define OP_I32_STORE 54
#define OP_I64_STORE 55
#define OP_F32_STORE 56
#define OP_F64_STORE 57
#define OP_I32_STORE8 58
#define OP_I32_STORE16 59
#define OP_MEMORY_SIZE 63
#define OP_MEMORY_GROW 64
#define OP_I32_CONST 65
#define OP_I64_CONST 66
#define OP_F32_CONST 67
#define OP_F64_CONST 68
#define OP_I32_EQZ 69
#define OP_I32_EQ 70
#define OP_I32_NE 71
#define OP_I32_LT_S 72
#define OP_I32_LT_U 73
#define OP_I32_GT_S 74
#define OP_I32_GT_U 75
#define OP_I32_LE_S 76
#define OP_I32_LE_U 77
#define OP_I32_GE_S 78
#define OP_I32_GE_U 79
#define OP_I64_EQZ 80
#define OP_I64_EQ 81
#define OP_I64_NE 82
#define OP_I64_LT_S 83
#define OP_I64_LT_U 84
#define OP_I64_GT_S 85
#define OP_I64_GT_U 86
#define OP_I64_LE_S 87
#define OP_I64_LE_U 88
#define OP_I64_GE_S 89
#define OP_I64_GE_U 90
#define OP_F32_EQ 91
#define OP_F32_NE 92
#define OP_F32_LT 93
#define OP_F32_GT 94
#define OP_F32_LE 95
#define OP_F32_GE 96
#define OP_F64_EQ 97
#define OP_F64_NE 98
#define OP_F64_LT 99
#define OP_F64_GT 100
#define OP_F64_LE 101
#define OP_F64_GE 102
#define OP_I32_CLZ 103
#define OP_I32_CTZ 104
#define OP_I32_POPCNT 105
#define OP_I32_ADD 106
#define OP_I32_SUB 107
#define OP_I32_MUL 108
#define OP_I32_DIV_S 109
#define OP_I32_DIV_U 110
#define OP_I32_REM_S 111
#define OP_I32_REM_U 112
#define OP_I32_AND 113
#define OP_I32_OR 114
#define OP_I32_XOR 115
#define OP_I32_SHL 116
#define OP_I32_SHR_S 117
#define OP_I32_SHR_U 118
#define OP_I64_CLZ 121
#define OP_I64_CTZ 122
#define OP_I64_POPCNT 123
#define OP_I64_ADD 124
#define OP_I64_SUB 125
#define OP_I64_MUL 126
#define OP_I64_DIV_S 127
#define OP_I64_DIV_U 128
#define OP_I64_REM_S 129
#define OP_I64_REM_U 130
#define OP_I64_AND 131
#define OP_I64_OR 132
#define OP_I64_XOR 133
#define OP_I64_SHL 134
#define OP_I64_SHR_S 135
#define OP_I64_SHR_U 136
#define OP_F32_ABS 139
#define OP_F32_NEG 140
#define OP_F32_CEIL 141
#define OP_F32_FLOOR 142
#define OP_F32_TRUNC 143
#define OP_F32_NEAREST 144
#define OP_F32_SQRT 145
#define OP_F32_ADD 146
#define OP_F32_SUB 147
#define OP_F32_MUL 148
#define OP_F32_DIV 149
#define OP_F64_ABS 153
#define OP_F64_NEG 154
#define OP_F64_CEIL 155
#define OP_F64_FLOOR 156
#define OP_F64_TRUNC 157
#define OP_F64_NEAREST 158
#define OP_F64_SQRT 159
#define OP_F64_ADD 160
#define OP_F64_SUB 161
#define OP_F64_MUL 162
#define OP_F64_DIV 163
#define OP_I32_WRAP_I64 167
#define OP_I64_EXTEND_I32_S 172
#define OP_I64_EXTEND_I32_U 173
#define OP_F32_CONVERT_I32_S 178
#define OP_F32_CONVERT_I32_U 179
#define OP_F32_CONVERT_I64_S 180
#define OP_F32_CONVERT_I64_U 181
#define OP_F32_DEMOTE_F64 182
#define OP_F64_CONVERT_I32_S 183
#define OP_F64_CONVERT_I32_U 184
#define OP_F64_CONVERT_I64_S 185
#define OP_F64_CONVERT_I64_U 186
#define OP_F64_PROMOTE_F32 187
#define OP_I64_REINTERPRET_F64 189
#define OP_F64_REINTERPRET_I64 191
#define OP_I32_EXTEND8_S 192
#define OP_I32_EXTEND16_S 193
#define OP_PREFIX_FC 252

// wasm-level layout
#define DATA_BASE   1024
#define STACK_SIZE  1048576

extern unsigned char g_code[CODE_MAX];
extern int g_code_len;

void emit(int b);
void emitu(long v);
void emits(long v);
void emit_i32(int v);
void emit_i64(long v);
void emit_f64_bits(long bits);
void emit_mem(int op, int align, int off);
void emit_fc(int sub);
int func_type_index(int mark, int nres, int res);
void wasm_assemble();

extern int g_ftype_param[NFUNC_MAX * 4];
extern int g_nftype_param;
extern int g_ftype_p0[NFUNC_MAX];
extern int g_ftype_np[NFUNC_MAX];
extern int g_ftype_res[NFUNC_MAX];
extern int g_nftype;

extern int g_wfunc[NFUNC_MAX];    // defined wasm functions in index order (func ids)
extern int g_nwfunc;
extern int g_wfunc_type[NFUNC_MAX];
extern int g_wfunc_body[NFUNC_MAX];     // start in g_codesec
extern int g_wfunc_blen[NFUNC_MAX];
extern unsigned char g_codesec[OUT_MAX];
extern int g_codesec_len;
extern unsigned char g_data[DATA_MAX];
extern int g_data_len;
extern int g_nimports;
extern int g_import_name[16];
extern int g_import_type[16];
extern int g_table[NFUNC_MAX];
extern int g_ntable;
extern int g_heap_base;
extern int g_stack_top;
extern int g_export_sid[256];
extern int g_export_func[256];
extern int g_nexport;
extern int g_stack_limit;
